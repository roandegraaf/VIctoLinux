//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]); with `VICTORY_LAZY_COPY_BACK=1` they
//! stay here until the title's CPU touches them ([`crate::owned`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, from an offset on,
    /// covering the ranges its shaders write: what the GPU wrote differs.
    before: Mutex<Option<(usize, Vec<u8>)>>,
    /// The same with GPU-owned blocks ([`crate::owned`]): by 64 KiB block
    /// of title memory, the block's part of the copy, for the blocks of the
    /// ranges that were the title's when the submission was prepared.
    pieces: Mutex<BTreeMap<usize, Vec<u8>>>,
}

/// Block snapshots kept for reuse.
static PIECES: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// Returns block snapshots for reuse (256 MB of them at most: fresh ones
/// cost a page fault per 4 KiB, see [`Shadow::writing`]).
fn pieces_done(pieces: impl IntoIterator<Item = Vec<u8>>) {
    let mut pool = PIECES.lock().unwrap_or_else(|e| e.into_inner());
    for piece in pieces {
        if pool.len() < 4096 {
            pool.push(piece);
        }
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool (at most 8 kept, the largest).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    if pool.len() > 8 {
        pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        pool.truncate(8);
    }
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
            pieces: Mutex::default(),
        }
    }

    /// What tells this copy from others.
    fn id(&self) -> usize {
        std::ptr::from_ref(self) as usize
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, grown to cover every range the
    /// submission declares (the world declares overlapping views of its
    /// 174 MB heap dozens of times), in reused memory (fresh snapshots cost
    /// ~2.3 million page faults per world frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        if crate::owned::enabled() {
            return self.writing_owned(address, len);
        }
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
        let (from, to) = self.inside(address, len);
        // SAFETY: inside the mapping.
        let host = |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (start, end) = match &*before {
            Some((at, bytes)) => ((*at).min(from), (at + bytes.len()).max(to)),
            None => (from, to),
        };
        if let Some((at, bytes)) = &*before
            && (*at, at + bytes.len()) == (start, end)
        {
            return;
        }
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(end - start);
        match before.take() {
            Some((at, bytes)) => {
                grown.extend_from_slice(host(start, at));
                grown.extend_from_slice(&bytes);
                grown.extend_from_slice(host(at + bytes.len(), end));
                give_back(bytes);
            }
            None => grown.extend_from_slice(host(start, end)),
        }
        *before = Some((start, grown));
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    ///
    /// With GPU-owned blocks ([`crate::owned`]) nothing is copied where the
    /// title did not write meanwhile: those blocks become the GPU's, unless
    /// they overlap `foreign` (ranges of title memory written otherwise by
    /// this submission: through other copies, or at the bottom of the pipe).
    pub(crate) fn copy_back(self: &Arc<Self>, address: usize, len: usize, foreign: &[(usize, usize)]) -> usize {
        if crate::owned::enabled() {
            return self.copy_back_owned(address, len, foreign);
        }
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some((taken, before)) = &*snapshot
            && *taken <= from
            && to <= taken + before.len()
            && from < to
        {
            let before = &before[from - taken..to - taken];
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut title = vec![0u8; PIECE];
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                if let Some(bytes) = self.merge(at, &before[at - from..at - from + n], &mut title[..n]) {
                    written += bytes;
                    synced.push((start + at) / PAGE);
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last
                && !synced.is_empty()
                && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
            {
                let base = start / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                    copied[page - base] = now;
                }
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (address, address + len));
        if pending.is_empty()
            && let Some((_, bytes)) = snapshot.take()
        {
            give_back(bytes);
        }
        written
    }

    /// One piece of [`Shadow::copy_back`]: the copy from offset `at` on,
    /// compared with `old`, its bytes before the submission. Where they
    /// differ title memory (read into `title`, of `old`'s length) gets the
    /// GPU's bytes, and the copy takes the title's in turn. Returns the
    /// bytes written to title memory, or `None` if the GPU wrote nothing
    /// there or the title's memory cannot be read.
    fn merge(&self, at: usize, old: &[u8], title: &mut [u8]) -> Option<usize> {
        let (start, n) = (self.start as usize, old.len());
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
        if host == old || !crate::host::read_into((start + at) as *const u8, title) {
            return None;
        }
        // Title memory with the GPU's bytes put in; the copy takes the
        // title's bytes in turn.
        let mut changed = false;
        for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
            if *h != *o && *t != *h {
                *t = *h;
                changed = true;
            }
            *h = *t;
        }
        if !changed {
            return Some(0);
        }
        crate::watch::writing(start + at, n);
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        Some(n)
    }

    /// The copy as [`crate::owned`] knows it.
    fn source(&self) -> crate::owned::Source {
        crate::owned::Source {
            id: self.id(),
            start: self.start as usize,
            len: self.len as usize,
            data: self.data as usize,
        }
    }

    /// The part of title block `block` the copy holds, as offsets in it.
    fn part(&self, block: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = (block * PAGE).max(start);
        (from - start, ((block + 1) * PAGE).min(end).max(from) - start)
    }

    /// [`Shadow::writing`] with GPU-owned blocks: the blocks of the range
    /// this copy owns need no snapshot (title memory is not compared with
    /// them; [`crate::owned`] keeps their bytes if the title touches one
    /// while the submission runs). The others are kept block by block.
    fn writing_owned(&self, address: usize, len: usize) {
        let range = (address, address + len);
        let again = {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            let again = pending.contains(&range);
            pending.push(range);
            again
        };
        let (from, to) = self.inside(address, len);
        if again || from >= to {
            return;
        }
        let start = self.start as usize;
        let owned = crate::owned::begin_writes(self.id(), start + from, start + to);
        let mut pieces = self.pieces.lock().unwrap_or_else(|e| e.into_inner());
        let mut pool = PIECES.lock().unwrap_or_else(|e| e.into_inner());
        for block in (start + from) / PAGE..(start + to).div_ceil(PAGE) {
            if owned.binary_search(&block).is_ok() || pieces.contains_key(&block) {
                continue;
            }
            let (a, e) = self.part(block);
            let mut piece = pool.pop().unwrap_or_default();
            piece.clear();
            // SAFETY: inside the mapping.
            piece.extend_from_slice(unsafe { std::slice::from_raw_parts(self.data.add(a), e - a) });
            pieces.insert(block, piece);
        }
    }

    /// [`Shadow::copy_back`] with GPU-owned blocks. Per block of the range:
    /// - still owned: nothing to do, the copy is the block;
    /// - given back to the title while the submission ran: what shaders
    ///   wrote since goes into title memory, as below;
    /// - the title's: taken if it can be ([`crate::owned::acquire`]: the
    ///   copy is as new as title memory, the title did not write it
    ///   meanwhile, it did not touch it lately), else compared with its
    ///   snapshot and what shaders wrote copied back, as without ownership.
    fn copy_back_owned(self: &Arc<Self>, address: usize, len: usize, foreign: &[(usize, usize)]) -> usize {
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        let (mut written, mut avoided, mut synced) = (0, 0, Vec::new());
        if from < to {
            let (first, last) = ((start + from) / PAGE, (start + to).div_ceil(PAGE));
            // The block's part of the range, as offsets.
            let ranged = |block: usize| {
                ((block * PAGE).max(start + from) - start, ((block + 1) * PAGE).min(start + to) - start)
            };
            let (kept, saved) = crate::owned::settle(self.id(), first, last);
            let mut title = vec![0u8; PAGE];
            for saved in &saved {
                let n = saved.bytes.len();
                if let Some(bytes) = self.merge(saved.at - start, &saved.bytes, &mut title[..n]) {
                    written += bytes;
                    synced.push(saved.at / PAGE);
                }
            }
            let mut pieces = self.pieces.lock().unwrap_or_else(|e| e.into_inner());
            // The title's blocks whose copy is current by every write the
            // watch has reported (as `copy_stale` judges it).
            let wanted: Vec<usize> = if self.watched.load(Ordering::Relaxed) == 1 {
                let journal = crate::watch::last_writes(first, last);
                let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                let base = start / PAGE;
                let stale = |block: usize| {
                    let write = journal.binary_search_by_key(&block, |&(p, _)| p).map_or(0, |i| journal[i].1);
                    copied[block - base] == 0 || write > copied[block - base]
                };
                pieces
                    .range(first..last)
                    .map(|(&b, _)| b)
                    .filter(|&b| !stale(b) && !foreign.iter().any(|&(a, e)| a < (b + 1) * PAGE && e > b * PAGE))
                    .collect()
            } else {
                Vec::new()
            };
            let keep: crate::owned::Keep = self.clone();
            let taken = crate::owned::acquire(self.source(), &keep, &wanted);
            for block in kept.iter().chain(&taken) {
                let (a, e) = ranged(*block);
                avoided += e - a;
            }
            pieces_done(taken.iter().filter_map(|b| pieces.remove(b)));
            let mut done = Vec::new();
            for (&block, piece) in pieces.range(first..last) {
                let ((a, e), (has, until)) = (ranged(block), self.part(block));
                if let Some(bytes) = self.merge(a, &piece[a - has..e - has], &mut title[..e - a]) {
                    written += bytes;
                    synced.push(block);
                }
                // Compared whole: a later range over it has nothing to do.
                if (a, e) == (has, until) {
                    done.push(block);
                }
            }
            pieces_done(done.iter().filter_map(|b| pieces.remove(b)));
            drop(pieces);
            crate::owned::avoided(avoided);
            // As in `copy_back`: pages made equal to title memory are
            // current in the copy.
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last
                && !synced.is_empty()
                && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
            {
                let base = start / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                    copied[page - base] = now;
                }
            }
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (address, address + len));
        if pending.is_empty() {
            let pieces = std::mem::take(&mut *self.pieces.lock().unwrap_or_else(|e| e.into_inner()));
            pieces_done(pieces.into_values());
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let owned = crate::owned::any();
        if owned {
            // What other copies hold of the range goes back to the title
            // first and is copied from there whatever the watch says (it
            // may have answered for this submission already).
            let given = crate::owned::release_others(self.id(), from, to);
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            for block in given {
                copied[block - start / PAGE] = 0;
            }
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if owned {
            // The blocks this copy owns are newer here than in title memory.
            busy.extend(crate::owned::held_runs(self.id(), from, to));
        }
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owned::{S, stat};
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::s;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualQuery,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    const WORDS: usize = PAGE / 4;

    // A shader that reads and writes its buffer: every word grows by `add`,
    // so a second run shows what the first left. Structured, for both
    // compilers (see `crate::indirect`), whose root signature it uses.
    const SOURCE: &str = r#"
cbuffer Run : register(b0) { uint words; uint add; uint unused0; uint unused1; };
RWStructuredBuffer<uint> buffer : register(u0);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x < words) buffer[id.x] = buffer[id.x] + add;
}
"#;

    fn adder(host: &crate::host::Host) -> (ID3D12RootSignature, ID3D12PipelineState) {
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; the blob's
        // bytes are read before it is released.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("adder"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
            .unwrap();
            let code = code.unwrap();
            let root = crate::indirect::root_signature(&host.device).unwrap();
            let pso = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.GetBufferPointer(),
                        BytecodeLength: code.GetBufferSize(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        }
    }

    /// A title buffer of whole blocks whose word `i` holds `i`, and a
    /// submission that adds to its words on the GPU through its copy.
    struct Buffer {
        host: &'static crate::host::Host,
        at: usize,
        blocks: usize,
        /// A view to unmap when dropped, not an allocation to free.
        view: bool,
        shadow: Arc<Shadow>,
        adder: (ID3D12RootSignature, ID3D12PipelineState),
    }

    impl Buffer {
        /// `None` without a host GPU.
        fn new(blocks: usize) -> Option<Buffer> {
            // SAFETY: a fresh write-watched allocation, freed when dropped.
            let at = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    blocks * PAGE,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            } as usize;
            assert_ne!(at, 0);
            Buffer::over(at, blocks, blocks, false)
        }

        /// Over memory at `at` of which the first `committed` blocks can
        /// be written.
        fn over(at: usize, blocks: usize, committed: usize, view: bool) -> Option<Buffer> {
            let host = host()?;
            for i in 0..committed * WORDS {
                // SAFETY: committed memory of ours.
                unsafe { (at as *mut u32).add(i).write(i as u32) };
            }
            register(at, at as u64, (blocks * PAGE) as u64);
            let shadow = find(at as u64, (blocks * PAGE) as u64).unwrap();
            Some(Buffer { host, at, blocks, view, shadow, adder: adder(host) })
        }

        /// Runs the shader over words `[from, to)` in one title submission.
        fn add(&self, add: u32, from: usize, to: usize) {
            self.add_reading(add, from, to, None);
        }

        /// The same; with `word`, the title reads that word while the
        /// submission is under way (its shaders not yet done with the
        /// range). Returns what it read.
        fn add_reading(&self, add: u32, from: usize, to: usize, word: Option<usize>) -> u32 {
            let list = self.host.new_list().unwrap();
            let (address, len) = ((self.at + from * 4) as u64, ((to - from) * 4) as u64);
            // As a draw's use of an unordered-access view.
            self.shadow.refresh(&list, address, len);
            self.shadow.gpu_writes(&list, address, len);
            let read = Arc::new(std::sync::atomic::AtomicU32::new(0));
            if let Some(word) = word {
                let (at, read) = (self.at + word * 4, read.clone());
                // SAFETY: committed memory of ours (see `word`).
                list.defer_host(move || read.store(unsafe { (at as *const u32).read_volatile() }, Ordering::Relaxed));
            }
            let constants = [(to - from) as u32, add, 0, 0];
            let gpu = self.shadow.gpu(address);
            // SAFETY: an open list, our root signature and pipeline, and
            // the copy's buffer, which the list keeps.
            unsafe {
                let l = &list.list;
                l.SetComputeRootSignature(&self.adder.0);
                l.SetPipelineState(&self.adder.1);
                l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                l.SetComputeRootUnorderedAccessView(1, gpu);
                l.SetComputeRootUnorderedAccessView(2, gpu);
                l.Dispatch(((to - from) as u32).div_ceil(64), 1, 1);
            }
            list.close();
            self.host.execute_title(&[&list]);
            read.load(Ordering::Relaxed)
        }

        fn add_all(&self, add: u32) {
            self.add(add, 0, self.blocks * WORDS);
        }

        /// Word `i` as the title's code reads it.
        fn word(&self, i: usize) -> u32 {
            // SAFETY: committed memory of ours; a GPU-owned block faults
            // into the handler, which gives it back.
            unsafe { (self.at as *const u32).add(i).read_volatile() }
        }

        fn set_word(&self, i: usize, value: u32) {
            // SAFETY: as above.
            unsafe { (self.at as *mut u32).add(i).write_volatile(value) };
        }

        /// Word `i` as the GPU has it.
        fn gpu_word(&self, i: usize) -> u32 {
            // SAFETY: inside the mapping; no submission runs.
            unsafe { self.shadow.data.cast::<u32>().add(i).read_volatile() }
        }

        fn held(&self) -> Vec<usize> {
            let first = self.at / PAGE;
            crate::owned::held(first, first + self.blocks).iter().map(|b| b - first).collect()
        }

        fn protection(&self, block: usize) -> u32 {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery((self.at + block * PAGE) as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        }
    }

    impl Drop for Buffer {
        fn drop(&mut self) {
            unregister(self.at, self.at as u64);
            crate::title_memory_remapped(self.at, self.blocks * PAGE);
            if !self.view {
                // SAFETY: our allocation.
                unsafe { VirtualFree(self.at as *mut _, 0, MEM_RELEASE) };
            }
        }
    }

    /// The whole of it: what shaders wrote is not copied back; the GPU
    /// reads it again from its copy; the title's read faults once and sees
    /// it; a title write is what the GPU reads next.
    #[test]
    fn gpu_written_blocks_stay_with_the_gpu() {
        let _on = crate::owned::force(true);
        let Some(b) = Buffer::new(4) else {
            return; // no host GPU here
        };
        let (back, avoided, touches) = (crate::host::copied_back(), stat(S::Avoided), stat(S::Touches));
        b.add_all(1000);
        assert_eq!(b.held(), [0, 1, 2, 3]);
        assert_eq!(b.protection(0), PAGE_NOACCESS);
        assert_eq!(crate::host::copied_back() - back, 0, "nothing was copied back");
        assert_eq!(stat(S::Avoided) - avoided, (4 * PAGE) as u64);
        assert_eq!((b.gpu_word(5), b.gpu_word(4 * WORDS - 1)), (1005, 999 + 4 * WORDS as u32));
        // Again: the GPU reads its own bytes (title memory still has 5),
        // and nothing is copied either way.
        let copied = crate::host::PREPARING[1].load(Ordering::Relaxed);
        b.add_all(1000);
        assert_eq!(b.gpu_word(5), 2005);
        assert_eq!(crate::host::PREPARING[1].load(Ordering::Relaxed), copied, "no refresh copied");
        assert_eq!(crate::host::copied_back() - back, 0);
        assert_eq!(stat(S::Avoided) - avoided, (8 * PAGE) as u64);
        assert_eq!(stat(S::Touches), touches);
        // The title reads: one fault, the GPU's data, that block its own.
        assert_eq!(b.word(5), 2005);
        assert_eq!(stat(S::Touches) - touches, 1);
        assert_eq!(b.held(), [1, 2, 3]);
        assert_eq!(b.word(WORDS - 1), 1999 + WORDS as u32);
        assert_eq!(stat(S::Touches) - touches, 1);
        // The title writes there; the GPU's next run starts from it, and
        // the block, which rests, gets the result back at once.
        b.set_word(6, 7);
        b.add_all(1000);
        assert_eq!((b.gpu_word(5), b.gpu_word(6)), (3005, 1007));
        assert_eq!(b.held(), [1, 2, 3]);
        assert_eq!(b.protection(0), PAGE_READWRITE);
        assert_eq!((b.word(5), b.word(6)), (3005, 1007));
        assert_eq!(stat(S::Touches) - touches, 1, "copied back, not faulted for");
        // A write into an owned block: the fault, then the write on top of
        // the GPU's data.
        b.set_word(WORDS + 1, 9);
        assert_eq!(stat(S::Touches) - touches, 2);
        assert_eq!(b.word(WORDS + 2), 3002 + WORDS as u32);
        b.add_all(1);
        assert_eq!((b.word(WORDS + 1), b.word(WORDS + 2)), (10, 3003 + WORDS as u32));
        assert_eq!(b.held(), [2, 3]);
        // The last blocks never left the GPU in five runs.
        assert_eq!(b.gpu_word(3 * WORDS), 3001 + 3 * WORDS as u32);
        assert_eq!(b.word(3 * WORDS), 3001 + 3 * WORDS as u32);
    }

    /// With the switch off nothing changes: everything is copied back
    /// after the submission and no block is closed.
    #[test]
    fn without_the_switch_everything_is_copied_back() {
        let _off = crate::owned::force(false);
        let Some(b) = Buffer::new(2) else {
            return; // no host GPU here
        };
        let (back, bytes, avoided, touches) =
            (crate::host::copied_back(), stat(S::Bytes), stat(S::Avoided), stat(S::Touches));
        b.add_all(1000);
        assert_eq!(b.held(), []);
        assert_eq!(b.protection(0), PAGE_READWRITE);
        assert_eq!(crate::host::copied_back() - back, (2 * PAGE) as u64);
        assert_eq!((b.word(5), b.word(2 * WORDS - 1)), (1005, 999 + 2 * WORDS as u32));
        b.set_word(5, 1);
        b.add_all(1000);
        assert_eq!((b.word(5), b.word(6)), (1001, 2006));
        assert_eq!((stat(S::Bytes), stat(S::Avoided), stat(S::Touches)), (bytes, avoided, touches));
    }

    /// A range shaders write in part. A block the copy has only part of
    /// up to date is not taken: its half of the range is compared and
    /// copied back. Once the copy has all of the block, it is taken, and
    /// a later range over part of it changes nothing for the title.
    #[test]
    fn partly_written_blocks() {
        let _on = crate::owned::force(true);
        let Some(b) = Buffer::new(3) else {
            return; // no host GPU here
        };
        let (back, touches) = (crate::host::copied_back(), stat(S::Touches));
        // From the middle of block 0 to the middle of block 2.
        let (from, to) = (WORDS / 2, 2 * WORDS + WORDS / 2);
        b.add(1000, from, to);
        assert_eq!(b.held(), [1]);
        assert_eq!(crate::host::copied_back() - back, PAGE as u64, "the two half blocks");
        assert_eq!((b.word(1), b.word(from)), (1, 1000 + from as u32));
        assert_eq!((b.word(to - 1), b.word(to)), (999 + to as u32, to as u32));
        // The title writes beside the range; the range again.
        b.set_word(2, 77);
        b.add(1000, from, to);
        assert_eq!(b.held(), [1]);
        assert_eq!(crate::host::copied_back() - back, 2 * PAGE as u64);
        assert_eq!((b.word(2), b.word(from)), (77, 2000 + from as u32));
        assert_eq!(stat(S::Touches), touches);
        // All of it: the copy now has blocks 0 and 2 whole.
        b.add_all(1);
        assert_eq!(b.held(), [0, 1, 2]);
        b.add(1000, from, to);
        assert_eq!(b.held(), [0, 1, 2]);
        assert_eq!(crate::host::copied_back() - back, 2 * PAGE as u64);
        assert_eq!((b.word(2), b.word(from), b.word(to)), (78, 3001 + from as u32, 1 + to as u32));
        assert_eq!(b.word(WORDS + 3), 3004 + WORDS as u32);
        assert_eq!(stat(S::Touches) - touches, 3);
    }

    /// The title touches an owned block while the submission that writes
    /// it is under way: it reads what the copy held then, and what shaders
    /// wrote after is in title memory once the submission is done.
    #[test]
    fn a_touch_while_the_submission_runs() {
        let _on = crate::owned::force(true);
        let Some(b) = Buffer::new(2) else {
            return; // no host GPU here
        };
        b.add_all(1000);
        assert_eq!(b.held(), [0, 1]);
        let (back, touches) = (crate::host::copied_back(), stat(S::Touches));
        assert_eq!(b.add_reading(1000, 0, 2 * WORDS, Some(WORDS + 5)), 1005 + WORDS as u32);
        assert_eq!(b.held(), [0]);
        assert_eq!(crate::host::copied_back() - back, PAGE as u64);
        assert_eq!(b.word(WORDS + 5), 2005 + WORDS as u32);
        assert_eq!(stat(S::Touches) - touches, 1);
        assert_eq!(b.word(5), 2005);
    }

    /// Remapped title memory is no longer the GPU's, and gets nothing back.
    #[test]
    fn remapping_drops_ownership() {
        let _on = crate::owned::force(true);
        let Some(b) = Buffer::new(3) else {
            return; // no host GPU here
        };
        b.add_all(1000);
        assert_eq!(b.held(), [0, 1, 2]);
        let (dropped, touches) = (stat(S::Dropped), stat(S::Touches));
        crate::title_memory_remapped(b.at + PAGE, PAGE);
        assert_eq!(b.held(), [0, 2]);
        assert_eq!(stat(S::Dropped) - dropped, 1);
        assert_eq!(b.protection(1), PAGE_READWRITE);
        assert_eq!(b.word(WORDS + 4), WORDS as u32 + 4, "what the title had there");
        assert_eq!(stat(S::Touches), touches);
        assert_eq!(b.word(4), 1004);
    }

    /// Our own reads of title memory see what the GPU wrote: a copy at
    /// submission (a title buffer copy, an upload) without taking the
    /// blocks from the GPU, and another host copy of the same memory, whose
    /// refresh takes them back for the title first.
    #[test]
    fn our_own_reads_see_gpu_data() {
        let _on = crate::owned::force(true);
        let Some(b) = Buffer::new(3) else {
            return; // no host GPU here
        };
        b.add_all(1000);
        assert_eq!(b.held(), [0, 1, 2]);
        let mut words = vec![0u32; 2 * WORDS];
        crate::host::copy_mapped(words.as_mut_ptr().cast(), (b.at + PAGE / 2) as *const u8, 2 * PAGE, false);
        assert!(words.iter().enumerate().all(|(i, &w)| w == (1000 + WORDS / 2 + i) as u32));
        assert_eq!(b.held(), [0, 1, 2]);
        // A second title buffer inside the first: its own copy.
        let (start, len) = ((b.at + PAGE) as u64, PAGE as u64);
        register(b.at + 1, start, len);
        let inner = find(start, len).unwrap();
        assert!(!Arc::ptr_eq(&inner, &b.shadow));
        let (released, touches) = (stat(S::Released), stat(S::Touches));
        crate::watch::new_submission();
        inner.update(start as usize, len as usize);
        // SAFETY: inside the inner copy's mapping.
        assert_eq!(unsafe { inner.data.cast::<u32>().add(3).read() }, 1003 + WORDS as u32);
        assert_eq!(b.held(), [0, 2]);
        assert_eq!((stat(S::Released) - released, stat(S::Touches) - touches), (1, 0));
        unregister(b.at + 1, start);
    }

    /// The alias policy: a buffer in a view of a section (as xmem's
    /// physical pages, the memory that can be mapped twice) is never taken:
    /// what shaders wrote is in title memory after the submission, through
    /// every view of it.
    #[test]
    fn views_keep_the_copy_back() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, UnmapViewOfFile,
        };
        let _on = crate::owned::force(true);
        if host().is_none() {
            return; // no host GPU here
        }
        // SAFETY: a pagefile section of two blocks, mapped twice.
        let (view, alias) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * PAGE) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let map = || MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * PAGE).Value as usize;
            (map(), map())
        };
        assert!(view != 0 && alias != 0 && view != alias);
        let b = Buffer::over(view, 2, 2, true).unwrap();
        let (back, unwatched, touches) = (crate::host::copied_back(), stat(S::Unwatched), stat(S::Touches));
        b.add_all(1000);
        assert_eq!(b.held(), []);
        assert_eq!(crate::host::copied_back() - back, (2 * PAGE) as u64);
        assert_eq!(stat(S::Unwatched) - unwatched, (2 * PAGE) as u64);
        // SAFETY: the other view of the same pages.
        assert_eq!(unsafe { (alias as *const u32).add(5).read() }, 1005);
        assert_eq!(b.word(WORDS + 5), 1005 + WORDS as u32);
        b.add_all(1000);
        assert_eq!(b.word(5), 2005);
        assert_eq!(stat(S::Touches), touches);
        drop(b);
        // SAFETY: our views.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: alias as *mut _ });
        }
    }

    /// A partly committed title buffer: its unmapped block is never taken,
    /// and the rest works around it.
    #[test]
    fn uncommitted_blocks_are_left_alone() {
        let _on = crate::owned::force(true);
        if host().is_none() {
            return; // no host GPU here
        }
        // SAFETY: a fresh write-watched reservation, freed when dropped.
        let at = unsafe {
            let at = VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE);
            VirtualAlloc(at, 2 * PAGE, MEM_COMMIT, PAGE_READWRITE);
            at as usize
        };
        // The buffer as the title declares it runs into the hole.
        let b = Buffer::over(at, 3, 2, false).unwrap();
        b.add(1000, 0, 3 * WORDS);
        assert_eq!(b.held(), [0, 1]);
        assert_eq!(b.word(WORDS + 5), 1005 + WORDS as u32);
        b.add(1000, 0, 3 * WORDS);
        assert_eq!(b.held(), [0]);
        assert_eq!((b.word(5), b.word(WORDS + 5)), (2005, 2005 + WORDS as u32));
    }
}
