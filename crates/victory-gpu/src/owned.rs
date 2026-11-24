//! Title memory the GPU owns (`VICTORY_LAZY_COPY_BACK=1`, off by default).
//!
//! The console's memory is one pool; here shaders write host copies of
//! title buffers ([`crate::shadow`]), and what they wrote went back into
//! title memory after every submission, read or not. With the switch on, a
//! 64 KiB block of title memory a submission's shaders were given to write
//! is not copied back: the host copy becomes the block's only current
//! holder, the title's pages are made inaccessible, and refreshes of the
//! block copy nothing (the copy is newer than title memory). The first
//! access of the title's CPU to the block faults into [`on_access`], which
//! gives access back and writes the copy's bytes into it: from then on the
//! block is the title's again and is tracked as before.
//!
//! Which memory: write-watched private memory only (xmem's reservations).
//! Physical-page views cannot be watched, and they are the memory that has
//! aliases (xmem's `map_alias`, `CopyPageMappingsBatchX`): protecting one
//! view would leave the others open. They keep the copy-back.
//!
//! Which blocks: the ranges shaders are given are far larger than what
//! they write (the world hands out views of its whole 174 MB heap, which
//! also holds the command memory its threads write meanwhile). So a block
//! is taken only if the title did not write it since its last copy (no
//! write-watch bit before and after its pages are closed), and a block the
//! title touches is left alone for a number of submissions that doubles
//! with every touch: memory the CPU works in costs a fault now and then,
//! not one per frame.
//!
//! Locks: one, [`STATE`]. Under it nothing else is locked and title memory
//! is only written through the kernel, which fails where we would fault;
//! the handler takes only this lock. A thread that faults holds any locks
//! of the title's or ours, so nothing here waits for another thread.
//!
//! shortcut: giving a block back is not atomic for the title's other
//! threads: between the protection change and the write of the copy's
//! bytes (one 64 KiB kernel write) another thread reads the older bytes,
//! and a write of its own there is lost. Private memory has no second view
//! to fill it through; section-backed title memory would.
//! shortcut: the title's own `VirtualProtect` state is taken to be
//! read-write (checked on the first page of each run only).

use std::any::Any;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, MutexGuard};

/// The unit owned: the unit copies are tracked in.
pub(crate) const BLOCK: usize = crate::watch::PAGE;
const KERNEL_PAGE: usize = 4096;

/// Whether GPU-written blocks stay with the GPU (`VICTORY_LAZY_COPY_BACK=1`).
pub(crate) fn enabled() -> bool {
    match ON.load(Relaxed) {
        0 => {
            let on = std::env::var("VICTORY_LAZY_COPY_BACK").as_deref() == Ok("1");
            ON.store(1 + u8::from(on), Relaxed);
            if on {
                tracing::info!("GPU-written title memory is copied back when the title touches it");
            }
            on
        }
        on => on == 2,
    }
}

/// Unknown, off, on.
static ON: AtomicU8 = AtomicU8::new(0);

/// A host copy of title memory `[start, start + len)`, at `data`; `id`
/// tells copies apart.
#[derive(Clone, Copy)]
pub(crate) struct Source {
    pub(crate) id: usize,
    pub(crate) start: usize,
    pub(crate) len: usize,
    pub(crate) data: usize,
}

impl Source {
    /// The part of `block` the copy holds, as title addresses.
    fn part(&self, block: usize) -> (usize, usize) {
        let from = (block * BLOCK).max(self.start);
        (from, ((block + 1) * BLOCK).min(self.start + self.len).max(from))
    }

    /// Where the copy holds title address `at`.
    fn host(&self, at: usize) -> *const u8 {
        (self.data + (at - self.start)) as *const u8
    }
}

/// What keeps a copy's memory alive while it owns blocks.
pub(crate) type Keep = Arc<dyn Any + Send + Sync>;

struct Held {
    src: Source,
    keep: Keep,
    /// Shaders of the running submission may write the block's copy.
    pending: bool,
}

/// What a block's history says about taking it.
#[derive(Default)]
struct Heat {
    /// Times the title touched or wrote it when we wanted it.
    strikes: u8,
    /// The submission from which it may be taken again.
    until: u64,
    /// Not write-watched or not read-write: never (until remapped).
    never: bool,
    /// It was GPU-owned at some time, and the title's CPU touched it while
    /// it was (the census).
    owned: bool,
    touched: bool,
}

/// The bytes title memory got when a block was given back while shaders
/// could still write its copy: what they wrote after differs from these.
pub(crate) struct Saved {
    id: usize,
    /// The title address of the first byte.
    pub(crate) at: usize,
    pub(crate) bytes: Vec<u8>,
}

struct State {
    blocks: BTreeMap<usize, Held>,
    saved: BTreeMap<usize, Saved>,
    heat: BTreeMap<usize, Heat>,
    /// Copies no longer owning the block they were kept for: dropped by
    /// the next submission, not in the handler.
    retired: Vec<Keep>,
    submission: u64,
    /// Room for the kernel's write-watch answer.
    found: Vec<usize>,
}

static STATE: Mutex<State> = Mutex::new(State {
    blocks: BTreeMap::new(),
    saved: BTreeMap::new(),
    heat: BTreeMap::new(),
    retired: Vec::new(),
    submission: 0,
    found: Vec::new(),
});
/// Blocks owned now, and blocks with a history: zero of both means no lock.
static HELD: AtomicUsize = AtomicUsize::new(0);
static KNOWN: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// This thread holds [`STATE`]: a fault now is not one to take.
    static INSIDE: Cell<bool> = const { Cell::new(false) };
}

struct Locked(MutexGuard<'static, State>);

impl Drop for Locked {
    fn drop(&mut self) {
        KNOWN.store(self.0.heat.len(), Relaxed);
        INSIDE.with(|i| i.set(false));
    }
}

impl std::ops::Deref for Locked {
    type Target = State;
    fn deref(&self) -> &State {
        &self.0
    }
}

impl std::ops::DerefMut for Locked {
    fn deref_mut(&mut self) -> &mut State {
        &mut self.0
    }
}

fn lock() -> Locked {
    let state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    INSIDE.with(|i| i.set(true));
    Locked(state)
}

/// Counters, since the last diagnostics line or for good.
#[derive(Clone, Copy)]
pub(crate) enum S {
    /// Runs of blocks and bytes made GPU-owned.
    Runs,
    Bytes,
    /// Bytes of ranges shaders could write that were not compared or
    /// copied back.
    Avoided,
    /// Blocks the title's CPU touched (faults), their bytes copied back,
    /// and the time in the handler.
    Touches,
    TouchBytes,
    FaultUs,
    /// Blocks given back for our own use of title memory, and their bytes.
    Released,
    ReleasedBytes,
    /// Blocks whose ownership ended with their memory (remapped, freed).
    Dropped,
    /// Blocks found gone when they were to be given back.
    Lost,
    /// Bytes read from the owning copy in place of title memory.
    Served,
    /// Bytes not taken: not watched (views), resting after a touch, and
    /// written by the title during the submission.
    Unwatched,
    Resting,
    Written,
    /// The time taking blocks.
    TakeUs,
    /// For good: bytes of blocks ever owned, and of those ever touched.
    EverOwned,
    EverTouched,
    Count,
}
static STATS: [AtomicU64; S::Count as usize] = [const { AtomicU64::new(0) }; S::Count as usize];

fn add(s: S, n: u64) {
    STATS[s as usize].fetch_add(n, Relaxed);
}

#[cfg(test)]
pub(crate) fn stat(s: S) -> u64 {
    STATS[s as usize].load(Relaxed)
}

/// Bytes of a range shaders could write that stay with the GPU.
pub(crate) fn avoided(bytes: usize) {
    add(S::Avoided, bytes as u64);
}

/// Whether any block is owned.
pub(crate) fn any() -> bool {
    HELD.load(Relaxed) != 0
}

/// A submission begins: blocks rest by their number.
pub(crate) fn new_submission() {
    if !enabled() && !any() {
        return;
    }
    let retired = {
        let mut state = lock();
        state.submission += 1;
        std::mem::take(&mut state.retired)
    };
    // Outside the lock: the last reference frees a host buffer.
    drop(retired);
}

/// The blocks `[first, last)` as runs of neighbours.
fn runs(blocks: &[usize]) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &block in blocks {
        match runs.last_mut() {
            Some((_, end)) if *end == block => *end += 1,
            _ => runs.push((block, block + 1)),
        }
    }
    runs
}

/// The parts of `[from, to)` in blocks the copy `id` owns: its refresh
/// leaves them alone.
pub(crate) fn held_runs(id: usize, from: usize, to: usize) -> Vec<(usize, usize)> {
    if !any() || from >= to {
        return Vec::new();
    }
    let state = lock();
    let blocks: Vec<usize> = state
        .0
        .blocks
        .range(from / BLOCK..to.div_ceil(BLOCK))
        .filter(|(_, h)| h.src.id == id)
        .map(|(&b, _)| b)
        .collect();
    drop(state);
    runs(&blocks).into_iter().map(|(a, e)| ((a * BLOCK).max(from), (e * BLOCK).min(to))).collect()
}

/// Shaders of the submission being prepared may write `[from, to)` of the
/// copy `id`. Returns the blocks of the range it owns, in order: giving one
/// back before the submission is settled keeps what title memory got
/// ([`Saved`]).
pub(crate) fn begin_writes(id: usize, from: usize, to: usize) -> Vec<usize> {
    if !any() || from >= to {
        return Vec::new();
    }
    let mut state = lock();
    state
        .0
        .blocks
        .range_mut(from / BLOCK..to.div_ceil(BLOCK))
        .filter(|(_, h)| h.src.id == id)
        .map(|(&b, h)| {
            h.pending = true;
            b
        })
        .collect()
}

/// The submission ran: of blocks `[first, last)`, those the copy `id`
/// still owns (in order), and what was kept of those given back meanwhile.
pub(crate) fn settle(id: usize, first: usize, last: usize) -> (Vec<usize>, Vec<Saved>) {
    if !any() && KNOWN.load(Relaxed) == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut state = lock();
    let state = &mut *state;
    let kept = state
        .blocks
        .range_mut(first..last)
        .filter(|(_, h)| h.src.id == id)
        .map(|(&b, h)| {
            h.pending = false;
            b
        })
        .collect();
    let given: Vec<usize> =
        state.saved.range(first..last).filter(|(_, s)| s.id == id).map(|(&b, _)| b).collect();
    let saved = given.iter().filter_map(|b| state.saved.remove(b)).collect();
    (kept, saved)
}

/// Whether the write-watched pages of blocks `[first, last)` were written
/// since the watch was last reset, per block; `None` if the memory is not
/// one write-watched range. Nothing is reset: the next refresh still sees
/// the writes.
fn written(found: &mut Vec<usize>, first: usize, last: usize) -> Option<Vec<bool>> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    let mut count = (last - first) * BLOCK / KERNEL_PAGE;
    if found.len() < count {
        found.resize(count, 0);
    }
    let mut granularity = 0;
    // SAFETY: the answer buffer holds `count` entries.
    let ok = unsafe {
        GetWriteWatch(
            0,
            (first * BLOCK) as *const _,
            (last - first) * BLOCK,
            found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } == 0;
    if !ok {
        return None;
    }
    let mut blocks = vec![false; last - first];
    for &at in &found[..count] {
        blocks[at / BLOCK - first] = true;
    }
    Some(blocks)
}

/// Changes the protection of blocks `[first, last)`; the protection the
/// first page had, or `None` if it failed (something is not committed).
fn protect(first: usize, last: usize, protection: u32) -> Option<u32> {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    let mut old = 0;
    // SAFETY: title memory; only its protection changes.
    let ok = unsafe {
        VirtualProtect((first * BLOCK) as *const _, (last - first) * BLOCK, protection, &mut old)
    } != 0;
    ok.then_some(old)
}

/// The submissions a block rests after its `strikes`th touch: 8 to 4096.
fn rest(strikes: u8) -> u64 {
    4 << strikes.min(10)
}

/// The title used the block (or wrote it when we wanted it).
fn strike(state: &mut State, block: usize) -> &mut Heat {
    let now = state.submission;
    let heat = state.heat.entry(block).or_default();
    heat.strikes = heat.strikes.saturating_add(1);
    heat.until = now + rest(heat.strikes);
    heat
}

/// After the submission that wrote them ran: makes those of `blocks` (in
/// order) GPU-owned that can be, with `src` the copy holding them. The
/// caller vouches that the copy is as new as title memory by every write
/// the watch has reported; here the ones it has not reported yet are asked
/// for, before the pages are closed and again after. Returns the blocks
/// taken.
pub(crate) fn acquire(src: Source, keep: &Keep, blocks: &[usize]) -> Vec<usize> {
    use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READWRITE};
    if blocks.is_empty() {
        return Vec::new();
    }
    install();
    let started = std::time::Instant::now();
    let mut state = lock();
    let state = &mut *state;
    let now = state.submission;
    let mut wanted = Vec::new();
    for &block in blocks {
        let heat = state.heat.entry(block).or_default();
        if heat.never {
            add(S::Unwatched, BLOCK as u64);
        } else if heat.until > now {
            add(S::Resting, BLOCK as u64);
        } else if !state.blocks.contains_key(&block) {
            wanted.push(block);
        }
    }
    // Not written since the watch was reset (at or before their refresh).
    let mut found = std::mem::take(&mut state.found);
    let mut clean = Vec::new();
    for (first, last) in runs(&wanted) {
        // A run across allocations is asked about block by block.
        let pieces: Vec<(usize, usize)> = match written(&mut found, first, last) {
            Some(w) => {
                for (i, w) in w.into_iter().enumerate() {
                    if w {
                        strike(state, first + i);
                        add(S::Written, BLOCK as u64);
                    } else {
                        clean.push(first + i);
                    }
                }
                continue;
            }
            None => (first..last).map(|b| (b, b + 1)).collect(),
        };
        for (first, last) in pieces {
            match written(&mut found, first, last) {
                Some(w) if w[0] => {
                    strike(state, first);
                    add(S::Written, BLOCK as u64);
                }
                Some(_) => clean.push(first),
                None => {
                    state.heat.entry(first).or_default().never = true;
                    add(S::Unwatched, BLOCK as u64);
                }
            }
        }
    }
    // Closed, then asked again: a write that landed in between shows, and
    // from here on a write faults.
    let mut closed = Vec::new();
    for (first, last) in runs(&clean) {
        let pieces = match protect(first, last, PAGE_NOACCESS) {
            Some(old) => vec![(first, last, old)],
            // Something in the run is not committed: block by block, and
            // those that fail rest (a run of them is asked every time).
            None => (first..last)
                .filter_map(|b| {
                    let old = protect(b, b + 1, PAGE_NOACCESS);
                    if old.is_none() {
                        strike(state, b);
                    }
                    old.map(|old| (b, b + 1, old))
                })
                .collect(),
        };
        for (first, last, old) in pieces {
            if old == PAGE_READWRITE {
                closed.push((first, last));
                continue;
            }
            // Not ours to open later as read-write.
            protect(first, last, old);
            for block in first..last {
                state.heat.entry(block).or_default().never = true;
            }
        }
    }
    let mut taken = Vec::new();
    for (first, last) in closed {
        let written = written(&mut found, first, last).unwrap_or_else(|| vec![true; last - first]);
        for (i, written) in written.into_iter().enumerate() {
            let block = first + i;
            if written {
                protect(block, block + 1, PAGE_READWRITE);
                strike(state, block);
                add(S::Written, BLOCK as u64);
                continue;
            }
            state.blocks.insert(block, Held { src, keep: keep.clone(), pending: false });
            taken.push(block);
        }
    }
    state.found = found;
    HELD.store(state.blocks.len(), Relaxed);
    for &block in &taken {
        let heat = state.heat.entry(block).or_default();
        if !heat.owned {
            heat.owned = true;
            add(S::EverOwned, BLOCK as u64);
        }
    }
    add(S::Runs, runs(&taken).len() as u64);
    add(S::Bytes, (taken.len() * BLOCK) as u64);
    add(S::TakeUs, started.elapsed().as_micros() as u64);
    taken
}

/// Why a block goes back to the title.
#[derive(PartialEq)]
enum Why {
    /// The title's CPU touched it.
    Touch,
    /// We use its title memory ourselves.
    Ours,
}

/// Gives `block` back to the title: access, then the copy's bytes. False if
/// it was not owned, or its memory is gone.
fn give_back(state: &mut State, block: usize, why: Why) -> bool {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    let Some(held) = state.blocks.remove(&block) else {
        return false;
    };
    HELD.store(state.blocks.len(), Relaxed);
    let (from, to) = held.src.part(block);
    let host = held.src.host(from);
    let mut landed = protect(block, block + 1, PAGE_READWRITE).is_some();
    if landed && held.pending {
        // One read of the copy, which shaders may be writing: title memory
        // gets exactly the bytes the later comparison is made with.
        // SAFETY: inside the copy, kept alive by `held.keep`.
        let bytes = unsafe { std::slice::from_raw_parts(host, to - from) }.to_vec();
        landed = write(from, bytes.as_ptr(), bytes.len());
        if landed {
            state.saved.insert(block, Saved { id: held.src.id, at: from, bytes });
        }
    } else if landed {
        landed = write(from, host, to - from);
    }
    state.retired.push(held.keep);
    if !landed {
        // Freed or decommitted without our hearing of it: nothing to give
        // the bytes to, and whatever is mapped there next starts afresh.
        state.heat.remove(&block);
        add(S::Lost, 1);
        return false;
    }
    let heat = strike(state, block);
    if why == Why::Touch {
        if !heat.touched {
            heat.touched = true;
            add(S::EverTouched, BLOCK as u64);
        }
        add(S::Touches, 1);
        add(S::TouchBytes, (to - from) as u64);
    } else {
        add(S::Released, 1);
        add(S::ReleasedBytes, (to - from) as u64);
    }
    true
}

/// Writes `len` bytes at `src` into title memory at `dst` through the
/// kernel, which fails where a plain copy would fault (under our lock).
fn write(dst: usize, src: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the source is mapped; the kernel checks the destination.
    len == 0 || unsafe { WriteProcessMemory(GetCurrentProcess(), dst as _, src.cast(), len, &mut done) != 0 }
}

/// Before we read or write title memory `[at, at + len)` in a way that
/// does not fault into the handler (through the kernel), or that relies on
/// the write watch to see what changed: its blocks go back to the title.
pub(crate) fn release(at: usize, len: usize) {
    if !any() || len == 0 {
        return;
    }
    let mut state = lock();
    let blocks: Vec<usize> = state
        .0
        .blocks
        .range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK))
        .map(|(&b, _)| b)
        .collect();
    for block in blocks {
        give_back(&mut state, block, Why::Ours);
    }
}

/// Before the copy `id` is refreshed from `[from, to)`: blocks other copies
/// own there go back to the title. Returns them: the write watch may not
/// report our write of them in this submission (see
/// [`crate::watch::new_submission`]), so the caller copies them anyway.
pub(crate) fn release_others(id: usize, from: usize, to: usize) -> Vec<usize> {
    if !any() || from >= to {
        return Vec::new();
    }
    let mut state = lock();
    let blocks: Vec<usize> = state
        .0
        .blocks
        .range(from / BLOCK..to.div_ceil(BLOCK))
        .filter(|(_, h)| h.src.id != id)
        .map(|(&b, _)| b)
        .collect();
    blocks.into_iter().filter(|&b| give_back(&mut state, b, Why::Ours)).collect()
}

/// Title memory `[at, at + len)` is unmapped, remapped or freed: blocks
/// owned there are the title's again without their bytes (what it freed
/// needs none), and their history is forgotten.
pub(crate) fn forget(at: usize, len: usize) {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    if !any() && KNOWN.load(Relaxed) == 0 {
        return;
    }
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    let mut state = lock();
    let state = &mut *state;
    let blocks: Vec<usize> = state.blocks.range(range.clone()).map(|(&b, _)| b).collect();
    for block in blocks {
        if let Some(held) = state.blocks.remove(&block) {
            // Fails if the memory is gone already.
            protect(block, block + 1, PAGE_READWRITE);
            state.retired.push(held.keep);
            add(S::Dropped, 1);
        }
    }
    HELD.store(state.blocks.len(), Relaxed);
    state.saved.retain(|b, _| !range.contains(b));
    let known: Vec<usize> = state.heat.range(range).map(|(&b, _)| b).collect();
    for block in known {
        state.heat.remove(&block);
    }
}

/// The allocation at `base` is released whole (`VirtualFree` with
/// `MEM_RELEASE`): as [`forget`], for the blocks owned in it.
pub(crate) fn forget_allocation(base: usize) {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    if !any() && KNOWN.load(Relaxed) == 0 {
        return;
    }
    // Its extent: region by region up to the last block we know in it.
    let last = {
        let state = lock();
        let known = |m: Option<usize>| m.map_or(0, |b| (b + 1) * BLOCK);
        known(state.blocks.keys().next_back().copied()).max(known(state.heat.keys().next_back().copied()))
    };
    let mut end = base;
    while end < last {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(end as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.AllocationBase as usize != base {
            break;
        }
        end = info.BaseAddress as usize + info.RegionSize;
    }
    if end > base {
        forget(base, end - base);
    }
}

/// For a copy of title memory `[src, src + len)` to `dst`: the parts in
/// owned blocks are copied from their owners' copies, which are newer, and
/// returned in order for the caller to read around. A block its owner holds
/// only part of goes back to the title, to be read there. `kernel`: `dst`
/// is title memory, written through the kernel.
pub(crate) fn serve(src: usize, len: usize, dst: *mut u8, kernel: bool) -> Vec<(usize, usize)> {
    if !any() || len == 0 {
        return Vec::new();
    }
    let end = src + len;
    let mut state = lock();
    let blocks: Vec<usize> = state.blocks.range(src / BLOCK..end.div_ceil(BLOCK)).map(|(&b, _)| b).collect();
    let mut served: Vec<(usize, usize)> = Vec::new();
    for block in blocks {
        let held = state.blocks[&block].src;
        let (from, to) = ((block * BLOCK).max(src), ((block + 1) * BLOCK).min(end));
        let (has, until) = held.part(block);
        if has > from || until < to {
            give_back(&mut state, block, Why::Ours);
            continue;
        }
        let (host, into) = (held.host(from), dst.wrapping_add(from - src));
        if kernel {
            write(into as usize, host, to - from);
        } else {
            // SAFETY: the copy is kept alive by its block; the caller
            // vouches for the destination, which is not title memory.
            unsafe { std::ptr::copy(host, into, to - from) };
        }
        add(S::Served, (to - from) as u64);
        match served.last_mut() {
            Some((_, e)) if *e == from => *e = to,
            _ => served.push((from, to)),
        }
    }
    served
}

/// Registers the handler that takes the title's accesses to owned blocks.
fn install() {
    use windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler;
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler touches only our state and title protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_access));
    });
}

/// A read or write of an owned block by anything but us: the block goes
/// back to the title and the access is run again. Also the fault of a
/// thread that lost the race for a block another just got back.
unsafe extern "system" fn on_access(
    info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
) -> i32 {
    use windows_sys::Win32::Foundation::{
        EXCEPTION_ACCESS_VIOLATION, GetLastError, NTSTATUS, SetLastError,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
    };
    // SAFETY: the system passes the exception being dispatched.
    let r = unsafe { &*(*info).ExceptionRecord };
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
        || r.NumberParameters < 2
        || (!any() && KNOWN.load(Relaxed) == 0)
        || INSIDE.with(|i| i.get())
    {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let (kind, at) = (r.ExceptionInformation[0], r.ExceptionInformation[1]);
    let block = at / BLOCK;
    // The code that faulted may be between a call and its `GetLastError`.
    // SAFETY: plain thread state.
    let error = unsafe { GetLastError() };
    let started = std::time::Instant::now();
    let mut state = lock();
    let taken = if state.blocks.contains_key(&block) {
        let given = give_back(&mut state, block, Why::Touch);
        add(S::FaultUs, started.elapsed().as_micros() as u64);
        given
    } else {
        // A block we closed at some time, open now: another thread got it
        // back between this one's fault and its turn here.
        state.heat.contains_key(&block) && accessible(at, kind)
    };
    drop(state);
    // SAFETY: as above.
    unsafe { SetLastError(error) };
    if taken { EXCEPTION_CONTINUE_EXECUTION } else { EXCEPTION_CONTINUE_SEARCH }
}

/// Whether an access of `kind` (0 read, 1 write) at `at` would succeed now.
fn accessible(at: usize, kind: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_GUARD,
        PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    let allowed = match kind {
        0 => PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE,
        1 => PAGE_READWRITE | PAGE_EXECUTE_READWRITE,
        _ => 0,
    };
    n != 0 && info.State == MEM_COMMIT && info.Protect & PAGE_GUARD == 0 && info.Protect & allowed != 0
}

/// For tests: the switch as `on` until the guard is dropped, one test at a
/// time (ownership and its counters are the process's).
#[cfg(test)]
pub(crate) fn force(on: bool) -> Forced {
    static ONE: Mutex<()> = Mutex::new(());
    let one = ONE.lock().unwrap_or_else(|e| e.into_inner());
    enabled();
    Forced { was: ON.swap(1 + u8::from(on), Relaxed), _one: one }
}

#[cfg(test)]
pub(crate) struct Forced {
    was: u8,
    _one: MutexGuard<'static, ()>,
}

#[cfg(test)]
impl Drop for Forced {
    fn drop(&mut self) {
        ON.store(self.was, Relaxed);
    }
}

/// For tests: the blocks owned among `[first, last)`.
#[cfg(test)]
pub(crate) fn held(first: usize, last: usize) -> Vec<usize> {
    lock().0.blocks.range(first..last).map(|(&b, _)| b).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS,
        PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualQuery,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// Write-watched title memory of whole blocks holding `fill`, with a
    /// host copy of it holding `gpu` (what shaders made of it).
    struct Title {
        at: usize,
        blocks: usize,
        copy: Arc<Vec<u8>>,
    }

    impl Title {
        fn new(blocks: usize, fill: u8, gpu: u8) -> Title {
            // SAFETY: a fresh write-watched allocation, freed when dropped.
            let at = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    blocks * BLOCK,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            } as usize;
            assert_ne!(at, 0);
            // SAFETY: committed pages of ours.
            unsafe { (at as *mut u8).write_bytes(fill, blocks * BLOCK) };
            let title = Title { at, blocks, copy: Arc::new(vec![gpu; blocks * BLOCK]) };
            title.copied();
            title
        }

        /// As after a refresh: the watch has reported every write so far.
        fn copied(&self) {
            crate::watch::new_submission();
            crate::watch::writes(self.at, self.at + self.blocks * BLOCK, false).expect("watched");
        }

        fn first(&self) -> usize {
            self.at / BLOCK
        }

        fn source(&self) -> Source {
            Source {
                id: Arc::as_ptr(&self.copy) as usize,
                start: self.at,
                len: self.blocks * BLOCK,
                data: self.copy.as_ptr() as usize,
            }
        }

        /// Takes every block it can; returns them as indices.
        fn take(&self) -> Vec<usize> {
            let keep: Keep = self.copy.clone();
            let all: Vec<usize> = (self.first()..self.first() + self.blocks).collect();
            acquire(self.source(), &keep, &all).iter().map(|b| b - self.first()).collect()
        }

        fn held(&self) -> Vec<usize> {
            held(self.first(), self.first() + self.blocks).iter().map(|b| b - self.first()).collect()
        }

        /// The byte at `offset`, read as the title's code would.
        fn read(&self, offset: usize) -> u8 {
            // SAFETY: committed memory of ours; a GPU-owned block faults
            // into the handler, which gives it back.
            unsafe { ((self.at + offset) as *const u8).read_volatile() }
        }

        fn write(&self, offset: usize, byte: u8) {
            // SAFETY: as above.
            unsafe { ((self.at + offset) as *mut u8).write_volatile(byte) };
        }

        fn protection(&self, block: usize) -> u32 {
            protection(self.at + block * BLOCK)
        }

        /// Lets `n` submissions pass.
        fn submissions(&self, n: u64) {
            for _ in 0..n {
                new_submission();
            }
        }
    }

    impl Drop for Title {
        fn drop(&mut self) {
            forget(self.at, self.blocks * BLOCK);
            crate::watch::title_memory_remapped(self.at, self.blocks * BLOCK);
            // SAFETY: our allocation.
            unsafe { VirtualFree(self.at as *mut _, 0, MEM_RELEASE) };
        }
    }

    fn protection(at: usize) -> u32 {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        info.Protect
    }

    /// A block the GPU owns is closed to the title; its first access, read
    /// or write, brings the copy's bytes and costs one fault. A touched
    /// block rests before it is taken again, and a block the title wrote
    /// is not taken.
    #[test]
    fn a_touch_gives_the_block_back() {
        let _on = force(true);
        let title = Title::new(4, 1, 2);
        let (touches, bytes) = (stat(S::Touches), stat(S::TouchBytes));
        assert_eq!(title.take(), [0, 1, 2, 3]);
        assert_eq!(title.protection(1), PAGE_NOACCESS);
        // A read: the GPU's bytes, and only that block is the title's.
        assert_eq!(title.read(BLOCK + 100), 2);
        assert_eq!(title.held(), [0, 2, 3]);
        assert_eq!(title.protection(1), PAGE_READWRITE);
        assert_eq!((stat(S::Touches) - touches, stat(S::TouchBytes) - bytes), (1, BLOCK as u64));
        // A write lands in the GPU's bytes.
        title.write(2 * BLOCK + 7, 9);
        assert_eq!((title.read(2 * BLOCK + 6), title.read(2 * BLOCK + 7), title.read(3 * BLOCK - 1)), (2, 9, 2));
        assert_eq!(title.held(), [0, 3]);
        // Then the blocks are the title's: no more faults.
        assert_eq!(title.read(BLOCK), 2);
        assert_eq!(stat(S::Touches) - touches, 2);
        // The copy took them again (as a refresh does); they rest.
        title.copied();
        assert_eq!(title.take(), [], "touched blocks rest");
        title.submissions(8);
        assert_eq!(title.take(), [1, 2]);
        // A second touch: twice the rest.
        title.read(BLOCK);
        title.copied();
        title.submissions(8);
        assert_eq!(title.take(), []);
        title.submissions(8);
        assert_eq!(title.take(), [1]);
        // A block the title wrote since its last copy is not taken.
        forget(title.at, title.blocks * BLOCK);
        title.copied();
        title.write(3 * BLOCK, 5);
        let written = stat(S::Written);
        assert_eq!(title.take(), [0, 1, 2]);
        assert_eq!(stat(S::Written) - written, BLOCK as u64);
        assert_eq!(title.protection(3), PAGE_READWRITE);
        assert_eq!(title.read(3 * BLOCK), 5);
    }

    /// Many threads touching one block at once: one copy back, and none of
    /// them sees the fault.
    #[test]
    fn threads_touching_one_block_take_it_once() {
        let _on = force(true);
        let title = Title::new(2, 1, 2);
        let touches = stat(S::Touches);
        assert_eq!(title.take(), [0, 1]);
        let go = std::sync::Barrier::new(8);
        std::thread::scope(|s| {
            for i in 0..8 {
                let (title, go) = (&title, &go);
                s.spawn(move || {
                    go.wait();
                    assert_eq!(title.read(BLOCK + i * 4096), 2);
                });
            }
        });
        assert_eq!(stat(S::Touches) - touches, 1);
        assert_eq!(title.held(), [0]);
    }

    /// A block given back while shaders may still write its copy: title
    /// memory gets the copy's bytes of that moment, and they are kept for
    /// the comparison after the submission.
    #[test]
    fn a_touch_while_shaders_write_keeps_the_bytes() {
        let _on = force(true);
        let title = Title::new(3, 1, 2);
        assert_eq!(title.take(), [0, 1, 2]);
        let (id, first) = (title.source().id, title.first());
        assert_eq!(begin_writes(id, title.at + BLOCK, title.at + 3 * BLOCK), [first + 1, first + 2]);
        assert_eq!(title.read(BLOCK + 5), 2);
        assert_eq!(title.read(5), 2, "not being written: nothing kept");
        let (kept, saved) = settle(id, first, first + 3);
        assert_eq!(kept, [first + 2]);
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].at, title.at + BLOCK);
        assert!(saved[0].bytes.len() == BLOCK && saved[0].bytes.iter().all(|&b| b == 2));
        // Settled: a touch now keeps nothing.
        assert_eq!(title.read(2 * BLOCK), 2);
        assert!(settle(id, first, first + 3).1.is_empty());
    }

    /// Our own copies of title memory read owned blocks from their owner,
    /// without a failed read or a hole found; a block its owner holds part
    /// of, and anything the kernel is to write, go back to the title first.
    #[test]
    fn our_own_reads_are_served_and_kernel_writes_release() {
        use crate::host::{FAILED_READS, PROBES, copy_mapped};
        let _on = force(true);
        let title = Title::new(4, 1, 2);
        assert_eq!(title.take(), [0, 1, 2, 3]);
        // Block 1 goes back, so the copy below reads both kinds.
        title.read(BLOCK);
        title.write(BLOCK, 7);
        FAILED_READS.take();
        PROBES.take();
        let served = stat(S::Served);
        let mut copy = vec![0xaa; 3 * BLOCK];
        copy_mapped(copy.as_mut_ptr(), (title.at + 100) as *const u8, 3 * BLOCK, false);
        assert!(copy[..BLOCK - 100].iter().all(|&b| b == 2));
        assert_eq!(copy[BLOCK - 100], 7);
        assert!(copy[BLOCK - 99..].iter().all(|&b| b == 2));
        assert_eq!(stat(S::Served) - served, (2 * BLOCK) as u64);
        assert_eq!((FAILED_READS.take(), PROBES.take()), (0, 0));
        assert!(crate::watch::holes_in(title.first(), title.first() + 4).is_empty());
        assert_eq!(title.held(), [0, 2, 3], "served, not given back");
        // A read through the kernel for our own records.
        let mut bytes = [0u8; 16];
        assert!(crate::host::read_into((title.at + 2 * BLOCK) as *const u8, &mut bytes));
        assert_eq!(bytes, [2; 16]);
        assert_eq!(title.held(), [0, 3]);
        // A kernel write (a file read into title memory) lands.
        crate::kernel_will_write(title.at + 3 * BLOCK + 10, 4);
        assert_eq!(title.held(), [0]);
        assert!(write(title.at + 3 * BLOCK + 10, [8u8; 4].as_ptr(), 4));
        assert_eq!((title.read(3 * BLOCK + 9), title.read(3 * BLOCK + 10)), (2, 8));
        // A copy to title memory: the source is served, the destination
        // given back first.
        title.copied();
        title.submissions(8);
        assert_eq!(title.take(), [1, 2, 3]);
        copy_mapped((title.at + BLOCK) as *mut u8, title.at as *const u8, BLOCK, true);
        assert_eq!(title.held(), [0, 2, 3]);
        assert_eq!(title.read(BLOCK), 2);
        // An owner that holds only part of a block cannot serve all of it.
        forget(title.at, 4 * BLOCK);
        title.copied();
        let part = Source { start: title.at + 100, len: BLOCK, ..title.source() };
        let keep: Keep = title.copy.clone();
        assert_eq!(acquire(part, &keep, &[title.first()]), [title.first()]);
        copy_mapped(copy.as_mut_ptr(), title.at as *const u8, BLOCK, false);
        assert!(copy[..100].iter().all(|&b| b == 1), "the title's own bytes");
        assert!(copy[100..BLOCK].iter().all(|&b| b == 2));
        assert_eq!(title.held(), []);
        assert_eq!(FAILED_READS.take(), 0);
    }

    /// Memory the runtime remaps or the title frees is the title's again
    /// without its bytes; memory freed without a word is found gone.
    #[test]
    fn remapped_and_freed_memory_is_dropped() {
        let _on = force(true);
        let title = Title::new(4, 1, 2);
        assert_eq!(title.take(), [0, 1, 2, 3]);
        let (dropped, lost, touches) = (stat(S::Dropped), stat(S::Lost), stat(S::Touches));
        crate::title_memory_remapped(title.at + BLOCK, BLOCK);
        assert_eq!(title.held(), [0, 2, 3]);
        assert_eq!(title.read(BLOCK), 1, "the title's bytes, not the copy's");
        // The allocation released whole.
        let other = Title::new(2, 1, 2);
        assert_eq!(other.take(), [0, 1]);
        crate::title_memory_released(other.at);
        assert_eq!(other.held(), []);
        assert_eq!(other.read(0), 1);
        assert_eq!(title.held(), [0, 2, 3], "another allocation's blocks stay");
        assert_eq!(stat(S::Dropped) - dropped, 3);
        // Decommitted behind our back, then needed: nothing to give it to.
        // SAFETY: a block of our allocation.
        assert_ne!(unsafe { VirtualFree((title.at + 2 * BLOCK) as *mut _, BLOCK, MEM_DECOMMIT) }, 0);
        release(title.at + 2 * BLOCK, BLOCK);
        assert_eq!(title.held(), [0, 3]);
        assert_eq!(stat(S::Lost) - lost, 1);
        assert_eq!(stat(S::Touches), touches);
        // Committed again, it is ordinary memory.
        // SAFETY: as above.
        unsafe { VirtualAlloc((title.at + 2 * BLOCK) as *const _, BLOCK, MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(title.read(2 * BLOCK), 0);
    }

    /// The alias policy: views of a section (xmem's physical pages, which
    /// are what can be mapped twice) are never taken, so no alias is ever
    /// left open beside a closed one. Nor is memory that is not watched.
    #[test]
    fn views_and_unwatched_memory_are_not_taken() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, UnmapViewOfFile,
        };
        let _on = force(true);
        // SAFETY: a pagefile section of two blocks, mapped twice.
        let (view, alias) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let map = || MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize;
            (map(), map())
        };
        assert!(view != 0 && alias != 0 && view != alias);
        let copy = Arc::new(vec![2u8; 2 * BLOCK]);
        let src = Source { id: 1, start: view, len: 2 * BLOCK, data: copy.as_ptr() as usize };
        let keep: Keep = copy.clone();
        let unwatched = stat(S::Unwatched);
        assert_eq!(acquire(src, &keep, &[view / BLOCK, view / BLOCK + 1]), []);
        assert_eq!(stat(S::Unwatched) - unwatched, (2 * BLOCK) as u64);
        assert_eq!((protection(view), protection(alias)), (PAGE_READWRITE, PAGE_READWRITE));
        // Counted each time it is wanted, asked about once.
        assert_eq!(acquire(src, &keep, &[view / BLOCK]), []);
        assert_eq!(stat(S::Unwatched) - unwatched, (3 * BLOCK) as u64);
        forget(view, 2 * BLOCK);
        // SAFETY: our views.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ });
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: alias as *mut _ });
        }
        // Committed memory without write watching.
        // SAFETY: a fresh allocation, freed below.
        let plain = unsafe { VirtualAlloc(std::ptr::null(), BLOCK, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        let src = Source { start: plain, len: BLOCK, ..src };
        assert_eq!(acquire(src, &keep, &[plain / BLOCK]), []);
        assert_eq!(protection(plain), PAGE_READWRITE);
        forget(plain, BLOCK);
        // SAFETY: our allocation.
        unsafe { VirtualFree(plain as *mut _, 0, MEM_RELEASE) };
    }
}
