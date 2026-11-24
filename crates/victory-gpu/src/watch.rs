//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection.
//!
//! Blocks with nothing mapped (the gaps between views, uncommitted parts of
//! a pool) are remembered as holes ([`holes_in`]): they cannot be guarded,
//! and reading them fails.
//!
//! shortcut: writes into physical pages through another view (xmem's
//! `write_physical`, DirectStorage's page destinations) are not seen at
//! the guarded addresses; they go to the texture pool, whose textures are
//! hashed, not guarded. A system call writing into a guarded block fails
//! (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges the kernel was asked about since the submission began
    /// (see [`new_submission`]); under Wine only.
    asked_now: Vec<(usize, usize)>,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
    asked_now: Vec::new(),
});

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
fn on_wine() -> bool {
    static WINE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// A title allocation as Wine reports it, in [`PAGE`] units.
#[derive(Clone, Copy)]
struct Region {
    /// The allocation base, for dropping stale entries.
    base: usize,
    first: usize,
    last: usize,
    /// A file view (xmem's physical pages): not write-watched.
    mapped: bool,
    /// Small enough (4 GiB at most) to be asked about whole.
    whole: bool,
}

/// The allocation holding `at`, cached by allocation base (walking an
/// allocation's regions under Wine costs one VirtualQuery each). Those
/// above 4 GiB are not asked about whole.
fn allocation(at: usize) -> Option<Region> {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let region = allocation_now(at);
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Allocations, 1);
        add(C::AllocationUs, started.elapsed().as_micros() as u64);
    }
    region
}

fn allocation_now(at: usize) -> Option<Region> {
    use windows_sys::Win32::System::Memory::{MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let page = at / PAGE;
    // Known allocations first: under Wine VirtualQuery takes the same lock
    // the title's write faults do.
    if let Some((_, &r)) = REGIONS.lock().unwrap_or_else(|e| e.into_inner()).range(..=page).next_back()
        && page < r.last
    {
        return Some(r);
    }
    let query = |p: usize| {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        (n != 0).then_some(info)
    };
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::RegionWalks, 1);
    let info = query(at)?;
    let base = info.AllocationBase as usize;
    if base == 0 {
        return None;
    }
    // Walk the allocation's regions (Wine splits it by protection, one
    // region per run of pages); a long walk stops, and the allocation is
    // then only remembered as far as it went, never asked about whole.
    let mut end = base;
    let mut done = false;
    for _ in 0..256 {
        let Some(i) = query(end) else {
            done = true;
            break;
        };
        if i.AllocationBase as usize != base {
            done = true;
            break;
        }
        end = i.BaseAddress as usize + i.RegionSize;
    }
    if !done {
        // At least past `at`, so the next asking here is a cache hit.
        end = end.max((at / PAGE + 1) * PAGE).max(at + (256 << 20));
    }
    if end <= at {
        return None;
    }
    let r = Region {
        base,
        first: base.div_ceil(PAGE),
        last: end / PAGE,
        mapped: info.Type == MEM_MAPPED,
        whole: done && end - base <= 4 << 30,
    };
    if r.first * PAGE > at || r.last <= r.first {
        return None;
    }
    let mut regions = REGIONS.lock().unwrap_or_else(|e| e.into_inner());
    // An allocation that replaced older ones in its range.
    regions.retain(|_, o| o.last <= r.first || o.first >= r.last);
    regions.insert(r.first, r);
    Some(r)
}

/// Allocations seen by [`allocation`], by first page. Remaps (xmem's views)
/// and failed askings drop entries.
static REGIONS: Mutex<BTreeMap<usize, Region>> = Mutex::new(BTreeMap::new());

fn forget_region(base: usize) {
    REGIONS.lock().unwrap_or_else(|e| e.into_inner()).retain(|_, r| r.base != base);
}

/// A submission begins: memory asked about from now on is asked of the
/// kernel again. Within one submission, a range the kernel already
/// answered for is answered from the journal: Wine emulates write watching
/// page by page under one lock (about 1 ms an asking), and the world's
/// copies ask about the same big buffers dozens of times per submission
/// (17 GB checked per frame, 3 s of a frame). A write between two askings
/// in one submission keeps its dirty bit, so the next submission sees it.
pub(crate) fn new_submission() {
    if on_wine() {
        JOURNAL.lock().unwrap_or_else(|e| e.into_inner()).asked_now.clear();
    }
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    if on_wine() && j.asked_now.iter().any(|&(f, l)| f <= first && last <= l) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::JournalHits, 1);
        j.asked += 1;
        return Some((
            j.asked,
            j.written.range(first..last).map(|(&p, &w)| (p, w)).collect(),
        ));
    }
    // Under Wine: view memory cannot be watched (straight to its guards),
    // and watched memory is asked about whole allocations at a time, so
    // the other buffers in one are answered from the journal for the rest
    // of the submission (see `new_submission`).
    let region = if on_wine() { allocation(start) } else { None };
    let mapped = region.is_some_and(|r| r.mapped);
    let (kfirst, klast) = match region {
        Some(r) if !r.mapped && r.whole && r.first <= first && last <= r.last => (r.first, r.last),
        _ => (first, last),
    };
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let ask = |kfirst: usize, klast: usize, found: &mut Vec<usize>| -> Option<usize> {
        let mut count = (klast - kfirst) * PAGE / KERNEL_PAGE;
        if found.len() < count {
            found.resize(count, 0);
        }
        let mut granularity = 0;
        // SAFETY: the answer buffer holds `count` entries.
        let ok = unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (kfirst * PAGE) as *const _,
                (klast - kfirst) * PAGE,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } == 0;
        ok.then_some(count)
    };
    let mut asked = (kfirst, klast);
    let mut answer = if mapped { None } else { ask(kfirst, klast, &mut j.found) };
    if answer.is_none() && !mapped && (kfirst, klast) != (first, last) {
        // The allocation was not one watched range after all (or is gone):
        // just ours, and look it up again next time.
        if let Some(r) = region {
            forget_region(r.base);
        }
        asked = (first, last);
        answer = ask(first, last, &mut j.found);
    }
    let failed = answer.is_none();
    let mut count = answer.unwrap_or(0);
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        match guard(first * PAGE, last * PAGE, mapped) {
            Some(g) => g,
            None => {
                if let Some(r) = region {
                    forget_region(r.base);
                }
                return None;
            }
        }
    } else {
        if on_wine() && j.asked_now.len() < 4096 {
            j.asked_now.push(asked);
        }
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    // A written page is committed: no hole any more.
    mapped_now(j.found[..count].iter().map(|&at| at / PAGE));
    let pages = j.found[..count].iter().map(|&at| at / PAGE).chain(guarded);
    for page in pages.collect::<Vec<_>>() {
        j.written.insert(page, now);
    }
    let out = j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect();
    Some((now, out))
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize, known_mapped: bool) -> Option<Vec<usize>> {
    use windows_sys::Win32::System::Memory::{MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery};
    if !known_mapped {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(start as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.Type != MEM_MAPPED {
            return None;
        }
    }
    install();
    let mut g = guards();
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let holes = holes_in(first, last);
    for block in first..=last {
        // A hole has nothing to guard or to copy. It ends the run, so the
        // blocks before it are protected in one call.
        let hole = holes.binary_search(&block).is_ok() && still_hole(block);
        #[cfg(feature = "diagnostics")]
        if hole {
            crate::copystat::add(crate::copystat::C::HolesQuiet, 1);
        }
        let open = block < last && !hole && !g.clean.contains(&block);
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail (unmapped).
fn protect(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        } else if unmapped(block) {
            note_hole(block);
        }
    }
}

/// Whether nothing of `block` is committed (one region of reserved or free
/// memory holds it all). A block that merely cannot be made read-only may
/// be mapped in part, which is no hole.
fn unmapped(block: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    n != 0
        && info.State != MEM_COMMIT
        && info.BaseAddress as usize + info.RegionSize >= (block + 1) * BLOCK
}

/// Title blocks ([`PAGE`] numbers) found with nothing mapped: the gaps
/// between xmem's views and the uncommitted parts of the title's pools.
/// Before they were remembered, a refresh over one read its run whole
/// (failing), then every block of the run again, then the hole in 16 pieces
/// (failing); in view memory holes were also reported written every time,
/// as they cannot be guarded.
///
/// A hole stops being one when
/// - the runtime maps or unmaps there ([`title_memory_remapped`]);
/// - the kernel reports a write to it (watched memory, [`writes`]);
/// - its first page can be read: every use of a hole asks that first
///   ([`still_hole`], one read where the walk made 17), so memory mapped
///   without either of the above is seen at the next refresh, as before.
///   Not asking until a notification would be cheaper still, but xmem's
///   are not the only mappings (the title commits parts of its pools
///   itself).
///
/// Committed memory that was never written reads as zeros, which is what a
/// hole is given, so watched memory stays exact. shortcut: in memory
/// neither watched nor mapped by xmem, pages committed in a hole after its
/// first page stay zeros until one of the above (xmem maps whole blocks).
static HOLES: Mutex<BTreeSet<usize>> = Mutex::new(BTreeSet::new());
/// Their number, so that the usual case (none) takes no lock.
static HOLES_KNOWN: AtomicUsize = AtomicUsize::new(0);

/// Never held while another lock is taken or title memory is touched.
fn holes() -> std::sync::MutexGuard<'static, BTreeSet<usize>> {
    HOLES.lock().unwrap_or_else(|e| e.into_inner())
}

/// `VICTORY_COPY_HOLES=0` remembers no holes (as before them).
fn holes_wanted() -> bool {
    static WANTED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("VICTORY_COPY_HOLES").as_deref() != Ok("0"));
    *WANTED
}

/// How many holes are known.
#[cfg(feature = "diagnostics")]
pub(crate) fn holes_known() -> usize {
    HOLES_KNOWN.load(Relaxed)
}

/// `block` was found with nothing mapped.
pub(crate) fn note_hole(block: usize) {
    if !holes_wanted() {
        return;
    }
    let mut holes = holes();
    if holes.insert(block) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::HolesFound, 1);
    }
    HOLES_KNOWN.store(holes.len(), Relaxed);
}

/// The known holes among blocks `[first, last)`, in order. They may have
/// been mapped since: ask [`still_hole`] before treating one as such.
pub(crate) fn holes_in(first: usize, last: usize) -> Vec<usize> {
    if HOLES_KNOWN.load(Relaxed) == 0 || first >= last {
        return Vec::new();
    }
    holes().range(first..last).copied().collect()
}

/// Whether the known hole `block` still has nothing mapped where it
/// starts; if not, it is forgotten.
pub(crate) fn still_hole(block: usize) -> bool {
    if !crate::host::readable(block * PAGE) {
        return true;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::HolesMapped, 1);
    mapped_now([block]);
    false
}

/// The blocks in `blocks` (repeats together) have mapped memory.
fn mapped_now(blocks: impl IntoIterator<Item = usize>) {
    if HOLES_KNOWN.load(Relaxed) == 0 {
        return;
    }
    let mut holes = holes();
    let mut last = usize::MAX;
    for block in blocks {
        if block != last {
            holes.remove(&block);
            last = block;
        }
    }
    HOLES_KNOWN.store(holes.len(), Relaxed);
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len));
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded, and a hole
/// there may be none now (a new one is found by the next copy).
pub fn title_memory_remapped(at: usize, len: usize) {
    REGIONS.lock().unwrap_or_else(|e| e.into_inner()).clear();
    let blocks = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    mapped_now(holes_in(blocks.start, blocks.end));
    let mut g = guards();
    let blocks: Vec<usize> = g
        .protected
        .range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK))
        .copied()
        .collect();
    for block in blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        // Under Wine a range is asked of the kernel once per submission.
        new_submission();
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// Under Wine, a write after a range was asked about in a submission
    /// is reported in the next one, not lost.
    #[test]
    fn writes_within_a_submission_show_in_the_next() {
        if !on_wine() {
            return;
        }
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        // Kept, not freed: another test given this address would see the
        // journal's entries for it. The journal may already hold some (a
        // freed allocation of another test here), so compare with them.
        new_submission();
        let (_, before) = writes(p, p + 2 * PAGE, false).expect("watched");
        // SAFETY: committed pages of ours.
        unsafe { ((p + 8) as *mut u8).write(1) };
        let (_, same) = writes(p, p + 2 * PAGE, false).expect("watched");
        assert_eq!(same, before, "answered from the journal within the submission");
        new_submission();
        let (now, next) = writes(p, p + 2 * PAGE, false).expect("watched");
        assert!(next.contains(&(p / PAGE, now)), "the write shows in the next submission");
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK, false).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(guard(view, view + 2 * BLOCK, false), Some((view / PAGE..view / PAGE + 1).collect()));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, 2 * BLOCK);
    }

    /// A reservation of 64 KiB blocks that takes views of a section block by
    /// block, as xmem maps physical pages (`XMemVirtualAlloc`'s placeholder,
    /// `map` and `unmap` there).
    struct Pages {
        base: usize,
        blocks: usize,
        section: windows_sys::Win32::Foundation::HANDLE,
    }

    impl Pages {
        fn new(blocks: usize) -> Pages {
            use windows_sys::Win32::System::Memory::{
                CreateFileMappingW, MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, VirtualAlloc2,
            };
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            // SAFETY: a fresh placeholder and a pagefile section of its size.
            let (base, section) = unsafe {
                let base = VirtualAlloc2(
                    GetCurrentProcess(),
                    std::ptr::null(),
                    blocks * BLOCK,
                    MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                    PAGE_NOACCESS,
                    std::ptr::null_mut(),
                    0,
                );
                let section = CreateFileMappingW(
                    -1isize as _,
                    std::ptr::null(),
                    PAGE_READWRITE,
                    0,
                    (blocks * BLOCK) as u32,
                    std::ptr::null(),
                );
                (base as usize, section)
            };
            assert!(base != 0 && !section.is_null());
            Pages { base, blocks, section }
        }

        fn at(&self, block: usize) -> usize {
            self.base + block * BLOCK
        }

        /// Maps the section's block `block` at its place, filled with
        /// `fill`; the runtime's notification only if `tell`.
        fn map(&self, block: usize, fill: u8, tell: bool) {
            use windows_sys::Win32::System::Memory::{
                MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER, MapViewOfFile3,
            };
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            let at = self.at(block);
            // SAFETY: splits the placeholder around the block (failing
            // harmlessly when it is one already) and maps the section there.
            let view = unsafe {
                VirtualFree(at as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                MapViewOfFile3(
                    self.section,
                    GetCurrentProcess(),
                    at as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, at, "{}", std::io::Error::last_os_error());
            if tell {
                title_memory_remapped(at, BLOCK);
            }
            // SAFETY: the view just mapped, read-write.
            unsafe { (at as *mut u8).write_bytes(fill, BLOCK) };
        }

        /// Returns `block` to a placeholder, telling first as the runtime does.
        fn unmap(&self, block: usize) {
            use windows_sys::Win32::System::Memory::{
                MEM_PRESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile2,
            };
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            let at = self.at(block);
            title_memory_remapped(at, BLOCK);
            let view = MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut _ };
            // SAFETY: a view we mapped; it becomes a placeholder again.
            assert_ne!(unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) }, 0);
        }

        /// The known holes, as block indices here.
        fn holes(&self) -> Vec<usize> {
            let first = self.base / BLOCK;
            holes_in(first, first + self.blocks).iter().map(|b| b - first).collect()
        }

        /// What a buffer copy's refresh does (`Shadow::copy_stale`): the
        /// blocks reported written by this asking are copied into `copy`,
        /// neighbours in one run. Returns them.
        fn refresh(&self, copy: &mut [u8]) -> Vec<usize> {
            let (now, written) =
                writes(self.base, self.base + self.blocks * BLOCK, true).expect("views are guarded");
            let stale: Vec<usize> = written
                .iter()
                .filter(|&&(_, asked)| asked == now)
                .map(|&(page, _)| page - self.base / PAGE)
                .collect();
            let mut runs: Vec<(usize, usize)> = Vec::new();
            for &block in &stale {
                match runs.last_mut() {
                    Some((_, end)) if *end == block => *end += 1,
                    _ => runs.push((block, block + 1)),
                }
            }
            for (from, to) in runs {
                let into = copy[from * BLOCK..to * BLOCK].as_mut_ptr();
                crate::host::copy_mapped(into, self.at(from) as *const u8, (to - from) * BLOCK, false);
            }
            stale
        }
    }

    impl Drop for Pages {
        fn drop(&mut self) {
            use windows_sys::Win32::Foundation::CloseHandle;
            // Forgets our guards and holes here: the address may be reused.
            title_memory_remapped(self.base, self.blocks * BLOCK);
            // SAFETY: our section; views still mapped keep their memory.
            unsafe { CloseHandle(self.section) };
        }
    }

    /// Failed reads and probes on this thread since the last call.
    fn reads() -> (usize, usize) {
        (crate::host::FAILED_READS.take(), crate::host::PROBES.take())
    }

    /// `copy`'s blocks each hold one value: those.
    fn blocks_of(copy: &[u8]) -> Vec<u8> {
        copy.chunks(BLOCK)
            .map(|block| {
                assert!(block.iter().all(|&b| b == block[0]));
                block[0]
            })
            .collect()
    }

    /// The gaps between physical-page views are found once; after that a
    /// refresh costs one read each, they are not reported written, and they
    /// are seen again when mapped, with or without the runtime saying so.
    #[test]
    fn holes_between_views_are_read_once() {
        if !holes_wanted() {
            return;
        }
        let pages = Pages::new(8);
        for block in [0, 2, 3, 6] {
            pages.map(block, block as u8 + 1, true);
        }
        let range = pages.blocks * BLOCK;
        let mut copy = vec![0xaa; range];
        reads();

        // The first refresh takes everything and finds the holes: by the
        // failed protection, then no failed walk is needed at all.
        new_submission();
        assert_eq!(pages.refresh(&mut copy), [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(blocks_of(&copy), [1, 0, 3, 4, 0, 0, 7, 0]);
        assert_eq!(pages.holes(), [1, 4, 5, 7]);
        assert_eq!(reads(), (0, 4));

        // Nothing changed: nothing is reported, the holes cost a probe.
        new_submission();
        assert_eq!(pages.refresh(&mut copy), []);
        assert_eq!(reads(), (0, 4), "(failed reads, probes)");

        // A copy that does not ask first (unwatched memory, textures)
        // reads around them, also when it starts or ends in one.
        copy.fill(0xaa);
        crate::host::copy_mapped(copy.as_mut_ptr(), pages.base as *const u8, range, false);
        assert_eq!(blocks_of(&copy), [1, 0, 3, 4, 0, 0, 7, 0]);
        assert_eq!(reads(), (0, 4));
        let mut part = vec![0xaa; 4 * BLOCK];
        crate::host::copy_mapped(part.as_mut_ptr(), (pages.at(1) + 100) as *const u8, 4 * BLOCK, false);
        assert!(part[..BLOCK - 100].iter().all(|&b| b == 0));
        assert!(part[BLOCK - 100..2 * BLOCK - 100].iter().all(|&b| b == 3));
        assert!(part[2 * BLOCK - 100..3 * BLOCK - 100].iter().all(|&b| b == 4));
        assert!(part[3 * BLOCK - 100..].iter().all(|&b| b == 0));
        assert_eq!(reads(), (0, 3));

        // A title write to a guarded block still shows.
        // SAFETY: a mapped view; the guard gives write access back.
        unsafe { (pages.at(2) as *mut u8).write_bytes(9, BLOCK) };
        new_submission();
        assert_eq!(pages.refresh(&mut copy), [2]);
        assert_eq!(blocks_of(&copy), [1, 0, 9, 4, 0, 0, 7, 0]);
        reads();

        // The runtime maps a hole and says so.
        pages.map(4, 5, true);
        assert_eq!(pages.holes(), [1, 5, 7]);
        new_submission();
        assert_eq!(pages.refresh(&mut copy), [4]);
        assert_eq!(blocks_of(&copy), [1, 0, 9, 4, 5, 0, 7, 0]);
        assert_eq!(reads(), (0, 3));

        // Memory appears in a hole without a word: the probe finds it.
        pages.map(5, 6, false);
        assert_eq!(pages.holes(), [1, 5, 7]);
        new_submission();
        assert_eq!(pages.refresh(&mut copy), [5]);
        assert_eq!(blocks_of(&copy), [1, 0, 9, 4, 5, 6, 7, 0]);
        assert_eq!(pages.holes(), [1, 7]);
        assert_eq!(reads(), (0, 3));
        // ... also by a copy that does not ask first.
        pages.map(7, 8, false);
        copy.fill(0xaa);
        crate::host::copy_mapped(copy.as_mut_ptr(), pages.base as *const u8, range, false);
        assert_eq!(blocks_of(&copy), [1, 0, 9, 4, 5, 6, 7, 8]);
        assert_eq!(pages.holes(), [1]);
        assert_eq!(reads(), (0, 2));

        // An unmapped view becomes a hole: zeros, found without a walk.
        // (Block 7 was never guarded, so it is taken again.)
        pages.unmap(2);
        new_submission();
        assert_eq!(pages.refresh(&mut copy), [2, 7]);
        assert_eq!(blocks_of(&copy), [1, 0, 0, 4, 5, 6, 7, 8]);
        assert_eq!(pages.holes(), [1, 2]);
        assert_eq!(reads(), (0, 2));

        for block in [0, 3, 4, 5, 6, 7] {
            pages.unmap(block);
        }
        drop(pages);
    }

    /// A copy that meets unmapped blocks without having asked the guard
    /// finds them by its walk, once.
    #[test]
    fn a_failed_walk_finds_the_holes() {
        let pages = Pages::new(8);
        for block in [0, 2, 3, 6] {
            pages.map(block, block as u8 + 1, true);
        }
        let range = pages.blocks * BLOCK;
        let mut copy = vec![0xaa; range];
        reads();
        crate::host::copy_mapped(copy.as_mut_ptr(), pages.base as *const u8, range, false);
        assert_eq!(blocks_of(&copy), [1, 0, 3, 4, 0, 0, 7, 0]);
        // The whole read, then each hole whole and in its 16 pages.
        assert_eq!(reads(), (1 + 4 * 17, 0));
        copy.fill(0xaa);
        if !holes_wanted() {
            // `VICTORY_COPY_HOLES=0`: the same again, every time.
            crate::host::copy_mapped(copy.as_mut_ptr(), pages.base as *const u8, range, false);
            assert_eq!(blocks_of(&copy), [1, 0, 3, 4, 0, 0, 7, 0]);
            assert_eq!(reads(), (1 + 4 * 17, 0));
            assert_eq!(pages.holes(), []);
            return;
        }
        assert_eq!(pages.holes(), [1, 4, 5, 7]);
        crate::host::copy_mapped(copy.as_mut_ptr(), pages.base as *const u8, range, false);
        assert_eq!(blocks_of(&copy), [1, 0, 3, 4, 0, 0, 7, 0]);
        assert_eq!(reads(), (0, 4));
        // Parts of two known holes.
        crate::host::copy_mapped(copy.as_mut_ptr(), (pages.at(4) + 4096) as *const u8, BLOCK, false);
        assert!(copy[..BLOCK].iter().all(|&b| b == 0));
        assert_eq!(reads(), (0, 2));
        for block in [0, 2, 3, 6] {
            pages.unmap(block);
        }
        drop(pages);
    }

    /// In watched memory a hole ends with the first write the kernel
    /// reports, wherever in the block the title committed.
    #[test]
    fn a_write_ends_a_hole_in_watched_memory() {
        if !holes_wanted() {
            return;
        }
        // SAFETY: a fresh write-watched reservation, freed at the end.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        let commit = |at: usize, len: usize, fill: u8| {
            // SAFETY: inside our reservation; committed before it is written.
            unsafe {
                assert!(!VirtualAlloc(at as *const _, len, MEM_COMMIT, PAGE_READWRITE).is_null());
                (at as *mut u8).write_bytes(fill, len);
            }
        };
        commit(p, PAGE, 1);
        commit(p + 2 * PAGE, PAGE, 3);
        let first = p / PAGE;
        let holes = || holes_in(first, first + 4).iter().map(|b| b - first).collect::<Vec<_>>();
        new_submission();
        writes(p, p + 4 * PAGE, false).expect("watched");
        let mut copy = vec![0xaa; 4 * PAGE];
        crate::host::copy_mapped(copy.as_mut_ptr(), p as *const u8, 4 * PAGE, false);
        assert_eq!(blocks_of(&copy), [1, 0, 3, 0]);
        assert_eq!(holes(), [1, 3]);
        // One page in the middle of a hole: its first page still fails.
        let page = p + PAGE + 8 * KERNEL_PAGE;
        commit(page, KERNEL_PAGE, 7);
        new_submission();
        let (now, written) = writes(p, p + 4 * PAGE, false).expect("watched");
        assert!(written.contains(&(first + 1, now)));
        assert_eq!(holes(), [3]);
        crate::host::copy_mapped(copy.as_mut_ptr(), p as *const u8, 4 * PAGE, false);
        let at = PAGE + 8 * KERNEL_PAGE;
        assert!(copy[PAGE..at].iter().all(|&b| b == 0));
        assert!(copy[at..at + KERNEL_PAGE].iter().all(|&b| b == 7));
        assert!(copy[at + KERNEL_PAGE..2 * PAGE].iter().all(|&b| b == 0));
        assert_eq!(holes(), [3], "a block mapped in part is no hole");
        title_memory_remapped(p, 4 * PAGE);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }
}
