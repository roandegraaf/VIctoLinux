//! Title memory the runtime allocated itself (`victory-hle` xmem), as it
//! tells us: allocation lookups there are answered from these records, not
//! by asking the host. Under Wine `VirtualQuery` takes the lock the title's
//! write faults take, and walking an allocation's regions made up to 256 of
//! them per lookup (400 to 800 ms of a frame in the world).
//!
//! The title frees this memory with the host's `VirtualFree`, which we do
//! not see, so a record can outlive its allocation. A record therefore only
//! claims what is checked again or kept exact:
//! - a write-watched reservation is asked about whole, and when the host
//!   refuses that asking the record is dropped ([`forget_watched`]);
//! - views are those the runtime has mapped now, told as it maps and unmaps;
//! - anything else (the gaps between views, unwatched reservations) is
//!   answered "not a view, not to be asked whole", which is what an unknown
//!   address gets, without the walk.

use std::collections::BTreeMap;
use std::sync::Mutex;

use super::{PAGE, Region};

/// What the runtime reserved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TitleMemory {
    /// Reserved with write watching.
    Watched,
    /// A placeholder that takes views of physical pages.
    Views,
    /// Neither.
    Plain,
}

struct Records {
    /// Reservations by base: their length and kind.
    reserved: BTreeMap<usize, (usize, TitleMemory)>,
    /// Mapped views by base: their length.
    views: BTreeMap<usize, usize>,
}

/// Never held while another lock is taken: the runtime tells us under its
/// own lock, and lookups come under the journal's.
static RECORDS: Mutex<Records> = Mutex::new(Records {
    reserved: BTreeMap::new(),
    views: BTreeMap::new(),
});

fn records() -> std::sync::MutexGuard<'static, Records> {
    RECORDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The entry of `map` (by base; `len` gives an entry's length) holding `at`.
fn holding<T: Copy>(
    map: &BTreeMap<usize, T>,
    at: usize,
    len: impl Fn(T) -> usize,
) -> Option<(usize, T)> {
    let (&base, &entry) = map.range(..=at).next_back()?;
    (at - base < len(entry)).then_some((base, entry))
}

/// The runtime reserved title memory `[base, base + len)` (`len` as the
/// host rounds it, to its 4 KiB pages). Whatever was recorded there is gone.
pub fn title_memory_allocated(base: usize, len: usize, kind: TitleMemory) {
    let end = base.saturating_add(len);
    {
        let mut r = records();
        r.reserved
            .retain(|&b, &mut (l, _)| b >= end || b.saturating_add(l) <= base);
        r.views
            .retain(|&b, &mut l| b >= end || b.saturating_add(l) <= base);
        r.reserved.insert(base, (len, kind));
    }
    // What a walk found there before describes older memory.
    super::forget_regions(base / PAGE, end.div_ceil(PAGE));
}

/// The runtime released the reservation at `base`.
pub fn title_memory_freed(base: usize) {
    let mut r = records();
    if let Some((len, _)) = r.reserved.remove(&base) {
        r.views
            .retain(|&b, _| b < base || b >= base.saturating_add(len));
    }
}

/// The runtime mapped a view at `[at, at + len)`, in place of a
/// placeholder. Includes [`super::title_memory_remapped`].
pub fn title_view_mapped(at: usize, len: usize) {
    records().views.insert(at, len);
    super::title_memory_remapped(at, len);
}

/// The runtime is about to unmap its view at `[at, at + len)`. Includes
/// [`super::title_memory_remapped`].
pub fn title_view_unmapped(at: usize, len: usize) {
    records().views.remove(&at);
    super::title_memory_remapped(at, len);
}

/// The record holding `at`: its base, its length, and the kind of
/// reservation it is, or `None` for a mapped view.
pub fn title_memory_record(at: usize) -> Option<(usize, usize, Option<TitleMemory>)> {
    let r = records();
    if let Some((base, len)) = holding(&r.views, at, |len| len) {
        return Some((base, len, None));
    }
    let (base, (len, kind)) = holding(&r.reserved, at, |(len, _)| len)?;
    Some((base, len, Some(kind)))
}

/// What the records say of `at`: `None` if the runtime did not allocate
/// it; else what [`super::walk`] would answer (`Some(None)`: nothing to
/// remember), but for
/// - a watched reservation of more regions than the walk gets through: it
///   is known whole here;
/// - memory neither watched nor a view: its one page, not to be asked
///   about whole (that asking would fail; see `writes`).
pub(super) fn find(at: usize) -> Option<Option<Region>> {
    let region = |base: usize, len: usize, mapped: bool| {
        let region = Region {
            base,
            first: base.div_ceil(PAGE),
            last: (base + len) / PAGE,
            mapped,
            whole: len <= 4 << 30,
        };
        (region.first * PAGE <= at && region.last > region.first).then_some(region)
    };
    let (base, len, kind) = title_memory_record(at)?;
    Some(match kind {
        None => region(base, len, true),
        Some(TitleMemory::Watched) => region(base, len, false),
        Some(TitleMemory::Views | TitleMemory::Plain) => Some(Region {
            base,
            first: at / PAGE,
            last: at / PAGE + 1,
            mapped: false,
            whole: false,
        }),
    })
}

/// The host refused an asking about the allocation at `base`: if that was
/// a watched reservation of ours, it is gone.
pub(super) fn forget_watched(base: usize) {
    let mut r = records();
    if r.reserved
        .get(&base)
        .is_some_and(|&(_, kind)| kind == TitleMemory::Watched)
    {
        r.reserved.remove(&base);
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        QUERIES, allocation, forget_region, forget_regions, new_submission, on_wine,
        title_memory_remapped, walk, writes,
    };
    use super::*;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, MEM_COMMIT, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE,
        MEM_REPLACE_PLACEHOLDER, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS,
        MapViewOfFile3, PAGE_NOACCESS, PAGE_READWRITE, UnmapViewOfFile2, VirtualAlloc,
        VirtualAlloc2, VirtualFree,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// `VirtualQuery` calls of lookups on this thread since the last call.
    fn queries() -> usize {
        QUERIES.take()
    }

    /// The lookup of `at`, which the records must answer without asking the
    /// host, and as the host's walk does: the same allocation if it is
    /// watched or a view, and otherwise (no extent is recorded) no view.
    fn recorded(at: usize) -> Option<Region> {
        queries();
        let region = allocation(at);
        assert_eq!(queries(), 0, "answered from the records");
        let walked = walk(at);
        match region {
            Some(r) if r.mapped || r.whole => assert_eq!(Some(r), walked),
            Some(r) => {
                assert_eq!((r.first, r.last), (at / PAGE, at / PAGE + 1));
                assert!(!walked.is_some_and(|w| w.mapped), "{walked:?}");
            }
            None => assert_eq!(walked, None),
        }
        queries();
        region
    }

    fn reserve(len: usize, kind: u32) -> usize {
        // SAFETY: a fresh allocation, freed by the test.
        let p = unsafe { VirtualAlloc(std::ptr::null(), len, kind, PAGE_READWRITE) } as usize;
        assert_ne!(p, 0);
        p
    }

    fn free(p: usize) {
        // SAFETY: an allocation of the test's.
        assert_ne!(unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) }, 0);
    }

    /// A reservation as `XMemVirtualAlloc` makes them for the title's
    /// pools: watched, committed in parts.
    #[test]
    fn a_watched_reservation_is_answered_whole() {
        let p = reserve(6 * PAGE, MEM_RESERVE | MEM_WRITE_WATCH);
        for block in [0, 2, 4] {
            let at = (p + block * PAGE) as *const _;
            // SAFETY: inside our reservation.
            assert!(!unsafe { VirtualAlloc(at, PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        }
        title_memory_allocated(p, 6 * PAGE, TitleMemory::Watched);
        let whole = Region {
            base: p,
            first: p / PAGE,
            last: p / PAGE + 6,
            mapped: false,
            whole: true,
        };
        assert_eq!(recorded(p), Some(whole));
        assert_eq!(recorded(p + 3 * PAGE + 100), Some(whole));
        assert_eq!(recorded(p + 6 * PAGE - 1), Some(whole));
        // Under Wine an asking covers the whole reservation, so one about
        // other pages of it in the same submission finds the answer kept.
        // SAFETY: a committed page of ours.
        unsafe { ((p + 4 * PAGE + 8) as *mut u8).write(1) };
        new_submission();
        let (now, _) = writes(p, p + PAGE, false).expect("watched");
        let (_, later) = writes(p + 4 * PAGE, p + 5 * PAGE, false).expect("watched");
        assert_eq!(later.contains(&(p / PAGE + 4, now)), on_wine());
        assert_eq!(queries(), 0);
        title_memory_freed(p);
        free(p);
    }

    /// Less than a 64 KiB block (`XMemAlloc`'s 32-bit memory): no whole
    /// block to remember, as the walk finds, without the walk.
    #[test]
    fn a_reservation_under_a_block_has_no_region() {
        let p = reserve(4096, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH);
        title_memory_allocated(p, 4096, TitleMemory::Watched);
        assert_eq!(recorded(p), None);
        title_memory_freed(p);
        free(p);
    }

    /// The title frees its memory with the host's `VirtualFree`, unseen:
    /// the record goes when the host refuses the asking it leads to.
    #[test]
    fn a_reservation_freed_unseen_is_forgotten() {
        let p = reserve(4 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH);
        title_memory_allocated(p, 4 * PAGE, TitleMemory::Watched);
        assert!(recorded(p).is_some());
        free(p);
        // Unwatched memory of the title's own in its place (unless the
        // host gave the address away meanwhile).
        let kind = MEM_RESERVE | MEM_COMMIT;
        // SAFETY: a fresh allocation where ours was, freed below.
        let q = unsafe { VirtualAlloc(p as *const _, 2 * PAGE, kind, PAGE_READWRITE) } as usize;
        new_submission();
        if on_wine() {
            assert!(writes(p, p + PAGE, false).is_none(), "nothing watched here");
        } else {
            // Only Wine's askings look allocations up.
            forget_region(p);
        }
        assert_eq!(find(p), None);
        if q != 0 {
            queries();
            let region = allocation(p).expect("an allocation");
            assert!(queries() > 0, "asked of the host");
            assert_eq!(
                (region.base, region.last, region.mapped),
                (p, p / PAGE + 2, false)
            );
            forget_regions(p / PAGE, p / PAGE + 2);
            free(q);
        }
    }

    /// Told of the release (`XMemFree` of 32-bit memory), the record goes
    /// at once.
    #[test]
    fn a_freed_reservation_is_not_answered() {
        let p = reserve(2 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH);
        title_memory_allocated(p, 2 * PAGE, TitleMemory::Watched);
        assert!(recorded(p + PAGE).is_some());
        title_memory_freed(p);
        assert_eq!(find(p + PAGE), None);
        free(p);
    }

    /// A placeholder taking views of a section block by block, as xmem's
    /// reservations of 64 KiB pages do (`map` and `unmap` there).
    struct Placeholder {
        base: usize,
        blocks: usize,
        section: HANDLE,
    }

    impl Placeholder {
        fn new(blocks: usize) -> Placeholder {
            // SAFETY: a fresh placeholder and a pagefile section of its size.
            let (base, section) = unsafe {
                let base = VirtualAlloc2(
                    GetCurrentProcess(),
                    std::ptr::null(),
                    blocks * PAGE,
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
                    (blocks * PAGE) as u32,
                    std::ptr::null(),
                );
                (base as usize, section)
            };
            assert!(base != 0 && !section.is_null());
            title_memory_allocated(base, blocks * PAGE, TitleMemory::Views);
            Placeholder {
                base,
                blocks,
                section,
            }
        }

        fn at(&self, block: usize) -> usize {
            self.base + block * PAGE
        }

        /// One view of `n` blocks from `block` on.
        fn map(&self, block: usize, n: usize) {
            let at = self.at(block);
            // SAFETY: splits the placeholder around the blocks (failing
            // harmlessly when they are a whole one) and maps the section.
            let view = unsafe {
                VirtualFree(
                    at as *mut _,
                    n * PAGE,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    self.section,
                    GetCurrentProcess(),
                    at as *const _,
                    (block * PAGE) as u64,
                    n * PAGE,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(
                view.Value as usize,
                at,
                "{}",
                std::io::Error::last_os_error()
            );
            title_view_mapped(at, n * PAGE);
        }

        fn unmap(&self, block: usize, n: usize) {
            let at = self.at(block);
            title_view_unmapped(at, n * PAGE);
            let view = MEMORY_MAPPED_VIEW_ADDRESS {
                Value: at as *mut _,
            };
            // SAFETY: a view we mapped; it becomes a placeholder again.
            let unmapped =
                unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
            assert_ne!(unmapped, 0);
        }
    }

    impl Drop for Placeholder {
        fn drop(&mut self) {
            title_memory_freed(self.base);
            title_memory_remapped(self.base, self.blocks * PAGE);
            // SAFETY: our section. The placeholder's pieces stay reserved,
            // so no other test is given their addresses.
            unsafe { CloseHandle(self.section) };
        }
    }

    #[test]
    fn views_in_a_placeholder_are_answered_one_by_one() {
        let pages = Placeholder::new(6);
        pages.map(1, 1);
        pages.map(2, 1);
        pages.map(4, 2);
        let view = |block: usize, n: usize| {
            Some(Region {
                base: pages.at(block),
                first: pages.at(block) / PAGE,
                last: pages.at(block) / PAGE + n,
                mapped: true,
                whole: true,
            })
        };
        assert_eq!(recorded(pages.at(1)), view(1, 1));
        assert_eq!(recorded(pages.at(2) + 100), view(2, 1));
        assert_eq!(recorded(pages.at(4)), view(4, 2));
        assert_eq!(recorded(pages.at(5) + PAGE - 1), view(4, 2));
        // The gaps: no views, and nothing to ask the host about whole.
        for block in [0, 3] {
            let gap = recorded(pages.at(block)).expect("in the placeholder");
            assert!(!gap.mapped && !gap.whole);
        }
        // A copy's asking takes the views by their guards, the lookup
        // costing no query; one that starts in a gap is not answered (as
        // before the records).
        new_submission();
        let (_, written) = writes(pages.at(1), pages.at(3), true).expect("views are guarded");
        assert_eq!(written.len(), 2);
        assert!(writes(pages.at(0), pages.at(1), true).is_none());
        assert_eq!(queries(), 0);
        // Failed askings in a gap do not cost the placeholder its record,
        // and guarded views are the same allocations.
        assert!(recorded(pages.at(3)).is_some());
        assert_eq!(recorded(pages.at(1)), view(1, 1));

        pages.unmap(1, 1);
        let gap = recorded(pages.at(1)).expect("in the placeholder");
        assert!(!gap.mapped && !gap.whole);
        assert_eq!(recorded(pages.at(2)), view(2, 1));
        // Part of a view unmapped, as xmem does it: all of it, then the
        // rest mapped again.
        pages.unmap(4, 2);
        pages.map(5, 1);
        assert!(!recorded(pages.at(4)).expect("in the placeholder").mapped);
        assert_eq!(recorded(pages.at(5)), view(5, 1));
        pages.unmap(2, 1);
        pages.unmap(5, 1);
    }

    /// Memory the runtime was not asked for (the title's own
    /// `VirtualAlloc`): walked once, then remembered until a remap there.
    #[test]
    fn unrecorded_memory_is_walked_and_remembered() {
        let p = reserve(3 * PAGE, MEM_RESERVE | MEM_COMMIT);
        let other = reserve(PAGE, MEM_RESERVE | MEM_COMMIT);
        assert_eq!(find(p), None);
        queries();
        let region = allocation(p + PAGE).expect("an allocation");
        assert!(queries() >= 2, "walked");
        assert_eq!(Some(region), walk(p + PAGE));
        let whole = Region {
            base: p,
            first: p / PAGE,
            last: p / PAGE + 3,
            mapped: false,
            whole: true,
        };
        assert_eq!(region, whole);
        queries();
        assert_eq!(allocation(p + 2 * PAGE), Some(region));
        assert_eq!(queries(), 0, "remembered");
        // A remap elsewhere leaves it; one there does not.
        title_memory_remapped(other, PAGE);
        assert_eq!(allocation(p), Some(region));
        assert_eq!(queries(), 0, "still remembered");
        title_memory_remapped(p + 2 * PAGE, PAGE);
        assert_eq!(allocation(p), Some(region));
        assert!(queries() >= 2, "walked again");
        // The runtime reserving there (after a free we did not see) ends
        // what the walk remembered.
        title_memory_allocated(p, 3 * PAGE, TitleMemory::Plain);
        let plain = recorded(p + PAGE).expect("recorded");
        assert!(!plain.mapped && !plain.whole);
        title_memory_freed(p);
        assert_eq!(allocation(p), Some(region));
        assert!(queries() >= 2, "not remembered from before the record");
        forget_regions(p / PAGE, p / PAGE + 3);
        free(p);
        free(other);
    }
}
