# Linux status

Where VIctory stands on Linux under Proton, measured on one machine, and how
to find out what Wine costs before porting anything natively.

## Setup (2026-11-24)

| | |
|---|---|
| OS | Bazzite (Fedora-based), Linux 7.3.4 |
| GPU | AMD RX 7900 XT, RADV NAVI31, Mesa 26.2.7, Vulkan 1.4 |
| RAM | 31 GB |
| Wine | Proton Experimental `experimental-11.0-20261112` (ntsync) |
| D3D12 / DXGI | vkd3d-proton 3.2.0, build `cae8353ed` (from GE-Proton11-14), DXVK `dxgi.dll` from GE-Proton11-14 |
| d3dcompiler_47 | Wine's builtin |
| VIctory | `linux/bringup`: `ffe310e` + #892 + #844's `proton-fixes` (`993dff8`), `diag` build via `cargo xwin` |
| Title | 1.0.2.1, BLAKE3 `b99f876b…` (matches #752), Series X profile |

Command: `scripts/run-proton.sh --auto-enter --timeout 900 --mem-max 22G <game exe> --execute`

## Build and tests

| Check | Result |
|---|---|
| `cargo check --workspace` (Linux) | pass |
| `cargo clippy --workspace --all-targets -- -D warnings` (Linux) | pass |
| `cargo test --workspace` (Linux) | 234 passed, 0 failed, 5 ignored |
| `cargo xwin clippy --workspace --all-targets --target x86_64-pc-windows-msvc -- -D warnings` | pass |
| `cargo xwin build --release` / `--profile diag --features diagnostics` | pass |
| victory-gpu tests (Windows build, under Proton + vkd3d-proton 3.2 on the 7900 XT) | 52 passed, 0 failed, 1 ignored (WARP skipped); includes the argument clamp with Wine's builtin compiler |
| victory-loader / victory-pe tests (Windows build, under Wine) | 22/22 and 13/13; `load_probe` fails only because it runs `cargo` from inside Wine |

## How far it gets

`victory run` without `--execute`: every import binds (game exe: 88 HLE, 427
host, 72 title, 0 missing; its 4 DLLs likewise 0 missing).

With `--execute` (two runs, no crash in either, no `crashes.log`):

| Stage | Result |
|---|---|
| Boot, first-boot screen ("press A") | 30 fps, 33 ms/frame; waits for input |
| First-boot and start screens, intro | 22–30 fps; `--auto-enter` gets through them unattended |
| Loading into the world | reached ~85 s after start; loading artwork draws correctly |
| The world (first mission's opening cutscene) | runs at **0.1–1.5 fps** (log); mean 4.0 s/frame. The screen stays black, with snippets of the cutscene's audio (the scene is dark on Windows too) |
| End | run 1: our 5-minute cap, during the load. Run 2: we stopped it at 400 s, ~5 min into the world. No crash, no `crashes.log`, no amdgpu fault or ring reset in the kernel log |

The fps come from two sources. The window caption (`VIctory N fps …`,
sampled every 5 s) gave the menu and load figures. The diag build's
statistics lines gave the world figures.

Memory: about 13.8 GB for the process in the world. 452 shaders did not
translate (their draws are dropped; known on Windows too).

## Frame time in the world

These figures come from the diag build's statistics lines in run 2: 39 lines
over 5 min in the world, after loading. Each line covers one window of at
least 1 s, logged at most every 5 s. They are means of those windows' per-frame
values. This is one run of a cutscene, so treat the numbers as indicative.

| | ms/frame | share |
|---|---:|---:|
| **Whole frame** | **4,016** (median 3,733; 683–10,200) | 100 % |
| Preparing submissions: copies of title memory (273 MB) | 2,910 | 72 % |
| Preparing submissions: textures (detile, hash) | 282 | 7 % |
| Preparing submissions: rest | 173 | 4 % |
| Copying back (GPU-written ranges to title memory) | 217 | 5 % |
| Waiting for the GPU | 112 (median 65; max 313) | 3 % |
| Everything else (title CPU, waits, present) | 321 | 8 % |

The table counts 2,360 draws/frame submitted and 1,538 dropped (no host
pipeline). The GPU figure is therefore a lower bound for a complete frame.

### Inside the copies

The copy statistics (`copystat`, the same 39 windows, per frame):

- **About 8,500 copies and 11,300 refreshes per frame.** A refresh's "MB"
  in the log is the size of the range asked for (10 GB/frame), not bytes
  moved. A refresh copies only the pages the write watch reports as
  changed, and those bytes are part of the 273 MB. `host.rs` (`Batch::flush`,
  `shadow.update`) confirms it.
- **The timed parts add up to under 0.3 s of the 2.9 s:**

  | Part | ms/frame |
  |---|---:|
  | `ReadProcessMemory` calls (`moves`) | 60 |
  | Refreshes of the ten biggest ranges | 40 |
  | Write-watch queries | 161 |

  The other **~2.6 s is not attributed** by any counter.
- **The likely candidates are not timed separately.** By the code, they are
  the parts of `copy_mapped` around the timed read:
  - `texture::committed`, a `VirtualQuery` on title destinations;
  - `watch::writing`;
  - the **995 fallbacks per frame**: a copy that meets an unmapped page
    re-reads its range in 64 KiB and then 4 KiB pieces, and every failed
    piece is a failed `ReadProcessMemory`.

  Separately, the title takes 311 guard faults per frame on view memory.
  Under Wine every one of these goes through a signal handler or Wine's
  virtual-memory lock. That is a hypothesis to test, not a measurement.
- **Ten title memory pages are checked per byte that has changed.** Of the
  10.1 GB of watched memory checked per frame, 97.5 % is unchanged.

A run with `VICTORY_COPY_MODE=plain` (copies as a plain `memcpy`) was
meant to measure the whole copy path at once. It cannot: the plain copy
read a reserved, never-committed title page as the world began loading, and
the process ended (run 3, about 95 s in). The unmapped-page handling is
needed, so its cost has to be measured, not switched off.

## Splitting the frame: GPU, VIctory's copies, Wine

The table already separates the GPU (3 %) from VIctory's CPU-side work
(about 90 %). Two things are still open. Which part of the copy path takes
the unattributed 2.6 s? How much of that is Wine's emulation of cheap Windows
calls rather than VIctory's own work? Proposed measurements, in order:

1. **A profile of the process in the world**, first because it answers both
   questions at once and needs no code change:
   - Run `perf record -g --call-graph dwarf` for 60 s on the `victory.exe`
     process and on `wineserver`.
   - Group the samples. Wine's `ntdll.so`/`win32u.so`, wineserver and the
     kernel's fault, signal and `process_vm_*` paths count as Wine.
     `victory.exe` counts as VIctory. RADV and vkd3d-proton count as the
     driver.
   - Within `victory.exe`, the samples under `copy_mapped` show which of
     `committed`, `writing` and the fallback walk dominates.
2. **Timers for the untimed parts**, diagnostics only:
   - add `copystat` counters for the time in `texture::committed`,
     `watch::writing` and the fallback walk;
   - count the pieces the fallback walk reads.

   Those counters then work identically on Windows, which turns the next
   step into a like-for-like comparison.
3. **The same scene on Windows** on this machine, if a Windows install is
   available. Compare the copy counters category by category; the ratio is
   Wine's factor per category.
4. **Micro-benchmarks, Windows against this Proton**, in a loop:
   `ReadProcessMemory` of own memory, including a range that hits an
   unmapped page; `VirtualQuery` in a big reserved pool; a guard-page fault
   round-trip; `GetWriteWatch` on 256 MB.

## What should go native first

**Not the renderer.** The GPU takes 3 % of the frame and vkd3d-proton is not
on the critical path. A Vulkan backend would gain nothing measurable today.

**Nothing yet.** The measurements so far say "the copy path", not which
call in it, and the obvious candidate doesn't fit. Replacing write watching
(`GetWriteWatch`) with a Linux mechanism (the `PAGEMAP_SCAN` ioctl or
userfaultfd write-protect, via a small Wine unixlib) could save at most the
161 ms the counters see. That is unless its cost lands on the title's
threads as faults, which only the profile can show.

The order instead:

1. Profile (step 1 above).
2. If the fallback walk dominates, the fix is in VIctory and helps Windows
   too. Remember which ranges contain unmapped pages, and skip them
   per page, instead of failing a whole read and retrying in pieces.
3. If `VirtualQuery` or the fault path dominates under Wine only, that call
   is the first candidate for a native path. It would be a Wine unixlib that
   answers "is this range committed" and "which pages changed" from
   `/proc/self/pagemap` or `PAGEMAP_SCAN`, with no signals. Everything else
   stays the Windows build.

Re-measure after each change with the statistics lines above.

Done since, not yet measured in the world:

- **The timers of step 2 exist.** The diagnostics build logs a second
  line, "copy path per frame", next to "copies per frame": title copies
  and fills apart, the time of all merged refreshes, failed reads and the
  time in the fallback walk, `texture::committed` and `watch::writing` on
  title destinations, allocation lookups, write-watch askings answered
  from the journal, and the holes below.
- **The copy time was counted twice in part.** A copy into title memory
  first does the refreshes waiting before it; their time went into "copies"
  once on their own and once more with the copy. The 2.9 s above includes
  that.
- **Unmapped blocks are remembered** (`watch::HOLES`, off with
  `VICTORY_COPY_HOLES=0`). A 64 KiB block with nothing mapped used to be
  reported written at every refresh of view memory (it cannot be guarded).
  The copy then read its run whole, failing, then block by block, then the
  hole in 16 pieces, failing. Now a known hole costs one one-byte read per
  use, which also notices memory mapped there since.

## Unified memory: can the GPU use title memory in place? (probe, 2026-11-24)

The Xbox shares one memory pool between CPU and GPU. VIctory instead keeps
host GPU copies of title memory and keeps them up to date. A synthetic
probe answered whether the GPU can use title memory in place on this
machine (`crates/victory-gpu/src/inplace_probe.rs`, ignored tests, no title
data). Setup: RX 7900 XT, vkd3d-proton 3.2.0 `cae8353`, Proton
Experimental.

Opening title memory as a D3D12 heap (`OpenExistingHeapFromAddress*`):

| Memory | Result |
|---|---|
| `VirtualAlloc` (with or without `MEM_WRITE_WATCH`), 64 MB to 8 GB, and page-aligned sub-ranges | **in place**: both ways, no copies, no 4 GiB ceiling |
| Any section view (`MapViewOfFile`, `MapViewOfFile3` into a placeholder, as xmem's physical pages), and `OpenExistingHeapFromFileMapping` | `S_OK`, but **not in place** |
| A reservation with only part committed | `E_INVALIDARG` |

Why sections fail: vkd3d-proton asks RADV to import the memory, and the
amdgpu kernel driver only takes anonymous memory (`AMDGPU_GEM_USERPTR_ANONONLY`).
Wine's sections are shared file mappings. vkd3d-proton then falls back to
memory of its own and still returns `S_OK`. Any use of this must be checked
by a round trip, never by the HRESULT.

Two more constraints:
- Placed buffers in such a heap need `ALLOW_CROSS_ADAPTER`.
- Write watch stops being useful on imported memory: every submission marks
  the whole heap written.

Bandwidth for the GPU, 256 MB, GB/s:

| Memory | seq read | scattered read | seq write | scattered write |
|---|---:|---:|---:|---:|
| Imported `VirtualAlloc` / CUSTOM WRITE_BACK L0 (system memory) | 2.3 | 0.8 | 4.5–7.9 | 0.8 |
| VRAM (DEFAULT, or GPU_UPLOAD) | 108 | 27 | 79 | 24 |

CPU access: imported memory and CUSTOM heaps run at full speed (~15 GB/s).
**GPU_UPLOAD** (VRAM the CPU maps; Resizable BAR is on here) writes at 22 GB/s
but reads at **20 MB/s**.

Consequences:
- **GPU-only data belongs in VRAM**: what the GPU writes and the CPU never
  reads (render targets already; most UAV-written buffers). It should stop
  being copied back at all.
- **CPU-written, GPU-read data can be shared without copies**, but only
  through memory VIctory allocates:
  - private `VirtualAlloc`, imported;
  - or a CPU-visible arena, as upstream #407 proposes;
  - or GPU_UPLOAD, only for memory the title never reads back.

  xmem's section-backed physical pages cannot be imported under Wine, so
  they would have to move to such an arena.
- **System memory is 30–45× slower for the GPU than VRAM.** In-place
  access suits data written each frame and read once. Hot buffers still
  want VRAM.

## Holes cache: measured (2026-11-24)

The holes cache (`VICTORY_COPY_HOLES`, on by default) remembers 64 KiB title
blocks with nothing mapped. It stops copying them and stops reporting them
written. Two runs on the same build, same setup as above, 7 minutes each,
`--auto-enter`. Means over the world's statistics lines, first four after
loading skipped:

| Per frame | `VICTORY_COPY_HOLES=0` | default (cache on) |
|---|---:|---:|
| Whole frame (median) | 6,054 ms (5,333) | **3,711 ms (3,100)** |
| Copies | 4,546 ms, 245 MB | **1,181 ms**, 40 MB |
| Failed `ReadProcessMemory` / fallback walk | 28,578 / 2,626 ms | **0 / 0 ms** |
| Merged refreshes | 1,731, 3,114 ms | 1,306, 346 ms |
| Allocation lookups (`VirtualQuery` walks) | 788 ms | 407 ms |
| Hole probes | – | 859, 74 ms |
| GPU | 180 ms | 126 ms |
| Everything else (title CPU, waits) | 440 ms | 1,149 ms |

The copies fell by three quarters, and the unmapped-page walk was nearly
all of what they cost. The whole frame fell less: in the run with the cache
on, "everything else" grew. That part is untimed by VIctory. It is either
the cutscene reaching a heavier part in the same 7 minutes, or the title's
own CPU work now being visible. These are single runs, so the split is
indicative.

The next copy-path cost is `allocation()`: about 400 ms per frame of
`VirtualQuery` walks, which take Wine's virtual-memory lock. Stage 3 answers
them from xmem's own records.

## Allocation lookups from xmem's records (not yet measured in the world)

`watch::allocation()` asked the host which allocation an address is in: one
`VirtualQuery` per region of the allocation, up to 256, and every map or
unmap of physical pages emptied the cache of answers. Now:

- **xmem tells the GPU side what it allocates** (`watch/records.rs`): each
  reservation with its size and kind (write-watched, a placeholder for
  views, or neither), and each view as it is mapped and unmapped. Lookups
  inside those are answered from the records with no `VirtualQuery`.
- **Only memory xmem did not allocate is still walked** (the title's own
  `VirtualAlloc`, which kernel32 passes through). Those answers are cached
  as before.
- **A remap drops only the cached answers it overlaps**, not all of them.
- The title frees xmem's reservations with the host's `VirtualFree`, unseen.
  A record left behind is dropped when the host refuses the write-watch
  asking it leads to, as a stale cache entry was.

One answer differs from the walk's. A watched reservation of at most 4 GiB
with more than 256 regions used to be asked about piece by piece. It is now
asked about whole, once per submission. If "watches … ms" grows in the
statistics line, this is why.

The "copy path per frame" line shows the split: "N allocation lookups (R
from records, W walked)". The rest were walked before and remembered.

## GPU-owned title memory (stage 2, `VICTORY_LAZY_COPY_BACK=1`)

Off by default; not yet measured in the world. Code:
`crates/victory-gpu/src/owned.rs`, the `…_owned` paths of `shadow.rs`,
`crates/victory-hle/src/modules/vmem.rs`.

Without the switch, every range a submission's shaders may write is kept
before the submission (a snapshot), compared after it, and what differs is
written back into title memory, read or not. With it, a 64 KiB block of such
a range is not copied back: the host copy is the block's only current
holder, the title's pages are `PAGE_NOACCESS`, and refreshes skip the block.
The title's first access (read or write) faults; the handler writes the
copy's 64 KiB into the block, restores access, and the block is the
title's again. The GPU using it again before that costs nothing: no
snapshot, no comparison, no refresh, no write-watch asking.

What it does not do, and why:

| Case | Handling |
|---|---|
| Physical-page views (xmem's sections), the only memory with aliases (`map_alias`, `CopyPageMappingsBatchX`) | never owned: protecting one view leaves the others open, and views are not write-watched. They keep the copy-back. The line's "unwatched (views)" MB is how much that leaves |
| Ranges far larger than what shaders write (views of the whole 174 MB heap, in which the title's threads write command memory meanwhile) | per block, not per range. A block is taken only if no write-watch bit is set before and after its pages are closed, and the copy is current by every write reported; else it is compared and copied back as before |
| Memory the CPU works in | a touched block rests for 8 submissions, doubling with every touch up to 4096; a block found written at take time counts as touched |
| The title touches a block while the submission that writes it runs | the block goes back with the copy's bytes of that moment, which are kept; after the submission what differs from them is merged into title memory. The handler never waits |
| Our own reads (`copy_mapped`: buffer copies, uploads, another copy's first use) | served from the owning copy; the block stays owned |
| Our own kernel writes, kernel reads for our records, texture and descriptor-heap refreshes (which rely on the write watch), another copy's refresh | the block is given back first |
| The title's `ReadFile`, `fread`, DirectStorage destinations, `WriteFile`, `fwrite` | given back first (`kernel_will_write`, `title_will_touch`) |
| The title's `VirtualAlloc` over owned pages (resets their protection), `VirtualProtect`, `XMemVirtualAlloc` at an address | given back first |
| `VirtualFree`, `XMemFreeDefault`, xmem's map and unmap | ownership dropped, nothing copied; a block only part of which is freed is given back with its bytes; a block found gone when it is to be given back is dropped and counted ("lost") |
| Bottom-of-pipe writes, ranges written through two copies in one submission | their blocks are not taken |

Known gaps:

- **On Windows itself** giving a block back is not atomic for the title's
  other threads: a closed page takes no write, so access comes first and
  the 64 KiB after, and a thread using the block in between reads older
  bytes or loses a write of its own. A test with 8 threads released on one
  block failed that way once in four runs under Proton before the next
  point.
- **Under Wine** the bytes land first: the Linux kernel's file of the
  process's memory (`/proc/<pid>/mem`) writes closed pages. Wine's server
  opens files, so `/proc/self` is the server; our own process is found
  once: `/proc` is listed and each process's file the server may open is
  opened for reading and read at the address of a marked page of ours
  (16 bytes); only ours is then opened for writing. If the kernel or the container refuses
  (`proc_mem.force_override`, ptrace limits) the Windows order is used; the
  log says which ("title memory through the kernel's file of it").
- Other host calls that hand title memory to the kernel (sockets,
  overlapped I/O, `VirtualAlloc2`, `Nt…` calls bound to the host) fail on
  an owned block instead of faulting. None is known to get GPU memory.
- Under Wine a touch costs a signal, two protection changes and a 64 KiB
  write through the kernel: 43 µs in the handler per touch (debug build, 63
  touches), the signal and exception dispatch not counted.

Primitives measured under Proton Experimental (256 MiB, write-watched):
a non-resetting `GetWriteWatch` over all of it 0.45 ms, over one block
3.5 µs; `PAGE_NOACCESS` over all of it 19 ms, one block back 6 µs;
`WriteProcessMemory` of 64 KiB 95 µs. `PAGE_NOACCESS` and back neither
set nor lose write-watch bits; `VirtualAlloc(MEM_COMMIT)` over a closed
page opens it. A write through the kernel's file is reported by the write
watch like any other, so a block given back is copied once more at its next
refresh (the same bytes).

The diagnostics build logs "lazy copy-back per frame": ranges and MB made
GPU-owned, MB of copy-back avoided, CPU touches with MB and time, blocks
given back for our own use, MB served from the owning copy, blocks dropped
and lost, MB not taken (unwatched, resting, written by the title), MB owned
now, and the census: MB ever GPU-owned and how much of it the CPU ever
touched.
