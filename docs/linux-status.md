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
| The world (first mission's opening cutscene) | runs, **0.1–1.5 fps** (mean 3.8 s/frame); the screen is black, with the cutscene's audio in snippets (dark scene as on Windows) |
| End | stopped by us after ~5 min in the world (run 1: cap at 5 min, during loading) |

Memory: about 13.8 GB for the process in the world. 452 shaders did not
translate (their draws are dropped; known on Windows too).

## Frame time in the world

From the diag build's 5-second statistics lines, 46 samples over 5 min 15 s
in the world (mean per frame; one run, a cutscene not driven identically
each time, so indicative):

| | ms/frame | share |
|---|---:|---:|
| **Whole frame** | **3,795** (median 3,367; 683–10,200) | 100 % |
| Preparing submissions: copies of title memory (263 MB) | 2,736 | 72 % |
| Preparing submissions: textures (detile, hash) | 273 | 7 % |
| Preparing submissions: rest | 147 | 4 % |
| Copying back (GPU-written ranges to title memory) | 220 | 6 % |
| Waiting for the GPU | 106 (median 65; max 313) | 3 % |
| Everything else (title CPU, waits, present) | 313 | 8 % |

2,169 draws/frame were submitted and 1,476 dropped (no host pipeline), so
the GPU figure is a lower bound for a complete frame.

The copy statistics (`copystat`) show what the copies are made of, per frame:

- 263 MB copied in 2.7 s: about 96 MB/s. That is per-call and per-page cost,
  not memory bandwidth (a plain `memcpy` does that in under 30 ms).
- 7.4 GB of watched title memory checked, 97 % of it unchanged; 122 ms in
  write-watch queries.
- 687 copies falling back after meeting an unmapped page, 233 guard faults
  on view memory, 46 ms in moves.

## Where the time goes: GPU, VIctory's copies, Wine

The table already separates the GPU (3 %) from VIctory's CPU-side work
(~90 %). What it cannot say is how much of the copies is Wine's doing. The
copy path calls `GetWriteWatch`, takes guard-page and access faults, and
falls back on `VirtualQuery`/SEH. All three are cheap on Windows and
emulated by Wine (a signal handler or a wineserver round-trip, under Wine's
virtual-memory lock). Proposed measurements, cheapest first:

1. **Switches that already exist, same build and same scene.**
   `VICTORY_COPY_MODE=plain` (copies without the mapped-page guard) and
   `VICTORY_COPY_MODE=none` (no copies; the image is wrong, the timing of
   everything else is right). `none` minus normal is the total copy cost.
   `plain` minus normal is the guard and fault cost, which is where Wine's
   fault delivery shows.
2. **A profile of the process.** `perf record -g --call-graph dwarf` on the
   `victory.exe` process and `wineserver` for 60 s in the world, then group
   samples by module: Wine's `ntdll.so`/`win32u.so`/wineserver and the
   kernel's fault and signal paths count as Wine overhead, `victory.exe` as
   ours, `libvulkan_radeon`/vkd3d-proton as the GPU driver path. This gives
   the Wine share directly.
3. **Micro-benchmarks on both OSes.** Time the three primitives the copies
   use in a loop: `GetWriteWatch` on 256 MB, a guard-page fault round-trip,
   and a 64 KiB `MapViewOfFile3`. Do it on Windows and under this Proton on
   the same machine. #844 measured `GetWriteWatch` at ~180 µs per call in the
   world under Wine; #834 measured 54,051 one-page views at 5–18 s under Wine.
4. **The same save and scene on Windows**, if a Windows install is available
   on this machine: the per-category times side by side give the Wine factor
   per category without any modelling.

## What should go native first

**Not the renderer.** The GPU takes 3 % of the frame, and vkd3d-proton is
not on the critical path. A Vulkan backend would gain nothing measurable
today, so it stays out of scope until the copies are fixed.

The candidate is **dirty-page tracking of title memory**, if steps 1–2 show
that the guard, fault and write-watch share of the 2.7 s is large. Under Wine
every one of those operations is a signal or a wineserver call. Linux can
answer "which pages of this range were written" in one syscall, with no
faults: the `PAGEMAP_SCAN` ioctl with `PM_SCAN_WP_MATCHING` (Linux 6.7+),
or userfaultfd's asynchronous write-protect mode. That can be done without
a port: a small Wine unixlib (a PE DLL with a Unix side, loaded only under
Wine) behind the same interface as `watch.rs`'s Windows calls. The rest of
VIctory stays the Windows build.

Before any native code, two VIctory-side reductions apply on Windows too and
are cheaper:

- **Check less.** 97 % of the 7.4 GB checked per frame is unchanged.
  Coarser ranges for memory the title rarely writes would cut the checks.
  So would trusting the journal for longer.
- **Look at the fallbacks.** The 687 copies per frame meeting unmapped
  pages each take an exception-and-retry path. Under Wine, an exception
  costs several µs and more.

Re-measure after each step with the statistics lines above. Make the next
change only where the numbers move.
