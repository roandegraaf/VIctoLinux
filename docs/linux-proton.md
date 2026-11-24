# Linux (Proton)

VIctory is a Windows program, and it also runs on Linux under Proton's Wine
with vkd3d-proton for Direct3D 12. Under Proton it gets as far as on Windows:
boot, first-boot screens, intro video, start screen, saves, loading, and the
opening of the first mission (credits over the scene; the scene itself is
still dark, as on Windows).

Tested on one machine (2026-11-24): CachyOS, Linux 7.3, AMD RX 9070 XT
(RADV, Mesa 26.3.1), Proton Experimental `experimental-11.0-20261112`.

## Requirements

- Steam's **Proton Experimental** (Steam → Library → Tools). Older Protons
  may not work: GE-Proton 11-11's vkd3d-proton turns some translated
  shaders into invalid SPIR-V, on which RADV crashes (`vkCreateGraphicsPipelines`,
  exit code 3).
- A Vulkan driver with mesh shaders (RADV on RDNA 2 or later has them).
- `rustup` and [`cargo-xwin`](https://github.com/rust-cross/cargo-xwin)
  to build the Windows target on Linux.

## Building

```sh
rustup target add x86_64-pc-windows-msvc
cargo install --locked cargo-xwin
# cargo-xwin downloads the MSVC CRT and Windows SDK; this accepts their licence.
export XWIN_ACCEPT_LICENSE=1
cargo xwin build --release -p victory --target x86_64-pc-windows-msvc
```

The diagnostics build likewise:
`cargo xwin build --profile diag -p victory --features diagnostics --target x86_64-pc-windows-msvc`.

## Running

`scripts/run-proton.sh` runs `victory.exe` under a chosen Proton or Wine
build (`--proton` / `PROTON`, default Proton Experimental), in a prefix of
its own per build (`~/.local/share/victory/prefixes/`). It fetches GE-Proton's
vkd3d-proton and DXVK `dxgi.dll` once into `~/.cache/victory/runtime/`
(`--ge-tag`, default GE-Proton11-14, whose vkd3d-proton 3.2 build `cae8353ed`
carries the dxil-spirv fix), checks the release checksum, copies them into the
prefix and sets `WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=n;d3dcompiler_47=b"`.
`--dlls proton` uses the Proton build's own instead. Linux paths work as
arguments; logs go to `~/.local/state/victory/logs/`.

```sh
scripts/run-proton.sh fetch                          # the DLLs only
scripts/run-proton.sh -- config
scripts/run-proton.sh /path/to/game/<the title's exe>             # binding report
scripts/run-proton.sh --diag --timeout 900 --mem-max 20G \
    /path/to/game/<the title's exe> --execute
```

`GAME_EXE` can stand in for the exe argument, `VICTORY_EXE` / `--exe` picks
another victory.exe, and `VICTORY_KEYS` and the other `VICTORY_*` switches pass
through. `--auto-enter` presses Enter (the pad's A) every 4 s from 15 s to
150 s after input starts, which gets through the first-boot and start screens
unattended; it uses the diagnostics build. `VICTORY_NO_WATCHDOG` is set to 1 (only effective with #454). The
script warns when there is no default audio sink: without one the title stops
about 15 s in (#834). `scripts/proton-run.sh` is the same script under its
older name.

Saves, settings and `crashes.log` live where `victory.toml` (read from
`~/.config/victory/victory.toml` when present) or `VICTORY_DATA_DIR` puts them,
else in the prefix's `%LOCALAPPDATA%\victory\data`.

With an Xbox controller connected, the keyboard is not a pad (as on
Windows), so `VICTORY_KEYS` presses do not reach the title either.

### Without Proton (system Wine)

A system Wine works too (reported with Wine 11.22 on RDNA2, RDNA3 and
RDNA 3.5). What Proton brings has to be added:

- **vkd3d-proton 3.2 or newer**: `d3d12.dll` and `d3d12core.dll` (Proton
  Experimental's `files/lib/wine/vkd3d-proton/x86_64-windows`). 3.1 turns
  some translated shaders into SPIR-V RADV cannot take.
- **DXVK's `dxgi.dll`**, and `WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=n"`.
- No Microsoft `d3dcompiler_47`: Wine's built-in one compiles the argument
  clamp since it no longer needs `RWByteAddressBuffer` methods.

### Diagnostics switches

In the diagnostics build:

- `VICTORY_NO_WATCHDOG=1` (from #454, if merged) keeps the title's stall
  watchdog from ending slow runs.
- `VICTORY_SHOW_TARGET=<w>x<h>-f<format>[-<n>]` presents an intermediate
  render target instead of the final frame, the `n`th of that size and
  format the frame writes (names as in `VICTORY_DUMP_TARGETS` files). The
  world's G-buffer shows its geometry while the final image is dark:
  `VICTORY_SHOW_TARGET=2560x1440-f28-3` on a Series X profile.
- `VICTORY_DEBUG_PS=1` draws depth-tested geometry with a debug pixel
  shader (surfaces coloured by their slope, from depth derivatives) instead
  of the title's: the geometry of every draw whose vertex or mesh stage
  translates, whether its pixel shader does or not. With the two switches
  together you can walk the world in a flat-colour view.

In every build, a draw of depth-tested geometry whose pixel shader does
not translate (or that the host refuses) is drawn with that debug shader
instead of dropped; `VICTORY_PS_FALLBACK=0` turns that off.
`VICTORY_PIPELINE_WAIT_MS` sets how long a frame may wait for pipelines
being made (500 by default; 0 never waits, pipelines then appear a few
frames later).

## Tests

```sh
CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER=$PWD/scripts/run-proton.sh \
    cargo xwin test --workspace --target x86_64-pc-windows-msvc
```

Two tests start Linux programs from inside Wine and fail there for that
reason only: `loads_binds_and_calls_probe` (builds its fixture with `cargo`)
and `policy::tests::repository_passes_policy` (runs `git`). Run
`cargo xtask policy` natively instead.

## Known differences

- **Speed.** Wine emulates write watching page by page, so the copies of
  title memory take longer than on Windows. Under Wine the kernel is asked
  about each title allocation once per submission (`watch::new_submission`),
  views of physical pages go straight to their guards, and allocations are
  looked up from a cache rather than with `VirtualQuery`, which takes the
  same lock as the title's write faults.
- **Software rendering does not work.** On lavapipe, Wine cannot handle the
  write faults of VIctory's guarded title memory when lavapipe's own
  threads take them.
- **vkd3d-proton differs from Windows drivers in a few places VIctory now
  handles:** its HLSL compiler (Wine's `D3DCompile`) lacks
  `RWByteAddressBuffer` methods and does not embed root signatures; it
  refuses an empty pipeline library given a non-null blob pointer; it does
  not check DXIL container hashes; and its descriptor layout makes a texture
  descriptor read as a constant buffer (a stale slot) a read at address 0.
