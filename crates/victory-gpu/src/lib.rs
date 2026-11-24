//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees. Planned:
//! `present` (swap chain).

// Off Windows there is no host device, so the code that feeds it (records,
// formats, barriers) builds but has no caller. Keep Windows strict and only
// quiet what the missing host leaves unused.
#![cfg_attr(
    not(windows),
    allow(dead_code, unused_imports, unused_variables, unused_mut)
)]

#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
pub mod console;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod draw;
pub mod dxgi;
pub mod fence;
pub mod frame;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
#[cfg(all(test, windows))]
mod inplace_probe;
pub mod list;
pub mod object;
#[cfg(windows)]
mod owned;
pub mod pipeline;
#[cfg(windows)]
mod primitive;
pub mod query;
pub mod queue;
pub mod resource;
pub mod root;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod shadow;
pub mod signature;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
mod watch;

#[cfg(windows)]
pub use host::on_failure;

#[cfg(windows)]
pub use watch::{
    TitleMemory, title_memory_allocated, title_memory_freed, title_memory_record,
    title_view_mapped, title_view_unmapped,
};

/// Before the kernel writes title memory `[at, at + len)` for the title (a
/// file read into it): the kernel does not fault into our handlers, so
/// blocks the GPU owns go back to the title and guarded ones are opened.
#[cfg(windows)]
pub fn kernel_will_write(at: usize, len: usize) {
    owned::release(at, len);
    watch::writing(at, len);
}

/// Before the kernel reads title memory `[at, at + len)` for the title, or
/// the title changes how it is committed or protected: blocks the GPU owns
/// go back to the title.
#[cfg(windows)]
pub fn title_will_touch(at: usize, len: usize) {
    owned::release(at, len);
}

/// The runtime mapped or unmapped title memory `[at, at + len)`, or the
/// title decommitted it: see [`watch::title_memory_remapped`]; blocks the
/// GPU owned there are dropped.
#[cfg(windows)]
pub fn title_memory_remapped(at: usize, len: usize) {
    owned::forget(at, len);
    watch::title_memory_remapped(at, len);
}

/// The title decommits `[at, at + len)`: blocks the GPU owned there are
/// dropped (what the title frees needs no bytes back).
#[cfg(windows)]
pub fn title_memory_decommitted(at: usize, len: usize) {
    owned::forget(at, len);
}

/// The title releases the allocation at `base` whole.
#[cfg(windows)]
pub fn title_memory_released(base: usize) {
    owned::forget_allocation(base);
}

/// Whether GPU-written title memory stays with the GPU until the title
/// touches it (`VICTORY_LAZY_COPY_BACK=1`): the runtime then tells us of the
/// title's own memory calls.
#[cfg(windows)]
pub fn lazy_copy_back() -> bool {
    owned::enabled()
}
