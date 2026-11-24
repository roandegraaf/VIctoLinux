//! The title's own calls on memory the GPU side may own.
//!
//! With `VICTORY_LAZY_COPY_BACK=1` the GPU side closes blocks of title
//! memory that shaders wrote, and gives them back at the title's first
//! access, which faults (`victory_gpu`, "GPU-owned title memory"). What
//! does not fault has to be told:
//!
//! - committing or protecting such memory would open it without its bytes
//!   (`VirtualAlloc` over committed pages resets their protection), so its
//!   blocks go back to the title first;
//! - freed or decommitted memory needs no bytes, but is no longer owned;
//! - the kernel reading such memory for the title (`WriteFile`, `fwrite`)
//!   fails where the title would fault.
//!
//! Without the switch none of these is bound: the title's imports go to
//! the host as before. Kernel writes into title memory are in `files` and
//! `dstorage` (they open guarded blocks too).
//!
//! shortcut: other host calls that hand title memory to the kernel
//! (sockets, overlapped writes, `VirtualAlloc2`, `VirtualFreeEx`, the
//! `Nt…` calls) are not covered; none is known to be given GPU memory.

#![allow(non_snake_case)]

use std::ffi::c_void;

use windows_sys::Win32::System::Memory::MEM_RELEASE;

use super::files::host;
use crate::{HleModule, hle_export};

pub extern "system" fn VirtualAlloc(base: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void {
    if !base.is_null() {
        victory_gpu::title_will_touch(base as usize, size);
    }
    host!("kernel32.dll", "VirtualAlloc": extern "system" fn(*mut c_void, usize, u32, u32) -> *mut c_void)(
        base, size, kind, protect,
    )
}

pub extern "system" fn VirtualFree(base: *mut c_void, size: usize, kind: u32) -> i32 {
    if kind & MEM_RELEASE != 0 || size == 0 {
        // The allocation whole (a decommit of size 0 is of all of it).
        victory_gpu::title_memory_released(base as usize);
    } else {
        victory_gpu::title_memory_decommitted(base as usize, size);
    }
    host!("kernel32.dll", "VirtualFree": extern "system" fn(*mut c_void, usize, u32) -> i32)(base, size, kind)
}

pub extern "system" fn VirtualProtect(base: *mut c_void, size: usize, protect: u32, old: *mut u32) -> i32 {
    victory_gpu::title_will_touch(base as usize, size);
    host!("kernel32.dll", "VirtualProtect": extern "system" fn(*mut c_void, usize, u32, *mut u32) -> i32)(
        base, size, protect, old,
    )
}

pub extern "system" fn WriteFile(
    file: *mut c_void,
    buffer: *const u8,
    len: u32,
    written: *mut u32,
    overlapped: *mut c_void,
) -> i32 {
    victory_gpu::title_will_touch(buffer as usize, len as usize);
    host!("kernel32.dll", "WriteFile": extern "system" fn(*mut c_void, *const u8, u32, *mut u32, *mut c_void) -> i32)(
        file, buffer, len, written, overlapped,
    )
}

/// The host CRT writes large requests straight from the caller's buffer.
pub extern "C" fn fwrite(buffer: *const u8, size: usize, count: usize, stream: *mut c_void) -> usize {
    victory_gpu::title_will_touch(buffer as usize, size.saturating_mul(count));
    host!("ucrtbase.dll", "fwrite": extern "C" fn(*const u8, usize, usize, *mut c_void) -> usize)(
        buffer, size, count, stream,
    )
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "vmem",
        dlls: &[
            "kernel32.dll",
            "kernelbase.dll",
            "api-ms-win-core-memory-*",
            "api-ms-win-core-file-*",
            "ucrtbase.dll",
            "api-ms-win-crt-stdio-*",
        ],
        exports: vec![
            hle_export!(VirtualAlloc),
            hle_export!(VirtualFree),
            hle_export!(VirtualProtect),
            hle_export!(WriteFile),
            hle_export!(fwrite),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_DECOMMIT, MEM_RESERVE, PAGE_READONLY, PAGE_READWRITE,
    };

    /// Bound, the calls do what the host's do.
    #[test]
    fn memory_calls_pass_through() {
        const SIZE: usize = 2 << 16;
        let base = VirtualAlloc(std::ptr::null_mut(), SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        assert!(!base.is_null());
        // SAFETY: committed memory of ours.
        unsafe { base.cast::<u8>().write(7) };
        let mut old = 0;
        assert_ne!(VirtualProtect(base, SIZE, PAGE_READONLY, &mut old), 0);
        assert_eq!(old, PAGE_READWRITE);
        assert_ne!(VirtualFree(base, 1 << 16, MEM_DECOMMIT), 0);
        assert_eq!(VirtualAlloc(base, 1 << 16, MEM_COMMIT, PAGE_READWRITE), base);
        // SAFETY: committed again, zeroed.
        assert_eq!(unsafe { base.cast::<u8>().read() }, 0);
        assert_ne!(VirtualFree(base, 0, MEM_RELEASE), 0);
    }

    /// Only with the switch are the title's memory calls ours.
    #[test]
    fn bound_only_with_the_switch() {
        let bound = crate::modules::all().iter().any(|m| m.name == "vmem");
        assert_eq!(bound, victory_gpu::lazy_copy_back());
    }
}
