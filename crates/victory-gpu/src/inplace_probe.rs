//! A synthetic probe, no title involved: can the host GPU read and write
//! title memory in place (`OpenExistingHeapFromAddress` and friends), and
//! how fast, on this host? Written for vkd3d-proton under Wine, where an
//! import that Vulkan refuses still answers `S_OK` (vkd3d then allocates
//! its own memory behind the heap), so every row is judged by a round trip
//! through the title's own pointer, never by the HRESULT.
//!
//! All tests are ignored (they take up to gigabytes and seconds). Run one
//! at a time, e.g.
//! `cargo xwin test -p victory-gpu --lib --target x86_64-pc-windows-msvc --
//! --ignored --nocapture --test-threads 1 inplace_import`, with
//! `PROBE_CASES=va,ww,...` choosing rows (see `inplace_import`) and
//! `PROBE_MB` the size of `inplace_bench` and `inplace_cpu`.
//!
//! The shaders use `RWStructuredBuffer<uint>` root UAVs: Wine's
//! `D3DCompile` (vkd3d-shader) has no `RWByteAddressBuffer` methods, and a
//! root UAV is a bare address either way.

use std::cell::Cell;
use std::ffi::c_void;
use std::time::{Duration, Instant};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::{Interface, s};
use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_ALL_ACCESS, GetWriteWatch, MEM_COMMIT, MEM_PRESERVE_PLACEHOLDER,
    MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MapViewOfFile,
    MapViewOfFile3, PAGE_NOACCESS, PAGE_READWRITE, ResetWriteWatch, SEC_COMMIT, SEC_RESERVE,
    UnmapViewOfFile, VirtualAlloc, VirtualAlloc2, VirtualFree,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use crate::host::host;

const MB: usize = 1 << 20;
const MEM_WRITE_WATCH: u32 = 0x0020_0000;
/// The flags of a buffer placed in an opened heap.
const IMPORTED: D3D12_RESOURCE_FLAGS = D3D12_RESOURCE_FLAGS(
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 | D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER.0,
);
/// Dwords one thread of a bandwidth pass reads or writes.
const PER_THREAD: u32 = 16;
/// The largest piece of a bandwidth pass: one dispatch, one submission.
const MAX_CHUNK: usize = 16 * MB;
/// The odd multiplier of the scattered order (logical dword -> dword).
const MUL: u32 = 0x9E37_79B1;

// mode 0: mem[i] = mem[i] * 3 + add + i, one dword a thread (round trip).
// Otherwise 16 dwords a thread; bit 1: scattered order; bit 2: write
// (mem[i] = logical + add), else read (sink[thread] = sum of mem[i] ^ logical).
const SOURCE: &str = r#"
cbuffer P : register(b0) { uint mode; uint base; uint mask; uint mul; uint add; uint p0; uint p1; uint p2; };
RWStructuredBuffer<uint> mem : register(u0);
RWStructuredBuffer<uint> sink : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint j = id.x;
    if (mode == 0) {
        uint i = base + j;
        mem[i] = mem[i] * 3 + add + i;
        return;
    }
    uint acc = 0;
    for (uint n = 0; n < 16; n++) {
        uint l = base + j * 16 + n;
        uint i = l;
        if ((mode & 2) != 0) i = (l * mul) & mask;
        if ((mode & 4) != 0) mem[i] = l + add;
        else acc += mem[i] ^ l;
    }
    if ((mode & 4) == 0) sink[j] = acc;
}
"#;

const SEQ_READ: u32 = 1;
const SCAT_READ: u32 = 1 | 2;
const SEQ_WRITE: u32 = 1 | 4;
const SCAT_WRITE: u32 = 1 | 2 | 4;

struct Gpu {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    value: Cell<u64>,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    queries: ID3D12QueryHeap,
    /// Readback: two timestamps, then a copy of `sink`.
    back: ID3D12Resource,
    /// Video memory the read passes write their sums to.
    sink: ID3D12Resource,
    frequency: u64,
}

const SINK_BYTES: u64 = (MAX_CHUNK / PER_THREAD as usize) as u64;

fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

fn committed(
    device: &ID3D12Device,
    kind: D3D12_HEAP_TYPE,
    size: u64,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
) -> windows::core::Result<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: kind,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain buffer creation.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc(size, flags),
            state,
            None,
            &mut buffer,
        )?;
    }
    buffer.ok_or_else(windows::core::Error::empty)
}

impl Gpu {
    fn new() -> Option<Self> {
        let device = host()?.device.clone();
        // SAFETY: plain object creation on a live device; the shader source
        // and the blobs live through the calls that read them.
        unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .ok()?;
            let allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .ok()?;
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .ok()?;
            list.Close().ok()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;

            let mut code: Option<ID3DBlob> = None;
            let mut errors: Option<ID3DBlob> = None;
            let compiled = D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("probe"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            );
            if compiled.is_err() {
                if let Some(e) = errors {
                    let text = std::slice::from_raw_parts(
                        e.GetBufferPointer().cast::<u8>(),
                        e.GetBufferSize(),
                    );
                    println!("PROBE| shader: {}", String::from_utf8_lossy(text));
                }
                return None;
            }
            let code = code?;
            let uav = |register| D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: register,
                        RegisterSpace: 0,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let params = [
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Num32BitValues: 8,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                uav(0),
                uav(1),
            ];
            let desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
            };
            let mut blob: Option<ID3DBlob> = None;
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)
                .ok()?;
            let blob = blob?;
            let root: ID3D12RootSignature = device
                .CreateRootSignature(
                    0,
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer().cast::<u8>(),
                        blob.GetBufferSize(),
                    ),
                )
                .ok()?;
            let pso: ID3D12PipelineState = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.GetBufferPointer(),
                        BytecodeLength: code.GetBufferSize(),
                    },
                    ..Default::default()
                })
                .ok()?;
            let mut queries: Option<ID3D12QueryHeap> = None;
            device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: 2,
                        NodeMask: 0,
                    },
                    &mut queries,
                )
                .ok()?;
            let back = committed(
                &device,
                D3D12_HEAP_TYPE_READBACK,
                64 * 1024 + SINK_BYTES,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )
            .ok()?;
            let sink = committed(
                &device,
                D3D12_HEAP_TYPE_DEFAULT,
                SINK_BYTES,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            )
            .ok()?;
            let frequency = queue.GetTimestampFrequency().ok()?;
            Some(Self {
                device,
                queue,
                allocator,
                list,
                fence,
                value: Cell::new(0),
                root,
                pso,
                queries: queries?,
                back,
                sink,
                frequency,
            })
        }
    }

    /// Records and runs one submission and waits for it: seconds on the
    /// GPU between its first and last command (timestamp queries), and
    /// seconds from submitting to done.
    fn run(&self, record: impl FnOnce(&ID3D12GraphicsCommandList)) -> (f64, f64) {
        // SAFETY: the list is ours and idle (every run waits); the objects
        // it names live in `self` or in the caller across the wait.
        unsafe {
            self.allocator.Reset().unwrap();
            self.list.Reset(&self.allocator, None).unwrap();
            self.list.SetComputeRootSignature(&self.root);
            self.list.SetPipelineState(&self.pso);
            self.list
                .EndQuery(&self.queries, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            record(&self.list);
            self.list
                .EndQuery(&self.queries, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            self.list.ResolveQueryData(
                &self.queries,
                D3D12_QUERY_TYPE_TIMESTAMP,
                0,
                2,
                &self.back,
                0,
            );
            self.list.Close().unwrap();
            let started = Instant::now();
            self.queue
                .ExecuteCommandLists(&[Some(self.list.cast().unwrap())]);
            let value = self.value.get() + 1;
            self.value.set(value);
            self.queue.Signal(&self.fence, value).unwrap();
            while self.fence.GetCompletedValue() < value {
                if let Err(e) = self.device.GetDeviceRemovedReason() {
                    panic!("device removed while waiting: {e}");
                }
                assert!(
                    started.elapsed() < Duration::from_secs(30),
                    "a submission did not finish in 30 s"
                );
                if started.elapsed() > Duration::from_millis(2) {
                    std::thread::sleep(Duration::from_micros(200));
                } else {
                    std::hint::spin_loop();
                }
            }
            let wall = started.elapsed().as_secs_f64();
            let mut data = std::ptr::null_mut();
            self.back.Map(0, None, Some(&mut data)).unwrap();
            let stamps = std::slice::from_raw_parts(data.cast::<u64>(), 2).to_vec();
            self.back.Unmap(0, None);
            (
                stamps[1].wrapping_sub(stamps[0]) as f64 / self.frequency as f64,
                wall,
            )
        }
    }

    /// One dispatch of `threads` threads over the buffer at `mem`.
    fn dispatch(
        &self,
        mem: u64,
        mode: u32,
        base: u32,
        mask: u32,
        add: u32,
        threads: u32,
    ) -> (f64, f64) {
        assert!(threads.is_multiple_of(64));
        let constants = [mode, base, mask, MUL, add, 0, 0, 0];
        // SAFETY: the sink lives in `self`.
        let sink = unsafe { self.sink.GetGPUVirtualAddress() };
        self.run(|l| {
            // SAFETY: an open list; the constants are copied by the call.
            unsafe {
                l.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                l.SetComputeRootUnorderedAccessView(1, mem);
                l.SetComputeRootUnorderedAccessView(2, sink);
                l.Dispatch(threads / 64, 1, 1);
            }
        })
    }

    /// The first `threads` sums the last read pass left in the sink.
    fn sums(&self, threads: u32) -> Vec<u32> {
        self.run(|l| {
            // SAFETY: both buffers live in `self`; the range is inside both.
            unsafe { l.CopyBufferRegion(&self.back, 64 * 1024, &self.sink, 0, threads as u64 * 4) };
        });
        // SAFETY: the copy ran (run waits); the range is inside the buffer.
        unsafe {
            let mut data = std::ptr::null_mut();
            self.back.Map(0, None, Some(&mut data)).unwrap();
            let sums = std::slice::from_raw_parts(
                data.cast::<u8>().add(64 * 1024).cast::<u32>(),
                threads as usize,
            )
            .to_vec();
            self.back.Unmap(0, None);
            sums
        }
    }

    /// A bandwidth pass over `words` dwords (a power of two) at `mem`, in
    /// pieces sized from the rate so far so that each stays near 0.15 s.
    /// Stops early past `budget` seconds or after a piece over a second.
    fn pass(&self, mem: u64, words: u32, mode: u32, add: u32, budget: f64) -> Pass {
        assert!(words.is_power_of_two());
        let mut pass = Pass::default();
        let mut chunk = MB as u32 / 4;
        let mut at = 0u32;
        while at < words {
            let n = chunk.min(words - at);
            let (gpu, wall) = self.dispatch(mem, mode, at, words - 1, add, n / PER_THREAD);
            pass.gpu += gpu;
            pass.wall += wall;
            pass.bytes += n as u64 * 4;
            pass.last = (at, n / PER_THREAD);
            pass.slowest = pass.slowest.max(wall);
            at += n;
            if wall > 1.0 || pass.wall > budget {
                pass.cut = at < words;
                break;
            }
            let bytes = (n as f64 * 4.0 / wall.max(1e-6) * 0.15) as usize;
            let bytes = bytes.clamp(MB, MAX_CHUNK);
            chunk = (1usize << bytes.ilog2()) as u32 / 4;
        }
        pass
    }
}

#[derive(Default, Clone, Copy)]
struct Pass {
    bytes: u64,
    gpu: f64,
    wall: f64,
    slowest: f64,
    cut: bool,
    /// The last piece: its first logical dword and thread count.
    last: (u32, u32),
}

impl Pass {
    fn rate(&self) -> String {
        format!(
            "{:.2} ({:.2}){}",
            self.bytes as f64 / self.gpu.max(1e-9) / 1e9,
            self.bytes as f64 / self.wall.max(1e-9) / 1e9,
            if self.cut {
                format!(" cut at {} MB", self.bytes >> 20)
            } else {
                String::new()
            }
        )
    }
}

fn hr<T>(r: &windows::core::Result<T>) -> String {
    match r {
        Ok(_) => "S_OK".into(),
        Err(e) => format!("{:#010x}", e.code().0 as u32),
    }
}

/// The inverse of an odd number modulo 2^32.
fn inverse(a: u32) -> u32 {
    let mut x = a;
    for _ in 0..5 {
        x = x.wrapping_mul(2u32.wrapping_sub(a.wrapping_mul(x)));
    }
    x
}

/// CPU writes a fresh pattern through `write`, the GPU rewrites the dwords
/// in place, the CPU reads through `read` (another alias of the same memory
/// or the same pointer). `at` is the dword in the buffer at `mem`.
fn round_trip(
    gpu: &Gpu,
    mem: u64,
    write: *mut u32,
    read: *const u32,
    at: u32,
    count: u32,
    salt: u32,
) -> String {
    let pattern = |i: u32| i.wrapping_mul(0x85EB_CA6B) ^ salt;
    // SAFETY: the caller's pointers cover `count` dwords at dword `at`.
    unsafe {
        for i in 0..count {
            write.add((at + i) as usize).write_volatile(pattern(at + i));
        }
    }
    let add = salt.rotate_left(7) | 1;
    gpu.dispatch(mem, 0, at, 0, add, count);
    let (mut good, mut untouched, mut first) = (0u32, 0u32, None);
    for i in at..at + count {
        // SAFETY: as above.
        let got = unsafe { read.add(i as usize).read_volatile() };
        let want = pattern(i).wrapping_mul(3).wrapping_add(add).wrapping_add(i);
        if got == want {
            good += 1;
        } else {
            if got == pattern(i) {
                untouched += 1;
            }
            first.get_or_insert((i, got, want));
        }
    }
    if good == count {
        format!("in place, 0 wrong of {count}")
    } else if untouched == count {
        "NOT in place: the CPU sees none of the GPU's writes".into()
    } else {
        let (i, got, want) = first.unwrap();
        format!(
            "WRONG: {} of {count} ({untouched} untouched), dword {i} is {got:#x}, should be {want:#x}",
            count - good
        )
    }
}

/// Places a buffer over all of `heap` and round-trips its first and last
/// 4 MB (or all of a smaller one), twice each: the result for the table.
/// `write` and `read` are CPU views of the heap's memory.
fn judge(gpu: &Gpu, heap: &ID3D12Heap, write: *mut u8, read: *const u8, size: usize) -> String {
    // SAFETY: a live heap.
    let desc = unsafe { heap.GetDesc() };
    let mut text = format!(
        "heap {} KB flags {:#x}",
        desc.SizeInBytes >> 10,
        desc.Flags.0
    );
    if desc.SizeInBytes < size as u64 {
        return text + " smaller than asked";
    }
    let mut placed = None;
    let mut how = String::new();
    // vkd3d-proton asks for the cross-adapter flag on these heaps.
    for flags in [IMPORTED, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS] {
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: a live heap; the buffer covers it from offset 0.
        let made = unsafe {
            gpu.device.CreatePlacedResource(
                heap,
                0,
                &buffer_desc(size as u64, flags),
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        };
        how = format!("placed {} (flags {:#x})", hr(&made), flags.0);
        if made.is_ok() {
            placed = buffer;
            break;
        }
    }
    text += &format!("; {how}");
    let Some(buffer) = placed else { return text };
    // SAFETY: a live buffer.
    let mem = unsafe { buffer.GetGPUVirtualAddress() };
    let count = ((4 * MB).min(size) / 4 / 64 * 64) as u32;
    let total = (size / 4) as u32;
    let mut windows = vec![0u32];
    if total > count {
        windows.push((total - count) / 64 * 64);
    }
    static SALT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x1234_5678);
    for at in windows {
        for _ in 0..2 {
            let salt = SALT.fetch_add(0x0101_0101, std::sync::atomic::Ordering::Relaxed);
            let result = round_trip(gpu, mem, write.cast(), read.cast(), at, count, salt);
            if !result.starts_with("in place") {
                return format!("{text}; at dword {at}: {result}");
            }
        }
    }
    // SAFETY: a plain query.
    if let Err(e) = unsafe { gpu.device.GetDeviceRemovedReason() } {
        text += &format!("; DEVICE REMOVED {e}");
    }
    format!(
        "{text}; in place both ways, first and last {} KB, twice",
        (count as usize * 4) >> 10
    )
}

fn row(case: &str, open: &str, ms: f64, result: &str) {
    println!("PROBE| {case} | {open} | {ms:.1} ms | {result}");
}

/// Opens with `open`, judges, prints the row. Returns the heap (alive).
fn open_and_judge(
    gpu: &Gpu,
    case: &str,
    open: impl FnOnce() -> windows::core::Result<ID3D12Heap>,
    write: *mut u8,
    read: *const u8,
    size: usize,
) -> Option<ID3D12Heap> {
    let started = Instant::now();
    let heap = open();
    let ms = started.elapsed().as_secs_f64() * 1e3;
    let result = match &heap {
        Ok(h) => judge(gpu, h, write, read, size),
        Err(e) => e.message(),
    };
    row(case, &hr(&heap), ms, &result);
    heap.ok()
}

fn alloc(size: usize, kind: u32) -> *mut u8 {
    // SAFETY: a fresh allocation.
    let p = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            size,
            MEM_RESERVE | MEM_COMMIT | kind,
            PAGE_READWRITE,
        )
    };
    assert!(
        !p.is_null(),
        "VirtualAlloc of {size}: {}",
        std::io::Error::last_os_error()
    );
    p.cast()
}

fn free(p: *mut u8) {
    // SAFETY: an allocation of `alloc`, its heaps released by the caller.
    unsafe { VirtualFree(p.cast(), 0, MEM_RELEASE) };
}

fn section(size: usize, kind: u32) -> *mut c_void {
    // SAFETY: plain section creation.
    let h = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            PAGE_READWRITE | kind,
            (size as u64 >> 32) as u32,
            size as u32,
            std::ptr::null(),
        )
    };
    assert!(
        !h.is_null(),
        "CreateFileMappingW: {}",
        std::io::Error::last_os_error()
    );
    h
}

fn view(section: *mut c_void, size: usize) -> *mut u8 {
    // SAFETY: a live section.
    let v = unsafe { MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, size) };
    assert!(
        !v.Value.is_null(),
        "MapViewOfFile: {}",
        std::io::Error::last_os_error()
    );
    v.Value.cast()
}

fn unview(p: *mut u8) {
    // SAFETY: a view of `view` or `map3`.
    unsafe {
        UnmapViewOfFile(
            windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: p.cast() },
        )
    };
}

fn placeholder(size: usize) -> *mut u8 {
    // SAFETY: a fresh placeholder.
    let p = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            std::ptr::null(),
            size,
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS,
            std::ptr::null_mut(),
            0,
        )
    };
    assert!(
        !p.is_null(),
        "placeholder: {}",
        std::io::Error::last_os_error()
    );
    p.cast()
}

/// As xmem's `map`: splits the placeholder around the range, maps there.
fn map3(section: *mut c_void, base: *mut u8, offset: usize, len: usize) -> *mut u8 {
    // SAFETY: the range lies in one placeholder of ours.
    let v = unsafe {
        VirtualFree(base.cast(), len, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
        MapViewOfFile3(
            section,
            GetCurrentProcess(),
            base.cast(),
            offset as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    assert_eq!(
        v.Value.cast::<u8>(),
        base,
        "MapViewOfFile3: {}",
        std::io::Error::last_os_error()
    );
    base
}

/// Pages of `[p, p + size)` the write watch has as written.
fn watched(p: *mut u8, size: usize) -> String {
    let mut pages = vec![std::ptr::null_mut::<c_void>(); size / 4096];
    let mut count = pages.len();
    let mut granularity = 0u32;
    // SAFETY: the out array holds `count` entries.
    let r = unsafe {
        GetWriteWatch(
            0,
            p.cast(),
            size,
            pages.as_mut_ptr(),
            &mut count,
            &mut granularity,
        )
    };
    if r != 0 {
        return format!("GetWriteWatch failed ({})", std::io::Error::last_os_error());
    }
    format!("{count} of {} pages written", size / 4096)
}

fn from_address(gpu: &Gpu, p: *const u8) -> windows::core::Result<ID3D12Heap> {
    let device: ID3D12Device3 = gpu.device.cast()?;
    // SAFETY: the probe's question; the memory outlives the heap.
    unsafe { device.OpenExistingHeapFromAddress(p.cast()) }
}

fn from_range(gpu: &Gpu, p: *const u8, size: usize) -> windows::core::Result<ID3D12Heap> {
    let device: ID3D12Device13 = gpu.device.cast()?;
    // SAFETY: as above.
    unsafe { device.OpenExistingHeapFromAddress1(p.cast(), size) }
}

fn from_section(gpu: &Gpu, section: *mut c_void) -> windows::core::Result<ID3D12Heap> {
    let device: ID3D12Device3 = gpu.device.cast()?;
    // SAFETY: a live section.
    unsafe { device.OpenExistingHeapFromFileMapping(HANDLE(section)) }
}

fn cases() -> Vec<String> {
    std::env::var("PROBE_CASES")
        .unwrap_or_else(|_| "va,partial,sub,view,placeholder,split,reserve".into())
        .split(',')
        .map(str::to_owned)
        .collect()
}

/// Table 1 and 2: which memory opens as a heap, and whether a buffer placed
/// over it is the same memory. Rows by `PROBE_CASES` (comma separated):
/// `va` `ww` `partial` `sub` `view` `placeholder` `split` `reserve` are the
/// default but for `ww`; `wwreset`, `filemap`, `filemap-reserve` and
/// `big:<MB>` are asked for by name, one process each.
#[test]
#[ignore = "probe"]
fn inplace_import() {
    let Some(gpu) = Gpu::new() else {
        println!("PROBE| no host GPU, skipped");
        return;
    };
    let mut existing = D3D12_FEATURE_DATA_EXISTING_HEAPS::default();
    // SAFETY: the structure matches the feature.
    let asked = unsafe {
        gpu.device.CheckFeatureSupport(
            D3D12_FEATURE_EXISTING_HEAPS,
            (&raw mut existing).cast(),
            size_of::<D3D12_FEATURE_DATA_EXISTING_HEAPS>() as u32,
        )
    };
    println!(
        "PROBE| D3D12_FEATURE_EXISTING_HEAPS: {} Supported={}; ID3D12Device13: {}",
        hr(&asked),
        existing.Supported.0,
        gpu.device.cast::<ID3D12Device13>().is_ok()
    );
    println!("PROBE| case | HRESULT | open time | result");
    for case in cases() {
        match case.as_str() {
            "va" | "ww" => {
                let kind = if case == "ww" { MEM_WRITE_WATCH } else { 0 };
                for mb in [64, 256, 1024] {
                    let p = alloc(mb * MB, kind);
                    let name = format!(
                        "VirtualAlloc {mb} MB{}",
                        if kind != 0 { " MEM_WRITE_WATCH" } else { "" }
                    );
                    let heap = open_and_judge(&gpu, &name, || from_address(&gpu, p), p, p, mb * MB);
                    if kind != 0 {
                        println!(
                            "PROBE|   write watch after the round trips: {}",
                            watched(p, mb * MB)
                        );
                    }
                    drop(heap);
                    free(p);
                }
            }
            "wwtouched" => {
                // Written by the CPU before the import: pages writable under Wine.
                let size = 64 * MB;
                let p = alloc(size, MEM_WRITE_WATCH);
                // SAFETY: inside the allocation.
                unsafe { std::ptr::write_bytes(p, 0x5a, size) };
                open_and_judge(
                    &gpu,
                    "MEM_WRITE_WATCH 64 MB, CPU-written first",
                    || from_address(&gpu, p),
                    p,
                    p,
                    size,
                );
                free(p);
            }
            "wwreset" => write_watch_states(&gpu),
            "partial" => {
                let size = 64 * MB;
                // SAFETY: a fresh reservation, half of it committed.
                let p: *mut u8 = unsafe {
                    let p = VirtualAlloc(std::ptr::null(), size, MEM_RESERVE, PAGE_READWRITE);
                    VirtualAlloc(p, size / 2, MEM_COMMIT, PAGE_READWRITE);
                    p.cast()
                };
                open_and_judge(
                    &gpu,
                    "64 MB reserved, half committed",
                    || from_address(&gpu, p),
                    p,
                    p,
                    size / 2,
                );
                free(p);
                let p = alloc(size, 0);
                // SAFETY: inside the allocation.
                let inner = unsafe { p.add(MB) };
                open_and_judge(
                    &gpu,
                    "FromAddress at +1 MB of an allocation",
                    || from_address(&gpu, inner),
                    inner,
                    inner,
                    4 * MB,
                );
                free(p);
            }
            "sub" => {
                let p = alloc(64 * MB, 0);
                for (name, offset, size) in [
                    ("FromAddress1 4 MB at +1 MB", MB, 4 * MB),
                    ("FromAddress1 4 MB at +4 KB", 4096, 4 * MB),
                    ("FromAddress1 4 MB + 4 KB at +8 MB", 8 * MB, 4 * MB + 4096),
                    (
                        "FromAddress1 4 MB at +16 MB + 256 bytes",
                        16 * MB + 256,
                        4 * MB,
                    ),
                ] {
                    // SAFETY: inside the allocation.
                    let q = unsafe { p.add(offset) };
                    open_and_judge(&gpu, name, || from_range(&gpu, q, size), q, q, size);
                }
                let w = alloc(64 * MB, MEM_WRITE_WATCH);
                // SAFETY: inside the allocation.
                let q = unsafe { w.add(MB) };
                open_and_judge(
                    &gpu,
                    "FromAddress1 4 MB at +1 MB, MEM_WRITE_WATCH",
                    || from_range(&gpu, q, 4 * MB),
                    q,
                    q,
                    4 * MB,
                );
                free(w);
                free(p);
            }
            "view" | "placeholder" | "reserve" => {
                let size = 64 * MB;
                let s = section(
                    size,
                    if case == "reserve" {
                        SEC_RESERVE
                    } else {
                        SEC_COMMIT
                    },
                );
                let a = if case == "placeholder" {
                    map3(s, placeholder(size), 0, size)
                } else {
                    view(s, size)
                };
                if case == "reserve" {
                    // SAFETY: commits the view's pages.
                    let r = unsafe { VirtualAlloc(a.cast(), size, MEM_COMMIT, PAGE_READWRITE) };
                    assert!(!r.is_null(), "commit: {}", std::io::Error::last_os_error());
                }
                let b = view(s, size);
                let name = match case.as_str() {
                    "view" => {
                        "SEC_COMMIT section 64 MB, MapViewOfFile view, FromAddress; CPU writes via 2nd view"
                    }
                    "placeholder" => {
                        "SEC_COMMIT section 64 MB, MapViewOfFile3 into a placeholder, FromAddress; CPU writes via 2nd view"
                    }
                    _ => {
                        "SEC_RESERVE section 64 MB, view committed, FromAddress; CPU writes via 2nd view"
                    }
                };
                let heap = open_and_judge(&gpu, name, || from_address(&gpu, a), b, a, size);
                drop(heap);
                unview(b);
                unview(a);
                // SAFETY: our section.
                unsafe { CloseHandle(s) };
            }
            "split" => {
                // As xmem: one placeholder, the section's slices mapped
                // into it in two runs, out of order.
                let size = 64 * MB;
                let s = section(size, SEC_COMMIT);
                let base = placeholder(size);
                let low = map3(s, base, 48 * MB, 16 * MB);
                // SAFETY: inside the placeholder.
                let high = map3(s, unsafe { base.add(16 * MB) }, 0, 48 * MB);
                let b = view(s, size);
                let heap = open_and_judge(
                    &gpu,
                    "SEC_COMMIT section, 48 MB run inside a split placeholder, FromAddress; CPU writes via another view",
                    || from_address(&gpu, high),
                    b,
                    high,
                    48 * MB,
                );
                drop(heap);
                // SAFETY: inside both views.
                let (q, qb) = unsafe { (high.add(MB), b.add(MB)) };
                let heap = open_and_judge(
                    &gpu,
                    "FromAddress1 4 MB at +1 MB of that run; CPU writes via another view",
                    || from_range(&gpu, q, 4 * MB),
                    qb,
                    q,
                    4 * MB,
                );
                drop(heap);
                unview(b);
                unview(high);
                unview(low);
                // SAFETY: our section.
                unsafe { CloseHandle(s) };
            }
            "filemap" | "filemap-reserve" => {
                let size = 16 * MB;
                let s = section(
                    size,
                    if case == "filemap" {
                        SEC_COMMIT
                    } else {
                        SEC_RESERVE
                    },
                );
                let a = view(s, size);
                if case != "filemap" {
                    // SAFETY: commits the view's pages.
                    unsafe { VirtualAlloc(a.cast(), size, MEM_COMMIT, PAGE_READWRITE) };
                }
                let b = view(s, size);
                let name = format!(
                    "{} section 16 MB, OpenExistingHeapFromFileMapping; CPU via two views",
                    if case == "filemap" {
                        "SEC_COMMIT"
                    } else {
                        "SEC_RESERVE"
                    }
                );
                let heap = open_and_judge(&gpu, &name, || from_section(&gpu, s), b, a, size);
                drop(heap);
                unview(b);
                unview(a);
                // SAFETY: our section.
                unsafe { CloseHandle(s) };
            }
            big if big.starts_with("big:") => {
                let mb: usize = big[4..].parse().unwrap();
                let p = alloc(mb * MB, 0);
                // A buffer over the first 4095 MB at most (dword indices),
                // then one placed over the heap's last 64 MB.
                let size = (mb * MB).min(4095 * MB);
                let heap = open_and_judge(
                    &gpu,
                    &format!("VirtualAlloc {mb} MB"),
                    || from_address(&gpu, p),
                    p,
                    p,
                    size,
                );
                if let Some(heap) = heap {
                    let tail = 64 * MB;
                    let offset = mb * MB - tail;
                    let mut buffer: Option<ID3D12Resource> = None;
                    // SAFETY: a live heap; the buffer ends at its end.
                    let made = unsafe {
                        gpu.device.CreatePlacedResource(
                            &heap,
                            offset as u64,
                            &buffer_desc(tail as u64, IMPORTED),
                            D3D12_RESOURCE_STATE_COMMON,
                            None,
                            &mut buffer,
                        )
                    };
                    let result = match &buffer {
                        // SAFETY: a live buffer; the pointers cover its 64 MB.
                        Some(b) => unsafe {
                            let q = p.add(offset).cast::<u32>();
                            let at = ((tail - 4 * MB) / 4) as u32;
                            round_trip(
                                &gpu,
                                b.GetGPUVirtualAddress(),
                                q,
                                q,
                                at,
                                MB as u32,
                                0x7777_0001,
                            )
                        },
                        None => String::new(),
                    };
                    row(
                        &format!(
                            "  a buffer placed at its last 64 MB (offset {} MB), last 4 MB",
                            offset / MB
                        ),
                        &hr(&made),
                        0.0,
                        &result,
                    );
                }
                free(p);
            }
            other => println!("PROBE| unknown case {other}"),
        }
        // SAFETY: a plain query.
        if let Err(e) = unsafe { gpu.device.GetDeviceRemovedReason() } {
            println!("PROBE| DEVICE REMOVED after {case}: {e}");
            break;
        }
    }
}

/// Write-watched memory in the states its pages can be in (never written,
/// written, reset), what the watch reports after GPU work, and what a
/// reset costs the next submission.
fn write_watch_states(gpu: &Gpu) {
    let size = 64 * MB;
    let part = 4 * MB;
    let p = alloc(size, MEM_WRITE_WATCH);
    let heap = from_address(gpu, p);
    println!(
        "PROBE| MEM_WRITE_WATCH 64 MB, untouched: open {}",
        hr(&heap)
    );
    let Ok(heap) = heap else { return };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: a live heap.
    let made = unsafe {
        gpu.device.CreatePlacedResource(
            &heap,
            0,
            &buffer_desc(size as u64, IMPORTED),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )
    };
    println!("PROBE|   placed {}", hr(&made));
    let Some(buffer) = buffer else { return };
    // SAFETY: a live buffer.
    let mem = unsafe { buffer.GetGPUVirtualAddress() };
    let words = (part / 4) as u32;
    let gpu_wrote = |add: u32| {
        gpu.dispatch(mem, SEQ_WRITE, 0, u32::MAX, add, words / PER_THREAD);
        // SAFETY: inside the allocation.
        let wrong = (0..words).filter(|&i| unsafe { p.cast::<u32>().add(i as usize).read_volatile() } != i.wrapping_add(add)).count();
        format!(
            "CPU reads {wrong} wrong of {words}; watch: {}",
            watched(p, part)
        )
    };
    println!(
        "PROBE|   GPU writes 4 MB never written by the CPU: {}",
        gpu_wrote(0x1111)
    );
    // SAFETY: the watched allocation.
    unsafe { ResetWriteWatch(p.cast(), size) };
    println!(
        "PROBE|   after ResetWriteWatch, GPU writes again: {}",
        gpu_wrote(0x2222)
    );
    // SAFETY: as above.
    unsafe { ResetWriteWatch(p.cast(), size) };
    println!(
        "PROBE|   after ResetWriteWatch, round trip: {}; watch: {}",
        round_trip(gpu, mem, p.cast(), p.cast(), 0, words, 0xabcd_0123),
        watched(p, part)
    );
    // SAFETY: as above.
    unsafe { ResetWriteWatch(p.cast(), size) };
    println!(
        "PROBE|   reset again, round trip of the last 4 MB (first CPU write there): {}",
        round_trip(
            gpu,
            mem,
            p.cast(),
            p.cast(),
            ((size - part) / 4) as u32,
            words,
            0x0bad_cafe
        ),
    );
    // SAFETY: a plain query.
    if let Err(e) = unsafe { gpu.device.GetDeviceRemovedReason() } {
        println!("PROBE|   DEVICE REMOVED {e}");
    }
    // Whose writes does the watch see? Of the whole allocation: with the
    // GPU idle, after it read 4 MB, after it wrote 4 MB, after a CPU write.
    // SAFETY: the watched allocation.
    unsafe { ResetWriteWatch(p.cast(), size) };
    println!(
        "PROBE|   reset, GPU idle: whole allocation {}",
        watched(p, size)
    );
    gpu.sums(64);
    println!(
        "PROBE|   then a submission that does not name the buffer: whole allocation {}",
        watched(p, size)
    );
    gpu.dispatch(mem, SEQ_READ, 0, u32::MAX, 0, words / PER_THREAD);
    println!(
        "PROBE|   then the GPU read the first 4 MB: whole allocation {}",
        watched(p, size)
    );
    // SAFETY: as above.
    unsafe { ResetWriteWatch(p.cast(), size) };
    gpu.dispatch(mem, SEQ_WRITE, 0, u32::MAX, 0x3333, words / PER_THREAD);
    println!(
        "PROBE|   reset, then the GPU wrote the first 4 MB: whole allocation {}",
        watched(p, size)
    );
    // SAFETY: as above; inside the allocation.
    unsafe {
        ResetWriteWatch(p.cast(), size);
        p.add(32 * MB).write_volatile(1);
    }
    println!(
        "PROBE|   reset, then the CPU wrote one byte: whole allocation {}",
        watched(p, size)
    );
    // What a reset costs the next submission (the driver takes the pages again).
    let ms = |reset: bool| {
        if reset {
            // SAFETY: as above.
            unsafe { ResetWriteWatch(p.cast(), size) };
        }
        gpu.dispatch(mem, SEQ_READ, 0, u32::MAX, 0, 64).1 * 1e3
    };
    let (plain, after) = (
        [ms(false), ms(false), ms(false)],
        [ms(true), ms(true), ms(true)],
    );
    println!(
        "PROBE|   a 4 KB dispatch, submit to done: {plain:.2?} ms; right after ResetWriteWatch of the 64 MB: {after:.2?} ms"
    );
    drop(buffer);
    drop(heap);
    free(p);
}

fn probe_mb() -> usize {
    let mb: usize = std::env::var("PROBE_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    assert!(mb.is_power_of_two() && mb <= 1024);
    mb
}

fn gpu_upload_supported(device: &ID3D12Device) -> bool {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    // SAFETY: the structure matches the feature.
    unsafe {
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS16,
                (&raw mut options).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS16>() as u32,
            )
            .is_ok()
            && options.GPUUploadHeapSupported.as_bool()
    }
}

/// A committed buffer shaders write in `kind` of heap, mapped if it can be.
fn gpu_upload(
    device: &ID3D12Device,
    size: usize,
) -> windows::core::Result<(ID3D12Resource, *mut u8)> {
    let buffer = committed(
        device,
        D3D12_HEAP_TYPE_GPU_UPLOAD,
        size as u64,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    )?;
    let mut data = std::ptr::null_mut();
    // SAFETY: a CPU-visible buffer.
    unsafe { buffer.Map(0, None, Some(&mut data))? };
    Ok((buffer, data.cast()))
}

/// Table 3: GB/s of a compute shader over `PROBE_MB` of each kind of
/// memory, by GPU timestamps (and by wall clock), first pass / second pass.
#[test]
#[ignore = "probe"]
fn inplace_bench() {
    let Some(gpu) = Gpu::new() else {
        println!("PROBE| no host GPU, skipped");
        return;
    };
    let size = probe_mb() * MB;
    let words = (size / 4) as u32;
    println!(
        "PROBE| {} MB; GB/s by GPU timestamps (by wall clock), first pass / second pass; sums = read passes checked on the CPU",
        size / MB
    );
    println!(
        "PROBE| memory | seq write | seq read | scattered read | scattered write | sums | CPU view after"
    );
    let only =
        std::env::var("PROBE_TARGETS").unwrap_or_else(|_| "import,shared,vram,upload".into());
    for target in only.split(',') {
        let mut keep: Vec<Box<dyn std::any::Any>> = Vec::new();
        let mut cpu: *mut u8 = std::ptr::null_mut();
        let mut imported: *mut u8 = std::ptr::null_mut();
        let (name, buffer) = match target {
            "import" => {
                let p = alloc(size, 0);
                imported = p;
                cpu = p;
                let heap = match from_address(&gpu, p) {
                    Ok(h) => h,
                    Err(e) => {
                        println!("PROBE| imported | open failed {e}");
                        continue;
                    }
                };
                let mut buffer: Option<ID3D12Resource> = None;
                // SAFETY: a live heap.
                unsafe {
                    gpu.device
                        .CreatePlacedResource(
                            &heap,
                            0,
                            &buffer_desc(size as u64, IMPORTED),
                            D3D12_RESOURCE_STATE_COMMON,
                            None,
                            &mut buffer,
                        )
                        .unwrap();
                }
                keep.push(Box::new(heap));
                (
                    "imported VirtualAlloc (OpenExistingHeapFromAddress)",
                    buffer.unwrap(),
                )
            }
            "shared" => {
                let (b, p) = host().unwrap().shared_buffer(size as u64).unwrap();
                cpu = p;
                ("CUSTOM WRITE_BACK L0 (shared_buffer)", b)
            }
            "vram" => (
                "DEFAULT (video memory)",
                committed(
                    &gpu.device,
                    D3D12_HEAP_TYPE_DEFAULT,
                    size as u64,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                )
                .unwrap(),
            ),
            "upload" => {
                if !gpu_upload_supported(&gpu.device) {
                    println!("PROBE| GPU_UPLOAD | GPUUploadHeapSupported = false");
                    continue;
                }
                match gpu_upload(&gpu.device, size) {
                    Ok((b, p)) => {
                        cpu = p;
                        ("GPU_UPLOAD", b)
                    }
                    Err(e) => {
                        println!("PROBE| GPU_UPLOAD | {e}");
                        continue;
                    }
                }
            }
            other => {
                println!("PROBE| unknown target {other}");
                continue;
            }
        };
        // SAFETY: a live buffer.
        let mem = unsafe { buffer.GetGPUVirtualAddress() };
        let (a, b) = (0x0a00_0001u32, 0x0b00_0003u32);
        let inv = inverse(MUL);
        let index = |mode: u32, l: u32| {
            if mode & 2 != 0 {
                l.wrapping_mul(MUL) & (words - 1)
            } else {
                l
            }
        };
        // What a read pass leaves in the sink for its last piece, given the
        // memory's contents as a function of the dword index.
        let check = |pass: &Pass, mode: u32, content: &dyn Fn(u32) -> u32| -> usize {
            let (base, threads) = pass.last;
            let sums = gpu.sums(threads);
            (0..threads)
                .filter(|&j| {
                    let want = (0..PER_THREAD).fold(0u32, |acc, n| {
                        let l = base + j * PER_THREAD + n;
                        acc.wrapping_add(content(index(mode, l)) ^ l)
                    });
                    sums[j as usize] != want
                })
                .count()
        };
        let twice = |mode: u32, add: u32| {
            let first = gpu.pass(mem, words, mode, add, 20.0);
            let second = if first.cut {
                first
            } else {
                gpu.pass(mem, words, mode, add, 20.0)
            };
            (first, second)
        };
        let show = |(f, s): &(Pass, Pass)| {
            format!(
                "{} / {} [max piece {:.0} ms]",
                f.rate(),
                s.rate(),
                s.slowest * 1e3
            )
        };
        let seq_write = twice(SEQ_WRITE, a);
        let after_a = |i: u32| i.wrapping_add(a);
        let seq_read = twice(SEQ_READ, 0);
        let mut wrong = if seq_write.1.cut {
            0
        } else {
            check(&seq_read.1, SEQ_READ, &after_a)
        };
        let scat_read = twice(SCAT_READ, 0);
        if !seq_write.1.cut {
            wrong += check(&scat_read.1, SCAT_READ, &after_a);
        }
        let scat_write = twice(SCAT_WRITE, b);
        let after_b = |i: u32| (i.wrapping_mul(inv) & (words - 1)).wrapping_add(b);
        let complete = !seq_write.1.cut && !scat_write.1.cut;
        if complete {
            let verify = gpu.pass(mem, words, SEQ_READ, 0, 20.0);
            wrong += check(&verify, SEQ_READ, &after_b);
        }
        let seen = if cpu.is_null() || !complete {
            "-".to_owned()
        } else {
            // SAFETY: the mapped or imported range, `size` bytes.
            let bad = (0..words)
                .step_by(61)
                .filter(
                    |&i| unsafe { cpu.cast::<u32>().add(i as usize).read_volatile() } != after_b(i),
                )
                .count();
            format!("{bad} wrong of {} sampled", words.div_ceil(61))
        };
        println!(
            "PROBE| {name} | {} | {} | {} | {} | {wrong} wrong | {seen}",
            show(&seq_write),
            show(&seq_read),
            show(&scat_read),
            show(&scat_write)
        );
        // SAFETY: a plain query.
        if let Err(e) = unsafe { gpu.device.GetDeviceRemovedReason() } {
            println!("PROBE| DEVICE REMOVED: {e}");
            return;
        }
        drop(buffer);
        drop(keep);
        if !imported.is_null() {
            free(imported);
        }
    }
}

/// MB/s of the CPU copying 64 MB pieces into and out of `[p, p + size)`,
/// best of three.
fn cpu_rates(p: *mut u8, size: usize, scratch: &mut [u8]) -> String {
    let piece = scratch.len();
    let mut best = [f64::MAX; 2];
    for _ in 0..3 {
        let started = Instant::now();
        for at in (0..size).step_by(piece) {
            // SAFETY: inside both ranges.
            unsafe { std::ptr::copy_nonoverlapping(scratch.as_ptr(), p.add(at), piece) };
        }
        best[0] = best[0].min(started.elapsed().as_secs_f64());
        let started = Instant::now();
        for at in (0..size).step_by(piece) {
            // SAFETY: as above.
            unsafe { std::ptr::copy_nonoverlapping(p.add(at), scratch.as_mut_ptr(), piece) };
            std::hint::black_box(&mut *scratch);
        }
        best[1] = best[1].min(started.elapsed().as_secs_f64());
    }
    format!(
        "write {:.0} MB/s | read {:.0} MB/s",
        size as f64 / 1e6 / best[0],
        size as f64 / 1e6 / best[1]
    )
}

/// Table 4: what the CPU pays for touching each kind of memory.
#[test]
#[ignore = "probe"]
fn inplace_cpu() {
    let Some(gpu) = Gpu::new() else {
        println!("PROBE| no host GPU, skipped");
        return;
    };
    let size = probe_mb().max(64) * MB;
    let words = (size / 4) as u32;
    let mut scratch: Vec<u8> = (0..64 * MB).map(|i| (i * 31) as u8).collect();
    println!(
        "PROBE| CPU memcpy over {} MB in 64 MB pieces, best of 3",
        size / MB
    );
    println!("PROBE| memory | write | read");

    let plain = alloc(size, 0);
    println!(
        "PROBE| VirtualAlloc, not imported | {}",
        cpu_rates(plain, size, &mut scratch)
    );
    free(plain);

    let p = alloc(size, 0);
    println!(
        "PROBE| VirtualAlloc before its import | {}",
        cpu_rates(p, size, &mut scratch)
    );
    let started = Instant::now();
    let heap = from_address(&gpu, p);
    println!(
        "PROBE|   open {} in {:.1} ms",
        hr(&heap),
        started.elapsed().as_secs_f64() * 1e3
    );
    if let Ok(heap) = heap {
        let result = judge(&gpu, &heap, p, p, size);
        println!("PROBE|   {result}");
        println!(
            "PROBE| the same, imported, after GPU round trips | {}",
            cpu_rates(p, size, &mut scratch)
        );
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: a live heap.
        unsafe {
            gpu.device
                .CreatePlacedResource(
                    &heap,
                    0,
                    &buffer_desc(size as u64, IMPORTED),
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .unwrap();
        }
        let buffer = buffer.unwrap();
        // SAFETY: a live buffer.
        gpu.pass(
            unsafe { buffer.GetGPUVirtualAddress() },
            words,
            SEQ_WRITE,
            1,
            20.0,
        );
        println!(
            "PROBE| the same, after the GPU wrote all of it | {}",
            cpu_rates(p, size, &mut scratch)
        );
        drop(buffer);
        drop(heap);
        println!(
            "PROBE| the same, heap released | {}",
            cpu_rates(p, size, &mut scratch)
        );
    }
    free(p);

    if let Some((buffer, data)) = host().unwrap().shared_buffer(size as u64) {
        println!(
            "PROBE| CUSTOM WRITE_BACK L0 (shared_buffer), mapped | {}",
            cpu_rates(data, size, &mut scratch)
        );
        drop(buffer);
    }
    if gpu_upload_supported(&gpu.device) {
        match gpu_upload(&gpu.device, size) {
            Ok((buffer, data)) => {
                println!(
                    "PROBE| GPU_UPLOAD, mapped | {}",
                    cpu_rates(data, size, &mut scratch)
                );
                drop(buffer);
            }
            Err(e) => println!("PROBE| GPU_UPLOAD | {e}"),
        }
    } else {
        println!("PROBE| GPU_UPLOAD | GPUUploadHeapSupported = false");
    }
}
