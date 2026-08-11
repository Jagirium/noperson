//! FFI bindings for NVIDIA Performance Primitives (NPP).
//!
//! NPP provides GPU-accelerated planar affine warp and bilinear resize.
//!
//! All functions operate on raw CUDA device pointers — zero-copy with cudarc and ort.
//! NPP is downloaded by the runtime bootstrap and resolved once with `dlopen`.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use libloading::Library;

// ── NPP Type Definitions ─────────────────────────────────────────────────

pub type NppStatus = i32;

pub const NPP_SUCCESS: NppStatus = 0;

/// Size in pixels.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NppiSize {
    pub width: i32,
    pub height: i32,
}

/// Rectangle ROI.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NppiRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

/// 2D point.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NppiPoint {
    pub x: i32,
    pub y: i32,
}

/// CUDA 12.8 application-managed NPP stream context.
///
/// This is copied from `nppGetStreamContext`; applications must not fabricate
/// or alter the reserved field.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct NppStreamContext {
    pub h_stream: *mut std::ffi::c_void,
    pub cuda_device_id: i32,
    pub multi_processor_count: i32,
    pub max_threads_per_multi_processor: i32,
    pub max_threads_per_block: i32,
    pub shared_mem_per_block: usize,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub stream_flags: u32,
    pub reserved0: i32,
}

// NPP interpolation modes
pub const NPPI_INTER_NN: i32 = 1;
pub const NPPI_INTER_LINEAR: i32 = 2;
pub const NPPI_INTER_CUBIC: i32 = 4;

type NppSetStream = unsafe extern "C" fn(*mut std::ffi::c_void) -> NppStatus;
type NppGetStreamContext = unsafe extern "C" fn(*mut NppStreamContext) -> NppStatus;
type WarpAffineP3Ctx = unsafe extern "C" fn(
    *const *const f32,
    NppiSize,
    i32,
    NppiRect,
    *const *mut f32,
    i32,
    NppiRect,
    *const [f64; 3],
    i32,
    NppStreamContext,
) -> NppStatus;
type ResizeP3Ctx = unsafe extern "C" fn(
    *const *const f32,
    i32,
    NppiSize,
    NppiRect,
    *const *mut f32,
    i32,
    NppiSize,
    NppiRect,
    i32,
    NppStreamContext,
) -> NppStatus;
struct NppApi {
    _libraries: Vec<Library>,
    set_stream: NppSetStream,
    get_stream_context: NppGetStreamContext,
    warp_affine_p3_ctx: WarpAffineP3Ctx,
    resize_p3_ctx: ResizeP3Ctx,
}

static NPP: OnceLock<NppApi> = OnceLock::new();
static NPP_STREAM_CAPTURE: Mutex<()> = Mutex::new(());

#[derive(Debug, thiserror::Error)]
pub enum NppLoadError {
    #[error("NPP library is missing: {0}")]
    MissingLibrary(PathBuf),
    #[error("failed to load NPP library {path}: {source}")]
    Library {
        path: PathBuf,
        #[source]
        source: libloading::Error,
    },
    #[error("NPP symbol is missing: {0}")]
    MissingSymbol(&'static str),
    #[error("NPP was already initialized from another runtime generation")]
    AlreadyInitialized,
}

pub fn initialize_runtime(root: &Path) -> Result<(), NppLoadError> {
    let api = unsafe { NppApi::load(root)? };
    NPP.set(api).map_err(|_| NppLoadError::AlreadyInitialized)
}

impl NppApi {
    unsafe fn load(root: &Path) -> Result<Self, NppLoadError> {
        #[cfg(target_os = "windows")]
        const LIBRARIES: &[&str] = &[
            "nppc64_12.dll",
            "nppig64_12.dll",
            "nppif64_12.dll",
            "nppim64_12.dll",
        ];
        #[cfg(not(target_os = "windows"))]
        const LIBRARIES: &[&str] = &[
            "libnppc.so.12",
            "libnppig.so.12",
            "libnppif.so.12",
            "libnppim.so.12",
        ];

        let mut libraries = Vec::with_capacity(LIBRARIES.len());
        for name in LIBRARIES {
            let path = root.join(name);
            if !path.exists() {
                return Err(NppLoadError::MissingLibrary(path));
            }
            libraries.push(unsafe { Library::new(&path) }.map_err(|source| {
                NppLoadError::Library {
                    path: path.clone(),
                    source,
                }
            })?);
        }

        unsafe fn resolve<T: Copy>(
            libraries: &[Library],
            name: &'static [u8],
            printable: &'static str,
        ) -> Result<T, NppLoadError> {
            for library in libraries {
                if let Ok(symbol) = unsafe { library.get::<T>(name) } {
                    return Ok(*symbol);
                }
            }
            Err(NppLoadError::MissingSymbol(printable))
        }

        Ok(Self {
            set_stream: unsafe { resolve(&libraries, b"nppSetStream\0", "nppSetStream")? },
            get_stream_context: unsafe {
                resolve(&libraries, b"nppGetStreamContext\0", "nppGetStreamContext")?
            },
            warp_affine_p3_ctx: unsafe {
                resolve(
                    &libraries,
                    b"nppiWarpAffine_32f_P3R_Ctx\0",
                    "nppiWarpAffine_32f_P3R_Ctx",
                )?
            },
            resize_p3_ctx: unsafe {
                resolve(
                    &libraries,
                    b"nppiResize_32f_P3R_Ctx\0",
                    "nppiResize_32f_P3R_Ctx",
                )?
            },
            _libraries: libraries,
        })
    }
}

fn api() -> Result<&'static NppApi, NppError> {
    NPP.get().ok_or(NppError::NotInitialized)
}

// ── Safe Rust Wrappers ──────────────────────────────────────────────────

/// Check NPP status and convert to Result.
#[inline]
pub fn check_npp(status: NppStatus, op: &str) -> Result<(), NppError> {
    if status == NPP_SUCCESS {
        Ok(())
    } else {
        Err(NppError::Status {
            status,
            op: op.to_string(),
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NppError {
    #[error("NPP error {status} in {op}")]
    Status { status: NppStatus, op: String },
    #[error("NPP runtime is not initialized")]
    NotInitialized,
    #[error("NPP CUDA stream pointer must be non-null")]
    NullStream,
}

/// Capture an application-managed NPP context for one CUDA stream.
///
/// # Safety
/// The stream must be a valid CUDA stream pointer and remain alive.
pub(crate) unsafe fn capture_stream_context(
    stream_ptr: *mut std::ffi::c_void,
) -> Result<NppStreamContext, NppError> {
    if stream_ptr.is_null() {
        return Err(NppError::NullStream);
    }
    let _capture_guard = NPP_STREAM_CAPTURE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let api = api()?;
    check_npp(unsafe { (api.set_stream)(stream_ptr) }, "nppSetStream")?;
    let mut context = MaybeUninit::<NppStreamContext>::uninit();
    check_npp(
        unsafe { (api.get_stream_context)(context.as_mut_ptr()) },
        "nppGetStreamContext",
    )?;
    Ok(unsafe { context.assume_init() })
}

/// Affine warp a 3-plane CHW f32 image (one contiguous buffer, plane stride = H*W).
///
/// `coeffs` is a FORWARD affine matrix (src → dst) — same convention as cv2.warpAffine.
/// NPP reads via texture cache which makes this ~5× faster than a naive scalar kernel.
///
/// # Safety
/// `src_base` and `dst_base` must be valid CUDA device pointers to buffers of size
/// `3 * src_h * src_w * sizeof(f32)` and `3 * dst_h * dst_w * sizeof(f32)` respectively.
pub(crate) unsafe fn warp_affine_f32_p3(
    src_base: *const f32,
    src_w: i32,
    src_h: i32,
    dst_base: *mut f32,
    dst_w: i32,
    dst_h: i32,
    coeffs: &[[f64; 3]; 2],
    stream_context: &NppStreamContext,
) -> Result<(), NppError> {
    let src_plane = (src_h as usize) * (src_w as usize);
    let dst_plane = (dst_h as usize) * (dst_w as usize);

    let src_ptrs: [*const f32; 3] = unsafe {
        [
            src_base,
            src_base.add(src_plane),
            src_base.add(2 * src_plane),
        ]
    };
    let dst_ptrs: [*mut f32; 3] = unsafe {
        [
            dst_base,
            dst_base.add(dst_plane),
            dst_base.add(2 * dst_plane),
        ]
    };

    let src_size = NppiSize {
        width: src_w,
        height: src_h,
    };
    let src_roi = NppiRect {
        x: 0,
        y: 0,
        width: src_w,
        height: src_h,
    };
    let dst_roi = NppiRect {
        x: 0,
        y: 0,
        width: dst_w,
        height: dst_h,
    };
    let src_step = src_w * 4; // single plane, 1 channel * sizeof(f32)
    let dst_step = dst_w * 4;

    check_npp(
        unsafe {
            (api()?.warp_affine_p3_ctx)(
                src_ptrs.as_ptr(),
                src_size,
                src_step,
                src_roi,
                dst_ptrs.as_ptr(),
                dst_step,
                dst_roi,
                coeffs.as_ptr(),
                NPPI_INTER_LINEAR,
                *stream_context,
            )
        },
        "nppiWarpAffine_32f_P3R_Ctx",
    )
}

/// Bilinear resize for a 3-plane CHW f32 image.
///
/// # Safety
/// Both pointers must be valid device pointers to CHW buffers.
pub(crate) unsafe fn resize_f32_p3(
    src_base: *const f32,
    src_w: i32,
    src_h: i32,
    dst_base: *mut f32,
    dst_w: i32,
    dst_h: i32,
    stream_context: &NppStreamContext,
) -> Result<(), NppError> {
    let src_plane = (src_h as usize) * (src_w as usize);
    let dst_plane = (dst_h as usize) * (dst_w as usize);

    let src_ptrs: [*const f32; 3] = unsafe {
        [
            src_base,
            src_base.add(src_plane),
            src_base.add(2 * src_plane),
        ]
    };
    let dst_ptrs: [*mut f32; 3] = unsafe {
        [
            dst_base,
            dst_base.add(dst_plane),
            dst_base.add(2 * dst_plane),
        ]
    };

    let src_size = NppiSize {
        width: src_w,
        height: src_h,
    };
    let src_roi = NppiRect {
        x: 0,
        y: 0,
        width: src_w,
        height: src_h,
    };
    let dst_size = NppiSize {
        width: dst_w,
        height: dst_h,
    };
    let dst_roi = NppiRect {
        x: 0,
        y: 0,
        width: dst_w,
        height: dst_h,
    };
    let src_step = src_w * 4;
    let dst_step = dst_w * 4;

    check_npp(
        unsafe {
            (api()?.resize_p3_ctx)(
                src_ptrs.as_ptr(),
                src_step,
                src_size,
                src_roi,
                dst_ptrs.as_ptr(),
                dst_step,
                dst_size,
                dst_roi,
                NPPI_INTER_LINEAR,
                *stream_context,
            )
        },
        "nppiResize_32f_P3R_Ctx",
    )
}

#[cfg(test)]
mod tests {
    use core::mem::{align_of, offset_of, size_of};

    use super::{
        NPP_SUCCESS, NppGetStreamContext, NppStatus, NppStreamContext, NppiRect, NppiSize,
        ResizeP3Ctx, WarpAffineP3Ctx,
    };

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn cuda_12_8_stream_context_matches_x64_abi() {
        assert_eq!(size_of::<NppStreamContext>(), 48);
        assert_eq!(align_of::<NppStreamContext>(), 8);
        assert_eq!(offset_of!(NppStreamContext, h_stream), 0);
        assert_eq!(offset_of!(NppStreamContext, cuda_device_id), 8);
        assert_eq!(offset_of!(NppStreamContext, multi_processor_count), 12);
        assert_eq!(
            offset_of!(NppStreamContext, max_threads_per_multi_processor),
            16
        );
        assert_eq!(offset_of!(NppStreamContext, max_threads_per_block), 20);
        assert_eq!(offset_of!(NppStreamContext, shared_mem_per_block), 24);
        assert_eq!(offset_of!(NppStreamContext, compute_capability_major), 32);
        assert_eq!(offset_of!(NppStreamContext, compute_capability_minor), 36);
        assert_eq!(offset_of!(NppStreamContext, stream_flags), 40);
        assert_eq!(offset_of!(NppStreamContext, reserved0), 44);
        assert_eq!(size_of::<NppiSize>(), 8);
        assert_eq!(align_of::<NppiSize>(), 4);
        assert_eq!(size_of::<NppiRect>(), 16);
        assert_eq!(align_of::<NppiRect>(), 4);
    }

    #[test]
    fn cuda_12_8_ctx_function_aliases_compile_with_context_by_value() {
        unsafe extern "C" fn get_context(_: *mut NppStreamContext) -> NppStatus {
            NPP_SUCCESS
        }
        unsafe extern "C" fn warp(
            _: *const *const f32,
            _: NppiSize,
            _: i32,
            _: NppiRect,
            _: *const *mut f32,
            _: i32,
            _: NppiRect,
            _: *const [f64; 3],
            _: i32,
            _: NppStreamContext,
        ) -> NppStatus {
            NPP_SUCCESS
        }
        unsafe extern "C" fn resize(
            _: *const *const f32,
            _: i32,
            _: NppiSize,
            _: NppiRect,
            _: *const *mut f32,
            _: i32,
            _: NppiSize,
            _: NppiRect,
            _: i32,
            _: NppStreamContext,
        ) -> NppStatus {
            NPP_SUCCESS
        }

        let _: NppGetStreamContext = get_context;
        let _: WarpAffineP3Ctx = warp;
        let _: ResizeP3Ctx = resize;
    }
}
