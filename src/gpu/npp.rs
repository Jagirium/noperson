//! FFI bindings for NVIDIA Performance Primitives (NPP).
//!
//! NPP provides GPU-accelerated image processing functions.
//! We use it for operations that would be complex to write as custom CUDA kernels:
//! affine warp, bilinear resize, gaussian blur, morphological dilate.
//!
//! All functions operate on raw CUDA device pointers — zero-copy with cudarc and ort.
//! NPP is downloaded by the runtime bootstrap and resolved once with `dlopen`.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

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

// NPP interpolation modes
pub const NPPI_INTER_NN: i32 = 1;
pub const NPPI_INTER_LINEAR: i32 = 2;
pub const NPPI_INTER_CUBIC: i32 = 4;

// NPP Gaussian mask sizes
pub const NPP_MASK_SIZE_3_X_3: i32 = 0;
pub const NPP_MASK_SIZE_5_X_5: i32 = 1;
pub const NPP_MASK_SIZE_7_X_7: i32 = 2;
pub const NPP_MASK_SIZE_9_X_9: i32 = 3;
pub const NPP_MASK_SIZE_11_X_11: i32 = 4;
pub const NPP_MASK_SIZE_13_X_13: i32 = 5;

type NppSetStream = unsafe extern "C" fn(*mut std::ffi::c_void) -> NppStatus;
type ResizeC3 = unsafe extern "C" fn(
    *const f32,
    i32,
    NppiSize,
    NppiRect,
    *mut f32,
    i32,
    NppiSize,
    NppiRect,
    i32,
) -> NppStatus;
type WarpAffineC3 = unsafe extern "C" fn(
    *const f32,
    NppiSize,
    i32,
    NppiRect,
    *mut f32,
    i32,
    NppiRect,
    *const [[f64; 3]; 2],
    i32,
) -> NppStatus;
type WarpAffineP3 = unsafe extern "C" fn(
    *const *const f32,
    NppiSize,
    i32,
    NppiRect,
    *const *mut f32,
    i32,
    NppiRect,
    *const [[f64; 3]; 2],
    i32,
) -> NppStatus;
type ResizeP3 = unsafe extern "C" fn(
    *const *const f32,
    i32,
    NppiSize,
    NppiRect,
    *const *mut f32,
    i32,
    NppiSize,
    NppiRect,
    i32,
) -> NppStatus;
type FilterGaussC1 =
    unsafe extern "C" fn(*const f32, i32, *mut f32, i32, NppiSize, i32) -> NppStatus;
type MorphologyC1 = unsafe extern "C" fn(*const f32, i32, *mut f32, i32, NppiSize) -> NppStatus;

struct NppApi {
    _libraries: Vec<Library>,
    set_stream: NppSetStream,
    resize_c3: ResizeC3,
    warp_affine_c3: WarpAffineC3,
    warp_affine_p3: WarpAffineP3,
    resize_p3: ResizeP3,
    filter_gauss_c1: FilterGaussC1,
    dilate_c1: MorphologyC1,
    erode_c1: MorphologyC1,
}

static NPP: OnceLock<NppApi> = OnceLock::new();

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
            resize_c3: unsafe {
                resolve(&libraries, b"nppiResize_32f_C3R\0", "nppiResize_32f_C3R")?
            },
            warp_affine_c3: unsafe {
                resolve(
                    &libraries,
                    b"nppiWarpAffine_32f_C3R\0",
                    "nppiWarpAffine_32f_C3R",
                )?
            },
            warp_affine_p3: unsafe {
                resolve(
                    &libraries,
                    b"nppiWarpAffine_32f_P3R\0",
                    "nppiWarpAffine_32f_P3R",
                )?
            },
            resize_p3: unsafe {
                resolve(&libraries, b"nppiResize_32f_P3R\0", "nppiResize_32f_P3R")?
            },
            filter_gauss_c1: unsafe {
                resolve(
                    &libraries,
                    b"nppiFilterGauss_32f_C1R\0",
                    "nppiFilterGauss_32f_C1R",
                )?
            },
            dilate_c1: unsafe {
                resolve(
                    &libraries,
                    b"nppiDilate3x3_32f_C1R\0",
                    "nppiDilate3x3_32f_C1R",
                )?
            },
            erode_c1: unsafe {
                resolve(
                    &libraries,
                    b"nppiErode3x3_32f_C1R\0",
                    "nppiErode3x3_32f_C1R",
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
}

/// Set NPP to use our cudarc CUDA stream.
///
/// # Safety
/// The stream must be a valid CUDA stream pointer and remain alive.
pub unsafe fn set_npp_stream(stream_ptr: *mut std::ffi::c_void) -> Result<(), NppError> {
    check_npp(unsafe { (api()?.set_stream)(stream_ptr) }, "nppSetStream")
}

/// Resize a 3-channel f32 image using bilinear interpolation.
///
/// # Safety
/// All pointers must be valid CUDA device pointers with correct dimensions.
pub unsafe fn resize_f32_c3(
    src: *const f32,
    src_w: i32,
    src_h: i32,
    dst: *mut f32,
    dst_w: i32,
    dst_h: i32,
) -> Result<(), NppError> {
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
    let src_step = src_w * 3 * 4; // 3 channels * sizeof(f32)
    let dst_step = dst_w * 3 * 4;

    check_npp(
        unsafe {
            (api()?.resize_c3)(
                src,
                src_step,
                src_size,
                src_roi,
                dst,
                dst_step,
                dst_size,
                dst_roi,
                NPPI_INTER_LINEAR,
            )
        },
        "nppiResize_32f_C3R",
    )
}

/// Affine warp a 3-channel f32 image.
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
pub unsafe fn warp_affine_f32_c3(
    src: *const f32,
    src_w: i32,
    src_h: i32,
    dst: *mut f32,
    dst_w: i32,
    dst_h: i32,
    coeffs: &[[f64; 3]; 2],
) -> Result<(), NppError> {
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
    let src_step = src_w * 3 * 4;
    let dst_step = dst_w * 3 * 4;

    check_npp(
        unsafe {
            (api()?.warp_affine_c3)(
                src,
                src_size,
                src_step,
                src_roi,
                dst,
                dst_step,
                dst_roi,
                coeffs as *const _,
                NPPI_INTER_LINEAR,
            )
        },
        "nppiWarpAffine_32f_C3R",
    )
}

/// Affine warp a 3-plane CHW f32 image (one contiguous buffer, plane stride = H*W).
///
/// `coeffs` is a FORWARD affine matrix (src → dst) — same convention as cv2.warpAffine.
/// NPP reads via texture cache which makes this ~5× faster than a naive scalar kernel.
///
/// # Safety
/// `src_base` and `dst_base` must be valid CUDA device pointers to buffers of size
/// `3 * src_h * src_w * sizeof(f32)` and `3 * dst_h * dst_w * sizeof(f32)` respectively.
pub unsafe fn warp_affine_f32_p3(
    src_base: *const f32,
    src_w: i32,
    src_h: i32,
    dst_base: *mut f32,
    dst_w: i32,
    dst_h: i32,
    coeffs: &[[f64; 3]; 2],
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
            (api()?.warp_affine_p3)(
                src_ptrs.as_ptr(),
                src_size,
                src_step,
                src_roi,
                dst_ptrs.as_ptr(),
                dst_step,
                dst_roi,
                coeffs as *const _,
                NPPI_INTER_LINEAR,
            )
        },
        "nppiWarpAffine_32f_P3R",
    )
}

/// Bilinear resize for a 3-plane CHW f32 image.
///
/// # Safety
/// Both pointers must be valid device pointers to CHW buffers.
pub unsafe fn resize_f32_p3(
    src_base: *const f32,
    src_w: i32,
    src_h: i32,
    dst_base: *mut f32,
    dst_w: i32,
    dst_h: i32,
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
            (api()?.resize_p3)(
                src_ptrs.as_ptr(),
                src_step,
                src_size,
                src_roi,
                dst_ptrs.as_ptr(),
                dst_step,
                dst_size,
                dst_roi,
                NPPI_INTER_LINEAR,
            )
        },
        "nppiResize_32f_P3R",
    )
}

/// Gaussian blur a 1-channel f32 image (for mask smoothing).
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
pub unsafe fn gaussian_blur_f32_c1(
    src: *const f32,
    dst: *mut f32,
    width: i32,
    height: i32,
    mask_size: i32,
) -> Result<(), NppError> {
    let roi = NppiSize { width, height };
    let step = width * 4;

    check_npp(
        unsafe { (api()?.filter_gauss_c1)(src, step, dst, step, roi, mask_size) },
        "nppiFilterGauss_32f_C1R",
    )
}

/// 3x3 morphological dilate on 1-channel f32 (replaces max_pool2d for masks).
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
pub unsafe fn dilate_3x3_f32_c1(
    src: *const f32,
    dst: *mut f32,
    width: i32,
    height: i32,
) -> Result<(), NppError> {
    let roi = NppiSize { width, height };
    let step = width * 4;

    check_npp(
        unsafe { (api()?.dilate_c1)(src, step, dst, step, roi) },
        "nppiDilate3x3_32f_C1R",
    )
}

/// 3x3 morphological erode on 1-channel f32.
///
/// # Safety
/// All pointers must be valid CUDA device pointers.
pub unsafe fn erode_3x3_f32_c1(
    src: *const f32,
    dst: *mut f32,
    width: i32,
    height: i32,
) -> Result<(), NppError> {
    let roi = NppiSize { width, height };
    let step = width * 4;

    check_npp(
        unsafe { (api()?.erode_c1)(src, step, dst, step, roi) },
        "nppiErode3x3_32f_C1R",
    )
}
