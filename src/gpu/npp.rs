//! FFI bindings for NVIDIA Performance Primitives (NPP).
//!
//! NPP provides GPU-accelerated image processing functions.
//! We use it for operations that would be complex to write as custom CUDA kernels:
//! affine warp, bilinear resize, gaussian blur, morphological dilate.
//!
//! All functions operate on raw CUDA device pointers — zero-copy with cudarc and ort.
//! NPP is part of CUDA toolkit, linked via build.rs.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

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

// ── NPP Stream Context ───────────────────────────────────────────────────

unsafe extern "C" {
    /// Set the CUDA stream used by NPP.
    pub fn nppSetStream(hStream: *mut std::ffi::c_void) -> NppStatus;

    /// Get the current NPP CUDA stream.
    pub fn nppGetStream() -> *mut std::ffi::c_void;
}

// ── NPP Image Geometry Transforms (libnppig) ────────────────────────────

unsafe extern "C" {
    /// Affine warp for 32-bit float, 3-channel interleaved image (HWC).
    pub fn nppiWarpAffine_32f_C3R(
        pSrc: *const f32,
        oSrcSize: NppiSize,
        nSrcStep: i32,
        oSrcROI: NppiRect,
        pDst: *mut f32,
        nDstStep: i32,
        oDstROI: NppiRect,
        aCoeffs: *const [[f64; 3]; 2],
        eInterpolation: i32,
    ) -> NppStatus;

    /// Affine warp for 32-bit float, 3-plane image (CHW layout).
    /// `aSrc`/`aDst` are arrays of 3 pointers (one per plane).
    pub fn nppiWarpAffine_32f_P3R(
        aSrc: *const *const f32,
        oSrcSize: NppiSize,
        nSrcStep: i32,
        oSrcROI: NppiRect,
        aDst: *const *mut f32,
        nDstStep: i32,
        oDstROI: NppiRect,
        aCoeffs: *const [[f64; 3]; 2],
        eInterpolation: i32,
    ) -> NppStatus;

    /// Bilinear resize for 32-bit float, 3-plane image (CHW).
    pub fn nppiResize_32f_P3R(
        aSrc: *const *const f32,
        nSrcStep: i32,
        oSrcSize: NppiSize,
        oSrcROI: NppiRect,
        aDst: *const *mut f32,
        nDstStep: i32,
        oDstSize: NppiSize,
        oDstROI: NppiRect,
        eInterpolation: i32,
    ) -> NppStatus;

    /// Affine warp for 32-bit float, 1-channel image (masks).
    pub fn nppiWarpAffine_32f_C1R(
        pSrc: *const f32,
        oSrcSize: NppiSize,
        nSrcStep: i32,
        oSrcROI: NppiRect,
        pDst: *mut f32,
        nDstStep: i32,
        oDstROI: NppiRect,
        aCoeffs: *const [[f64; 3]; 2],
        eInterpolation: i32,
    ) -> NppStatus;

    /// Bilinear resize for 32-bit float, 3-channel image.
    pub fn nppiResize_32f_C3R(
        pSrc: *const f32,
        nSrcStep: i32,
        oSrcSize: NppiSize,
        oSrcROI: NppiRect,
        pDst: *mut f32,
        nDstStep: i32,
        oDstSize: NppiSize,
        oDstROI: NppiRect,
        eInterpolation: i32,
    ) -> NppStatus;

    /// Bilinear resize for 32-bit float, 1-channel image (masks).
    pub fn nppiResize_32f_C1R(
        pSrc: *const f32,
        nSrcStep: i32,
        oSrcSize: NppiSize,
        oSrcROI: NppiRect,
        pDst: *mut f32,
        nDstStep: i32,
        oDstSize: NppiSize,
        oDstROI: NppiRect,
        eInterpolation: i32,
    ) -> NppStatus;
}

// ── NPP Image Filtering (libnppif) ──────────────────────────────────────

unsafe extern "C" {
    /// Gaussian filter for 32-bit float, 1-channel image.
    pub fn nppiFilterGauss_32f_C1R(
        pSrc: *const f32,
        nSrcStep: i32,
        pDst: *mut f32,
        nDstStep: i32,
        oSizeROI: NppiSize,
        eMaskSize: i32,
    ) -> NppStatus;

    /// Gaussian filter for 32-bit float, 3-channel image.
    pub fn nppiFilterGauss_32f_C3R(
        pSrc: *const f32,
        nSrcStep: i32,
        pDst: *mut f32,
        nDstStep: i32,
        oSizeROI: NppiSize,
        eMaskSize: i32,
    ) -> NppStatus;
}

// ── NPP Morphological Operations ────────────────────────────────────────

unsafe extern "C" {
    /// 3x3 dilate for 32-bit float, 1-channel (replaces max_pool2d for masks).
    pub fn nppiDilate3x3_32f_C1R(
        pSrc: *const f32,
        nSrcStep: i32,
        pDst: *mut f32,
        nDstStep: i32,
        oSizeROI: NppiSize,
    ) -> NppStatus;

    /// 3x3 erode for 32-bit float, 1-channel.
    pub fn nppiErode3x3_32f_C1R(
        pSrc: *const f32,
        nSrcStep: i32,
        pDst: *mut f32,
        nDstStep: i32,
        oSizeROI: NppiSize,
    ) -> NppStatus;
}

// ── NPP Color Conversion (libnppicc) ────────────────────────────────────

unsafe extern "C" {
    /// Swap channels for 8-bit unsigned, 3-channel (RGB↔BGR).
    pub fn nppiSwapChannels_8u_C3R(
        pSrc: *const u8,
        nSrcStep: i32,
        pDst: *mut u8,
        nDstStep: i32,
        oSizeROI: NppiSize,
        aDstOrder: *const i32,
    ) -> NppStatus;
}

// ── Safe Rust Wrappers ──────────────────────────────────────────────────

/// Check NPP status and convert to Result.
#[inline]
pub fn check_npp(status: NppStatus, op: &str) -> Result<(), NppError> {
    if status == NPP_SUCCESS {
        Ok(())
    } else {
        Err(NppError {
            status,
            op: op.to_string(),
        })
    }
}

#[derive(Debug, thiserror::Error)]
#[error("NPP error {status} in {op}")]
pub struct NppError {
    pub status: NppStatus,
    pub op: String,
}

/// Set NPP to use our cudarc CUDA stream.
///
/// # Safety
/// The stream must be a valid CUDA stream pointer and remain alive.
pub unsafe fn set_npp_stream(stream_ptr: *mut std::ffi::c_void) -> Result<(), NppError> {
    check_npp(unsafe { nppSetStream(stream_ptr) }, "nppSetStream")
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
            nppiResize_32f_C3R(
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
            nppiWarpAffine_32f_C3R(
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
            nppiWarpAffine_32f_P3R(
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
            nppiResize_32f_P3R(
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
        unsafe { nppiFilterGauss_32f_C1R(src, step, dst, step, roi, mask_size) },
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
        unsafe { nppiDilate3x3_32f_C1R(src, step, dst, step, roi) },
        "nppiDilate3x3_32f_C1R",
    )
}
