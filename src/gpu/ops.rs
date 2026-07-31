//! GPU operations — unified interface for all CUDA kernels.
//!
//! All ops work on CudaSlice<f32> buffers, zero CPU involvement.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    DriverError, LaunchConfig, PushKernelArg,
    sys::{CUevent_flags, CUresult},
};
use cudarc::nvrtc::Ptx;
use std::cell::RefCell;

use crate::gpu::npp;

/// Profiling stage labels — must stay in sync with process_frame_gpu marks.
pub const PROFILE_STAGES: &[&str] = &[
    "start",
    "after_detect",
    "after_recognize",
    "after_warp_swap",
    "after_swap_ort",
    "after_resize_512",
    "after_mask_gen",
    "after_paste_back",
];

/// Per-stage CUDA events for in-stream profiling (no serialization).
pub struct ProfilingState {
    pub events: Vec<CudaEvent>,
    pub frame_count: u64,
    pub report_every: u64,
    pub last_active_idx: usize,
}

/// All loaded GPU kernels + launch methods.
pub struct GpuOps {
    pub stream: Arc<CudaStream>,
    // Kernels
    normalize_fn: CudaFunction,
    denormalize_fn: CudaFunction,
    interlace_extract_fn: CudaFunction,
    interlace_scatter_fn: CudaFunction,
    enhancer_pack_tiles_fn: CudaFunction,
    enhancer_scatter_tiles_fn: CudaFunction,
    warp_affine_fn: CudaFunction,
    matmul_512_fn: CudaFunction,
    l2_normalize_fn: CudaFunction,
    resize_fn: CudaFunction,
    paste_back_fn: CudaFunction,
    border_oval_mask_fn: CudaFunction,
    alpha_blend_fn: CudaFunction,
    scalar_blend_inplace_fn: CudaFunction,
    // Frame conversion kernels
    hwc_u8_to_chw_f32_fn: CudaFunction,
    chw_f32_to_hwc_u8_fn: CudaFunction,
    chw_f32_to_nv12_scaled_fn: CudaFunction,
    letterbox_fn: CudaFunction,
    affine_scale_fn: CudaFunction,
    // Mask kernels
    blur_h_fn: CudaFunction,
    blur_v_fn: CudaFunction,
    mask_resize_fn: CudaFunction,
    mask_mul_fn: CudaFunction,
    occluder_threshold_fn: CudaFunction,
    xseg_postprocess_fn: CudaFunction,
    imagenet_normalize_fn: CudaFunction,
    parser_argmax_fn: CudaFunction,
    parser_class_mask_fn: CudaFunction,
    mask_invert_fn: CudaFunction,
    semantic_region_mask_fn: CudaFunction,
    semantic_temporal_mask_fn: CudaFunction,
    semantic_mark_valid_fn: CudaFunction,
    semantic_region_stats_fn: CudaFunction,
    semantic_composite_fn: CudaFunction,
    // Profiling (cell for interior mutability — methods only take &self)
    profiling: RefCell<ProfilingState>,
}

// SAFETY: `GpuOps` holds a `RefCell<ProfilingState>` which is `!Sync` by default.
// The profiling state is only accessed from the worker thread that owns the
// CUDA stream — there is no cross-thread mutation. `Arc<GpuOps>` is shared
// between the UI thread (which never touches profiling) and the worker thread
// (which owns the stream + profiling). Marking `Sync` is safe under this
// single-worker invariant.
unsafe impl Send for GpuOps {}
unsafe impl Sync for GpuOps {}

impl GpuOps {
    /// Load all PTX kernels. Panics if any kernel fails to load.
    pub fn new(ctx: &Arc<CudaContext>, stream: Arc<CudaStream>) -> Result<Self, DriverError> {
        let out_dir = env!("OUT_DIR");

        unsafe {
            npp::set_npp_stream(stream.cu_stream() as *mut std::ffi::c_void).map_err(|e| {
                tracing::error!("nppSetStream failed: {e}");
                DriverError(CUresult::CUDA_ERROR_UNKNOWN)
            })?;
        }

        let load = |name: &str, entry: &str| -> CudaFunction {
            let ptx = Ptx::from_file(format!("{out_dir}/{name}.ptx"));
            let module = ctx
                .load_module(ptx)
                .unwrap_or_else(|e| panic!("Failed to load {name}.ptx: {e}"));
            module
                .load_function(entry)
                .unwrap_or_else(|e| panic!("Failed to load {entry} from {name}.ptx: {e}"))
        };

        // Create timing-enabled events (CU_EVENT_DEFAULT = 0 allows elapsed_ms)
        let mut events = Vec::with_capacity(PROFILE_STAGES.len());
        for _ in 0..PROFILE_STAGES.len() {
            events.push(ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?);
        }

        Ok(Self {
            stream,
            normalize_fn: load("normalize", "normalize_kernel"),
            denormalize_fn: load("normalize", "denormalize_kernel"),
            interlace_extract_fn: load("interlace", "interlace_extract_kernel"),
            interlace_scatter_fn: load("interlace", "interlace_scatter_kernel"),
            enhancer_pack_tiles_fn: load("enhancer_tiles", "enhancer_pack_tiles_kernel"),
            enhancer_scatter_tiles_fn: load("enhancer_tiles", "enhancer_scatter_tiles_kernel"),
            warp_affine_fn: load("warp_affine", "warp_affine_chw_kernel"),
            resize_fn: load("warp_affine", "resize_chw_kernel"),
            paste_back_fn: load("paste_back", "paste_back_kernel"),
            border_oval_mask_fn: load("border_mask", "border_oval_mask_kernel"),
            alpha_blend_fn: load("alpha_blend", "alpha_blend_kernel"),
            scalar_blend_inplace_fn: load("alpha_blend", "scalar_blend_inplace_kernel"),
            hwc_u8_to_chw_f32_fn: load("frame_convert", "hwc_u8_to_chw_f32_kernel"),
            chw_f32_to_hwc_u8_fn: load("frame_convert", "chw_f32_to_hwc_u8_kernel"),
            chw_f32_to_nv12_scaled_fn: load("frame_convert", "chw_f32_to_nv12_scaled_kernel"),
            letterbox_fn: load("frame_convert", "letterbox_resize_kernel"),
            affine_scale_fn: load("frame_convert", "affine_scale_kernel"),
            matmul_512_fn: load("matmul_512", "matmul_512_kernel"),
            l2_normalize_fn: load("matmul_512", "l2_normalize_kernel"),
            blur_h_fn: load("gaussian_blur", "gaussian_blur_h_kernel"),
            blur_v_fn: load("gaussian_blur", "gaussian_blur_v_kernel"),
            mask_resize_fn: load("gaussian_blur", "mask_resize_kernel"),
            mask_mul_fn: load("gaussian_blur", "mask_mul_kernel"),
            occluder_threshold_fn: load("mask_postprocess", "occluder_threshold_kernel"),
            xseg_postprocess_fn: load("mask_postprocess", "xseg_postprocess_kernel"),
            imagenet_normalize_fn: load("mask_postprocess", "imagenet_normalize_kernel"),
            parser_argmax_fn: load("mask_postprocess", "parser_argmax_kernel"),
            parser_class_mask_fn: load("mask_postprocess", "parser_class_mask_kernel"),
            mask_invert_fn: load("mask_postprocess", "mask_invert_kernel"),
            semantic_region_mask_fn: load("mask_postprocess", "semantic_region_mask_kernel"),
            semantic_temporal_mask_fn: load("mask_postprocess", "semantic_temporal_mask_kernel"),
            semantic_mark_valid_fn: load("mask_postprocess", "semantic_mark_valid_kernel"),
            semantic_region_stats_fn: load("mask_postprocess", "semantic_region_stats_kernel"),
            semantic_composite_fn: load("mask_postprocess", "semantic_composite_kernel"),
            profiling: RefCell::new(ProfilingState {
                events,
                frame_count: 0,
                report_every: 30,
                last_active_idx: 0,
            }),
        })
    }

    // ── Profiling ───────────────────────────────────────────────────

    /// Record a CUDA event at this point in the stream. Cheap (~µs).
    pub fn profile_mark(&self, idx: usize) -> Result<(), DriverError> {
        let prof = self.profiling.borrow();
        if idx < prof.events.len() {
            prof.events[idx].record(&self.stream)?;
        }
        Ok(())
    }

    /// Update last_active_idx (must be called with highest recorded index).
    pub fn profile_set_active(&self, idx: usize) {
        self.profiling.borrow_mut().last_active_idx = idx;
    }

    /// Tick frame counter; every `report_every` frames, synchronize the last
    /// event, compute elapsed times between consecutive events, and log them.
    pub fn profile_tick(&self) -> Result<(), DriverError> {
        let mut prof = self.profiling.borrow_mut();
        prof.frame_count += 1;
        if !prof.frame_count.is_multiple_of(prof.report_every) {
            return Ok(());
        }
        let last = prof.last_active_idx;
        if last == 0 {
            return Ok(());
        }

        // Synchronize the last recorded event to ensure all prior work is done.
        prof.events[last].synchronize()?;

        let mut parts = Vec::with_capacity(last);
        for i in 0..last {
            let dt_ms = prof.events[i]
                .elapsed_ms(&prof.events[i + 1])
                .unwrap_or(0.0);
            parts.push(format!(
                "{}→{}={:.2}ms",
                PROFILE_STAGES[i],
                PROFILE_STAGES[i + 1],
                dt_ms
            ));
        }
        let total_ms = prof.events[0].elapsed_ms(&prof.events[last]).unwrap_or(0.0);
        tracing::info!("[profile] total={total_ms:.2}ms | {}", parts.join(" | "));
        Ok(())
    }

    // ── Kernel launchers ────────────────────────────────────────────

    /// Normalize in-place: data[i] /= 255.0
    pub fn normalize(&self, data: &mut CudaSlice<f32>) -> Result<(), DriverError> {
        self.normalize_prefix(data, data.len())
    }

    /// Normalize only the active prefix of a larger scratch buffer.
    pub fn normalize_prefix(
        &self,
        data: &mut CudaSlice<f32>,
        len: usize,
    ) -> Result<(), DriverError> {
        assert!(len <= data.len());
        let n = len as u32;
        let mut b = self.stream.launch_builder(&self.normalize_fn);
        b.arg(data);
        b.arg(&n);
        unsafe { b.launch(LaunchConfig::for_num_elems(n)) }?;
        Ok(())
    }

    /// Denormalize in-place: data[i] = clamp(data[i] * 255, 0, 255)
    pub fn denormalize(&self, data: &mut CudaSlice<f32>) -> Result<(), DriverError> {
        self.denormalize_prefix(data, data.len())
    }

    /// Denormalize only the active prefix of a larger scratch buffer.
    pub fn denormalize_prefix(
        &self,
        data: &mut CudaSlice<f32>,
        len: usize,
    ) -> Result<(), DriverError> {
        assert!(len <= data.len());
        let n = len as u32;
        let mut b = self.stream.launch_builder(&self.denormalize_fn);
        b.arg(data);
        b.arg(&n);
        unsafe { b.launch(LaunchConfig::for_num_elems(n)) }?;
        Ok(())
    }

    /// data[i] = data[i] * mul + add
    pub fn affine_scale(
        &self,
        data: &mut CudaSlice<f32>,
        mul: f32,
        add: f32,
    ) -> Result<(), DriverError> {
        let n = data.len() as u32;
        let mut b = self.stream.launch_builder(&self.affine_scale_fn);
        b.arg(data);
        b.arg(&n);
        b.arg(&mul);
        b.arg(&add);
        unsafe { b.launch(LaunchConfig::for_num_elems(n)) }?;
        Ok(())
    }

    /// Matrix-vector multiply: out[i] = sum_j(embedding[j] * emap[i][j]).
    /// `embedding` [512], `emap` [512*512] row-major, `output` [512].
    /// dim must be 512.
    pub fn matmul_512(
        &self,
        embedding: &CudaSlice<f32>,
        emap: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        dim: u32,
    ) -> Result<(), DriverError> {
        let mut b = self.stream.launch_builder(&self.matmul_512_fn);
        b.arg(embedding);
        b.arg(emap);
        b.arg(output);
        b.arg(&dim);
        unsafe { b.launch(LaunchConfig::for_num_elems(dim)) }?;
        Ok(())
    }

    /// L2 normalize a [dim] vector in-place.
    pub fn l2_normalize(&self, vec: &mut CudaSlice<f32>, dim: u32) -> Result<(), DriverError> {
        let mut b = self.stream.launch_builder(&self.l2_normalize_fn);
        b.arg(vec);
        b.arg(&dim);
        // l2_normalize uses shared memory with 512 threads max
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 2048,
            })
        }?;
        Ok(())
    }
    /// HWC u8 RGB [H, W, 3] → CHW f32 [3, H, W] in [0, 255].
    pub fn hwc_u8_to_chw_f32(
        &self,
        src: &CudaSlice<u8>,
        dst: &mut CudaSlice<f32>,
        h: u32,
        w: u32,
    ) -> Result<(), DriverError> {
        let total = h * w;
        let mut b = self.stream.launch_builder(&self.hwc_u8_to_chw_f32_fn);
        b.arg(src);
        b.arg(dst);
        b.arg(&h);
        b.arg(&w);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    /// CHW f32 [3, H, W] in [0, 255] → HWC u8 [H, W, 3].
    pub fn chw_f32_to_hwc_u8(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<u8>,
        h: u32,
        w: u32,
    ) -> Result<(), DriverError> {
        let total = h * w;
        let mut b = self.stream.launch_builder(&self.chw_f32_to_hwc_u8_fn);
        b.arg(src);
        b.arg(dst);
        b.arg(&h);
        b.arg(&w);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    /// Scale CHW f32 RGB directly into a tightly packed NV12 device buffer.
    pub fn chw_f32_to_nv12_scaled(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<u8>,
        src_h: u32,
        src_w: u32,
        dst_h: u32,
        dst_w: u32,
    ) -> Result<(), DriverError> {
        assert!(src_h > 0 && src_w > 0);
        assert!(dst_h > 0 && dst_w > 0 && dst_h.is_multiple_of(2) && dst_w.is_multiple_of(2));
        assert!(src.len() >= 3 * src_h as usize * src_w as usize);
        assert!(dst.len() >= dst_h as usize * dst_w as usize * 3 / 2);
        let total = dst_h * dst_w;
        let mut b = self.stream.launch_builder(&self.chw_f32_to_nv12_scaled_fn);
        b.arg(src);
        b.arg(dst);
        b.arg(&src_h);
        b.arg(&src_w);
        b.arg(&dst_h);
        b.arg(&dst_w);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    /// Letterbox resize + normalize. Destination area outside [0..new_h, 0..new_w]
    /// is filled with `add`. `new_h`/`new_w` must fit inside `target`.
    ///
    /// output[c, y, x] = (bilinear sample of src at scaled coord) * mul + add
    pub fn letterbox_resize(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        src_h: u32,
        src_w: u32,
        target: u32,
        new_h: u32,
        new_w: u32,
        mul: f32,
        add: f32,
    ) -> Result<(), DriverError> {
        let total = 3 * target * target;
        let mut b = self.stream.launch_builder(&self.letterbox_fn);
        b.arg(src);
        b.arg(dst);
        b.arg(&src_h);
        b.arg(&src_w);
        b.arg(&target);
        b.arg(&new_h);
        b.arg(&new_w);
        b.arg(&mul);
        b.arg(&add);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    /// NPP-accelerated affine warp for CHW f32 planar images. Uses
    /// `nppiWarpAffine_32f_P3R` which internally leverages CUDA texture
    /// memory — ~5× faster than the naive custom kernel.
    ///
    /// `fwd_affine` is a FORWARD matrix (src → dst), same convention as
    /// `cv2.warpAffine` / `skimage.SimilarityTransform`.
    pub fn warp_affine_npp(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        src_h: u32,
        src_w: u32,
        dst_h: u32,
        dst_w: u32,
        fwd_affine: &[[f64; 3]; 2],
    ) -> Result<(), DriverError> {
        let (src_ptr, _g1) = src.device_ptr(&self.stream);
        let (dst_ptr, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: pointers are live for the duration of the call; guards remain in scope.
        unsafe {
            npp::warp_affine_f32_p3(
                src_ptr as *const f32,
                src_w as i32,
                src_h as i32,
                dst_ptr as *mut f32,
                dst_w as i32,
                dst_h as i32,
                fwd_affine,
            )
            .map_err(|e| {
                tracing::error!("NPP warp_affine failed: {e}");
                DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN)
            })?;
        }
        Ok(())
    }

    /// NPP-accelerated bilinear resize for CHW f32 planar images.
    pub fn resize_npp(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        src_h: u32,
        src_w: u32,
        dst_h: u32,
        dst_w: u32,
    ) -> Result<(), DriverError> {
        let (src_ptr, _g1) = src.device_ptr(&self.stream);
        let (dst_ptr, _g2) = dst.device_ptr_mut(&self.stream);
        unsafe {
            npp::resize_f32_p3(
                src_ptr as *const f32,
                src_w as i32,
                src_h as i32,
                dst_ptr as *mut f32,
                dst_w as i32,
                dst_h as i32,
            )
            .map_err(|e| {
                tracing::error!("NPP resize failed: {e}");
                DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN)
            })?;
        }
        Ok(())
    }

    /// Affine warp: src [3, src_h, src_w] → dst [3, dst_h, dst_w]
    pub fn warp_affine(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        src_h: u32,
        src_w: u32,
        dst_h: u32,
        dst_w: u32,
        inv_affine: &[[f64; 3]; 2], // inverse affine (dst→src)
    ) -> Result<(), DriverError> {
        let total = dst_h * dst_w;
        let inv00 = inv_affine[0][0] as f32;
        let inv01 = inv_affine[0][1] as f32;
        let inv02 = inv_affine[0][2] as f32;
        let inv10 = inv_affine[1][0] as f32;
        let inv11 = inv_affine[1][1] as f32;
        let inv12 = inv_affine[1][2] as f32;

        let mut b = self.stream.launch_builder(&self.warp_affine_fn);
        b.arg(src);
        b.arg(dst);
        b.arg(&src_h);
        b.arg(&src_w);
        b.arg(&dst_h);
        b.arg(&dst_w);
        b.arg(&inv00);
        b.arg(&inv01);
        b.arg(&inv02);
        b.arg(&inv10);
        b.arg(&inv11);
        b.arg(&inv12);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    /// Bilinear resize: src [C, src_h, src_w] → dst [C, dst_h, dst_w]
    pub fn resize(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        src_h: u32,
        src_w: u32,
        dst_h: u32,
        dst_w: u32,
        channels: u32,
    ) -> Result<(), DriverError> {
        let total = dst_h * dst_w;
        let mut b = self.stream.launch_builder(&self.resize_fn);
        b.arg(src);
        b.arg(dst);
        b.arg(&src_h);
        b.arg(&src_w);
        b.arg(&dst_h);
        b.arg(&dst_w);
        b.arg(&channels);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    /// Interlace extract: face [3, dim*T, dim*T] → tiles [dim², 3, T, T]
    pub fn interlace_extract(
        &self,
        face: &CudaSlice<f32>,
        tiles: &mut CudaSlice<f32>,
        dim: u32,
        tile_size: u32,
    ) -> Result<(), DriverError> {
        let n = dim * dim * 3 * tile_size * tile_size;
        assert!((n as usize) <= tiles.len());
        let channels = 3u32;
        let mut b = self.stream.launch_builder(&self.interlace_extract_fn);
        b.arg(face);
        b.arg(tiles);
        b.arg(&dim);
        b.arg(&channels);
        b.arg(&tile_size);
        b.arg(&n);
        unsafe { b.launch(LaunchConfig::for_num_elems(n)) }?;
        Ok(())
    }

    /// Interlace scatter: tiles [dim², 3, T, T] → face [3, dim*T, dim*T]
    pub fn interlace_scatter(
        &self,
        tiles: &CudaSlice<f32>,
        face: &mut CudaSlice<f32>,
        dim: u32,
        tile_size: u32,
    ) -> Result<(), DriverError> {
        let n = dim * dim * 3 * tile_size * tile_size;
        assert!((n as usize) <= tiles.len());
        let channels = 3u32;
        let mut b = self.stream.launch_builder(&self.interlace_scatter_fn);
        b.arg(tiles);
        b.arg(face);
        b.arg(&dim);
        b.arg(&channels);
        b.arg(&tile_size);
        b.arg(&n);
        unsafe { b.launch(LaunchConfig::for_num_elems(n)) }?;
        Ok(())
    }

    /// Pack a CHW frame into row-major contiguous tiles, zero-padding edges.
    pub fn enhancer_pack_tiles(
        &self,
        frame: &CudaSlice<f32>,
        tiles: &mut CudaSlice<f32>,
        frame_h: u32,
        frame_w: u32,
        tiles_x: u32,
        tile_size: u32,
        tile_count: u32,
    ) -> Result<(), DriverError> {
        let total = tile_count * 3 * tile_size * tile_size;
        let mut b = self.stream.launch_builder(&self.enhancer_pack_tiles_fn);
        b.arg(frame);
        b.arg(tiles);
        b.arg(&frame_h);
        b.arg(&frame_w);
        b.arg(&tiles_x);
        b.arg(&tile_size);
        b.arg(&total);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    /// Scatter batched enhanced tiles into a cropped CHW output frame.
    pub fn enhancer_scatter_tiles(
        &self,
        tiles: &CudaSlice<f32>,
        frame: &mut CudaSlice<f32>,
        output_h: u32,
        output_w: u32,
        tiles_x: u32,
        output_tile: u32,
    ) -> Result<(), DriverError> {
        let total = 3 * output_h * output_w;
        let mut b = self.stream.launch_builder(&self.enhancer_scatter_tiles_fn);
        b.arg(tiles);
        b.arg(frame);
        b.arg(&output_h);
        b.arg(&output_w);
        b.arg(&tiles_x);
        b.arg(&output_tile);
        b.arg(&total);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    /// Paste back: inverse-map swapped face into frame with mask blending.
    pub fn paste_back(
        &self,
        frame: &mut CudaSlice<f32>, // [3, frame_h, frame_w]
        swap: &CudaSlice<f32>,      // [3, face_size, face_size]
        mask: &CudaSlice<f32>,      // [face_size, face_size]
        frame_h: u32,
        frame_w: u32,
        face_size: u32,
        bbox: [u32; 4],             // [left, top, right, bottom]
        fwd_affine: &[[f64; 3]; 2], // forward affine (frame→face)
    ) -> Result<(), DriverError> {
        let [left, top, right, bottom] = bbox;
        let total = (right - left) * (bottom - top);
        if total == 0 {
            return Ok(());
        }

        let a00 = fwd_affine[0][0] as f32;
        let a01 = fwd_affine[0][1] as f32;
        let a02 = fwd_affine[0][2] as f32;
        let a10 = fwd_affine[1][0] as f32;
        let a11 = fwd_affine[1][1] as f32;
        let a12 = fwd_affine[1][2] as f32;

        let mut b = self.stream.launch_builder(&self.paste_back_fn);
        b.arg(frame);
        b.arg(swap);
        b.arg(mask);
        b.arg(&frame_h);
        b.arg(&frame_w);
        b.arg(&face_size);
        b.arg(&left);
        b.arg(&top);
        b.arg(&right);
        b.arg(&bottom);
        b.arg(&a00);
        b.arg(&a01);
        b.arg(&a02);
        b.arg(&a10);
        b.arg(&a11);
        b.arg(&a12);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    /// Generate border + oval mask on GPU.
    pub fn border_oval_mask(
        &self,
        mask: &mut CudaSlice<f32>,
        size: u32,
        border_top: u32,
        border_bottom: u32,
        border_left: u32,
        border_right: u32,
        use_oval: bool,
    ) -> Result<(), DriverError> {
        let total = size * size;
        let oval_cx = size as f32 / 2.0;
        let oval_cy = size as f32 * 60.0 / 128.0;
        let oval_rx = size as f32 * 55.0 / 128.0;
        let oval_ry = size as f32 * 65.0 / 128.0;
        let oval_feather = size as f32 * 12.0 / 128.0;

        let b_bottom = size - border_bottom;
        let b_right = size - border_right;
        let use_oval = u32::from(use_oval);

        let mut b = self.stream.launch_builder(&self.border_oval_mask_fn);
        b.arg(mask);
        b.arg(&size);
        b.arg(&border_top);
        b.arg(&b_bottom);
        b.arg(&border_left);
        b.arg(&b_right);
        b.arg(&oval_cx);
        b.arg(&oval_cy);
        b.arg(&oval_rx);
        b.arg(&oval_ry);
        b.arg(&oval_feather);
        b.arg(&use_oval);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    /// Alpha blend: out = src * mask + dst * (1 - mask)
    pub fn alpha_blend(
        &self,
        src: &CudaSlice<f32>,
        dst: &CudaSlice<f32>,
        mask: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_pixels: u32,
        channels: u32,
    ) -> Result<(), DriverError> {
        let total = n_pixels * channels;
        let mut b = self.stream.launch_builder(&self.alpha_blend_fn);
        b.arg(src);
        b.arg(dst);
        b.arg(mask);
        b.arg(out);
        b.arg(&n_pixels);
        b.arg(&channels);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    /// Constant-alpha blend in place: dst = src * alpha + dst * (1-alpha).
    pub fn scalar_blend_inplace(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        len: usize,
        alpha: f32,
    ) -> Result<(), DriverError> {
        let n = len as u32;
        let alpha = alpha.clamp(0.0, 1.0);
        let mut b = self.stream.launch_builder(&self.scalar_blend_inplace_fn);
        b.arg(src);
        b.arg(dst);
        b.arg(&n);
        b.arg(&alpha);
        unsafe { b.launch(LaunchConfig::for_num_elems(n)) }?;
        Ok(())
    }

    /// Separable Gaussian blur on a single-channel mask. Uses `tmp` as scratch.
    /// Both `mask` and `tmp` must be sized `h * w`.
    pub fn gaussian_blur_mask(
        &self,
        mask: &mut CudaSlice<f32>,
        tmp: &mut CudaSlice<f32>,
        h: u32,
        w: u32,
        kernel: &CudaSlice<f32>,
        ks: u32,
    ) -> Result<(), DriverError> {
        if ks <= 1 {
            return Ok(());
        }
        let total = h * w;

        // Horizontal pass: mask (read) → tmp (write)
        {
            let mut b = self.stream.launch_builder(&self.blur_h_fn);
            b.arg(&*mask);
            b.arg(&mut *tmp);
            b.arg(kernel);
            b.arg(&h);
            b.arg(&w);
            b.arg(&ks);
            unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        }
        // Vertical pass: tmp (read) → mask (write)
        {
            let mut b = self.stream.launch_builder(&self.blur_v_fn);
            b.arg(&*tmp);
            b.arg(&mut *mask);
            b.arg(kernel);
            b.arg(&h);
            b.arg(&w);
            b.arg(&ks);
            unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        }
        Ok(())
    }

    /// Resize single-channel mask.
    pub fn mask_resize(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        src_h: u32,
        src_w: u32,
        dst_h: u32,
        dst_w: u32,
    ) -> Result<(), DriverError> {
        let total = dst_h * dst_w;
        let mut b = self.stream.launch_builder(&self.mask_resize_fn);
        b.arg(src);
        b.arg(dst);
        b.arg(&src_h);
        b.arg(&src_w);
        b.arg(&dst_h);
        b.arg(&dst_w);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    /// Element-wise multiply masks in place: a[i] *= b[i].
    pub fn mask_mul(
        &self,
        a: &mut CudaSlice<f32>,
        b_slice: &CudaSlice<f32>,
    ) -> Result<(), DriverError> {
        let n = a.len() as u32;
        let mut b = self.stream.launch_builder(&self.mask_mul_fn);
        b.arg(a);
        b.arg(b_slice);
        b.arg(&n);
        unsafe { b.launch(LaunchConfig::for_num_elems(n)) }?;
        Ok(())
    }

    pub fn occluder_threshold(&self, mask: &mut CudaSlice<f32>) -> Result<(), DriverError> {
        let total = mask.len() as u32;
        let mut b = self.stream.launch_builder(&self.occluder_threshold_fn);
        b.arg(mask);
        b.arg(&total);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    pub fn xseg_postprocess(&self, mask: &mut CudaSlice<f32>) -> Result<(), DriverError> {
        let total = mask.len() as u32;
        let mut b = self.stream.launch_builder(&self.xseg_postprocess_fn);
        b.arg(mask);
        b.arg(&total);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    pub fn imagenet_normalize_512(&self, image: &mut CudaSlice<f32>) -> Result<(), DriverError> {
        let plane = 512 * 512u32;
        let total = 3 * plane;
        let mut b = self.stream.launch_builder(&self.imagenet_normalize_fn);
        b.arg(image);
        b.arg(&plane);
        b.arg(&total);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    pub fn parser_argmax(
        &self,
        logits: &CudaSlice<f32>,
        classes: &mut CudaSlice<u8>,
    ) -> Result<(), DriverError> {
        let pixels = 512 * 512u32;
        let mut b = self.stream.launch_builder(&self.parser_argmax_fn);
        b.arg(logits);
        b.arg(classes);
        b.arg(&pixels);
        unsafe { b.launch(LaunchConfig::for_num_elems(pixels)) }?;
        Ok(())
    }

    pub fn parser_class_mask(
        &self,
        classes: &CudaSlice<u8>,
        mask: &mut CudaSlice<f32>,
        class_id: u32,
        foreground_mode: bool,
    ) -> Result<(), DriverError> {
        let pixels = 512 * 512u32;
        let foreground_mode = u32::from(foreground_mode);
        let mut b = self.stream.launch_builder(&self.parser_class_mask_fn);
        b.arg(classes);
        b.arg(mask);
        b.arg(&pixels);
        b.arg(&class_id);
        b.arg(&foreground_mode);
        unsafe { b.launch(LaunchConfig::for_num_elems(pixels)) }?;
        Ok(())
    }

    pub fn mask_invert(&self, mask: &mut CudaSlice<f32>) -> Result<(), DriverError> {
        let total = mask.len() as u32;
        let mut b = self.stream.launch_builder(&self.mask_invert_fn);
        b.arg(mask);
        b.arg(&total);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    pub fn semantic_region_mask(
        &self,
        classes: &CudaSlice<u8>,
        mask: &mut CudaSlice<f32>,
        count: &mut CudaSlice<u32>,
        region: u32,
    ) -> Result<(), DriverError> {
        let pixels = 512 * 512u32;
        self.stream.memcpy_htod(&[0u32], count)?;
        let mut b = self.stream.launch_builder(&self.semantic_region_mask_fn);
        b.arg(classes);
        b.arg(mask);
        b.arg(count);
        b.arg(&pixels);
        b.arg(&region);
        unsafe { b.launch(LaunchConfig::for_num_elems(pixels)) }?;
        Ok(())
    }

    pub fn semantic_temporal_mask(
        &self,
        current: &mut CudaSlice<f32>,
        previous: &mut CudaSlice<f32>,
        count: &CudaSlice<u32>,
        valid: &mut CudaSlice<u32>,
        alpha: f32,
    ) -> Result<(), DriverError> {
        let pixels = 512 * 512u32;
        let mut temporal = self.stream.launch_builder(&self.semantic_temporal_mask_fn);
        temporal.arg(current);
        temporal.arg(previous);
        temporal.arg(count);
        temporal.arg(&*valid);
        temporal.arg(&pixels);
        temporal.arg(&alpha);
        unsafe { temporal.launch(LaunchConfig::for_num_elems(pixels)) }?;

        let mut mark = self.stream.launch_builder(&self.semantic_mark_valid_fn);
        mark.arg(valid);
        mark.arg(count);
        unsafe { mark.launch(LaunchConfig::for_num_elems(1)) }?;
        Ok(())
    }

    pub fn semantic_region_stats(
        &self,
        swapped: &CudaSlice<f32>,
        original: &CudaSlice<f32>,
        mask: &CudaSlice<f32>,
        stats: &mut CudaSlice<f32>,
    ) -> Result<(), DriverError> {
        let pixels = 512 * 512u32;
        self.stream.memcpy_htod(&[0.0f32; 8], stats)?;
        let mut b = self.stream.launch_builder(&self.semantic_region_stats_fn);
        b.arg(swapped);
        b.arg(original);
        b.arg(mask);
        b.arg(stats);
        b.arg(&pixels);
        unsafe { b.launch(LaunchConfig::for_num_elems(pixels)) }?;
        Ok(())
    }

    pub fn semantic_composite(
        &self,
        swapped: &mut CudaSlice<f32>,
        original: &CudaSlice<f32>,
        mask: &CudaSlice<f32>,
        stats: &CudaSlice<f32>,
        count: &CudaSlice<u32>,
        blend: f32,
        luminance_factor: f32,
    ) -> Result<(), DriverError> {
        let pixels = 512 * 512u32;
        let total = 3 * pixels;
        let blend = blend.clamp(0.0, 1.0);
        let mut b = self.stream.launch_builder(&self.semantic_composite_fn);
        b.arg(swapped);
        b.arg(original);
        b.arg(mask);
        b.arg(stats);
        b.arg(count);
        b.arg(&pixels);
        b.arg(&blend);
        b.arg(&luminance_factor);
        b.arg(&total);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    /// Repeat CrossSwap's radius-N max-pool morphology using NPP 3x3 passes.
    pub fn morphology_mask(
        &self,
        mask: &mut CudaSlice<f32>,
        tmp: &mut CudaSlice<f32>,
        width: u32,
        height: u32,
        amount: i32,
    ) -> Result<(), DriverError> {
        let iterations = amount.unsigned_abs().min(100);
        for iteration in 0..iterations {
            let result = if iteration.is_multiple_of(2) {
                let (src_ptr, _src_guard) = mask.device_ptr(&self.stream);
                let (dst_ptr, _dst_guard) = tmp.device_ptr_mut(&self.stream);
                if amount > 0 {
                    unsafe {
                        npp::dilate_3x3_f32_c1(
                            src_ptr as *const f32,
                            dst_ptr as *mut f32,
                            width as i32,
                            height as i32,
                        )
                    }
                } else {
                    unsafe {
                        npp::erode_3x3_f32_c1(
                            src_ptr as *const f32,
                            dst_ptr as *mut f32,
                            width as i32,
                            height as i32,
                        )
                    }
                }
            } else {
                let (src_ptr, _src_guard) = tmp.device_ptr(&self.stream);
                let (dst_ptr, _dst_guard) = mask.device_ptr_mut(&self.stream);
                if amount > 0 {
                    unsafe {
                        npp::dilate_3x3_f32_c1(
                            src_ptr as *const f32,
                            dst_ptr as *mut f32,
                            width as i32,
                            height as i32,
                        )
                    }
                } else {
                    unsafe {
                        npp::erode_3x3_f32_c1(
                            src_ptr as *const f32,
                            dst_ptr as *mut f32,
                            width as i32,
                            height as i32,
                        )
                    }
                }
            };
            result.map_err(|error| {
                tracing::error!("NPP mask morphology failed: {error}");
                DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN)
            })?;
        }
        if !iterations.is_multiple_of(2) {
            self.stream.memcpy_dtod(tmp, mask)?;
        }
        Ok(())
    }

    // ── Allocation helpers ──────────────────────────────────────────

    /// Allocate a zeroed GPU buffer of f32.
    pub fn alloc_zeros(&self, len: usize) -> Result<CudaSlice<f32>, DriverError> {
        self.stream.alloc_zeros::<f32>(len)
    }

    /// Allocate a zeroed GPU buffer of u8.
    pub fn alloc_zeros_u8(&self, len: usize) -> Result<CudaSlice<u8>, DriverError> {
        self.stream.alloc_zeros::<u8>(len)
    }

    /// Upload f32 host data to GPU.
    pub fn upload(&self, data: &[f32]) -> Result<CudaSlice<f32>, DriverError> {
        self.stream.clone_htod(data)
    }

    /// Upload u8 host data to GPU.
    pub fn upload_u8(&self, data: &[u8]) -> Result<CudaSlice<u8>, DriverError> {
        self.stream.clone_htod(data)
    }

    /// Upload into an existing f32 buffer (asynchronous memcpy).
    pub fn upload_into(&self, data: &[f32], dst: &mut CudaSlice<f32>) -> Result<(), DriverError> {
        self.stream.memcpy_htod(data, dst)
    }

    /// Upload into an existing u8 buffer.
    pub fn upload_into_u8(&self, data: &[u8], dst: &mut CudaSlice<u8>) -> Result<(), DriverError> {
        self.stream.memcpy_htod(data, dst)
    }

    /// Download GPU data to host (f32).
    pub fn download(&self, buf: &CudaSlice<f32>) -> Result<Vec<f32>, DriverError> {
        self.stream.clone_dtoh(buf)
    }

    /// Download GPU data to host (u8).
    pub fn download_u8(&self, buf: &CudaSlice<u8>) -> Result<Vec<u8>, DriverError> {
        self.stream.clone_dtoh(buf)
    }

    /// Download GPU data into an existing host buffer (f32).
    pub fn download_into(&self, buf: &CudaSlice<f32>, dst: &mut [f32]) -> Result<(), DriverError> {
        self.stream.memcpy_dtoh(buf, dst)
    }

    /// Download GPU data into an existing host buffer (u8).
    pub fn download_into_u8(&self, buf: &CudaSlice<u8>, dst: &mut [u8]) -> Result<(), DriverError> {
        self.stream.memcpy_dtoh(buf, dst)
    }

    /// Synchronize the CUDA stream.
    pub fn sync(&self) -> Result<(), DriverError> {
        self.stream.synchronize()
    }
}
