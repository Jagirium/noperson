//! GPU operations — unified interface for all CUDA kernels.
//!
//! All ops work on CudaSlice<f32> buffers, zero CPU involvement.

use std::{collections::HashMap, sync::Arc};

use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    DriverError, LaunchConfig, PushKernelArg,
    sys::{CUevent_flags, CUresult},
};
use cudarc::nvrtc::Ptx;
use std::cell::RefCell;

use crate::gpu::npp;

include!(concat!(env!("OUT_DIR"), "/embedded_ptx.rs"));

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

fn profiling_requested(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|value| {
        let value = value.to_string_lossy();
        value == "1" || value.eq_ignore_ascii_case("true")
    })
}

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
    interlace_extract_fn: CudaFunction,
    interlace_scatter_fn: CudaFunction,
    enhancer_pack_tiles_fn: CudaFunction,
    enhancer_scatter_tiles_fn: CudaFunction,
    calc_latent_512_fn: CudaFunction,
    resize_fn: CudaFunction,
    paste_back_fn: CudaFunction,
    border_oval_mask_fn: CudaFunction,
    alpha_blend_fn: CudaFunction,
    scalar_blend_inplace_fn: CudaFunction,
    // Frame conversion kernels
    hwc_u8_to_chw_f32_fn: CudaFunction,
    chw_f32_to_hwc_u8_fn: CudaFunction,
    chw_f32_to_rgba_u8_pitched_fn: CudaFunction,
    nv12_to_chw_f32_fn: CudaFunction,
    chw_f32_to_nv12_scaled_fn: CudaFunction,
    letterbox_fn: CudaFunction,
    rotate_quadrants_fn: CudaFunction,
    affine_scale_fn: CudaFunction,
    affine_scale_copy_fn: CudaFunction,
    yolo_face_compact_fn: CudaFunction,
    anchor_face_compact_fn: CudaFunction,
    chw_rgb_to_nhwc_bgr_unit_fn: CudaFunction,
    nhwc_bgr_unit_to_chw_rgb_fn: CudaFunction,
    dfm_rct_stats_fn: CudaFunction,
    dfm_rct_apply_fn: CudaFunction,
    auto_color_dfl_stats_fn: CudaFunction,
    auto_color_dfl_apply_fn: CudaFunction,
    color_adjust_prep_fn: CudaFunction,
    color_contrast_saturation_fn: CudaFunction,
    color_sharpness_hue_noise_fn: CudaFunction,
    // Mask kernels
    blur_h_fn: CudaFunction,
    blur_v_fn: CudaFunction,
    blur_chw_h_fn: CudaFunction,
    blur_chw_v_fn: CudaFunction,
    mask_resize_fn: CudaFunction,
    mask_mul_fn: CudaFunction,
    occluder_threshold_fn: CudaFunction,
    xseg_postprocess_fn: CudaFunction,
    imagenet_normalize_copy_fn: CudaFunction,
    landmark_normalize_fn: CudaFunction,
    parser_argmax_fn: CudaFunction,
    parser_class_mask_fn: CudaFunction,
    parser_makeup_fn: CudaFunction,
    mask_invert_fn: CudaFunction,
    restore_ellipse_mask_fn: CudaFunction,
    fake_diff_mask_fn: CudaFunction,
    fake_diff_composite_fn: CudaFunction,
    fake_diff_composite_direct_fn: CudaFunction,
    semantic_region_mask_fn: CudaFunction,
    semantic_temporal_mask_fn: CudaFunction,
    semantic_mark_valid_fn: CudaFunction,
    semantic_region_stats_fn: CudaFunction,
    semantic_composite_fn: CudaFunction,
    // Profiling (cell for interior mutability — methods only take &self)
    profiling: Option<RefCell<ProfilingState>>,
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
        unsafe {
            npp::set_npp_stream(stream.cu_stream() as *mut std::ffi::c_void).map_err(|e| {
                tracing::error!("nppSetStream failed: {e}");
                DriverError(CUresult::CUDA_ERROR_UNKNOWN)
            })?;
        }

        // Several source files expose multiple entry points. Loading the same
        // PTX once per function makes CUDA re-JIT and retain duplicate modules,
        // which is especially expensive when constructing a shadow generation.
        let mut modules = HashMap::new();
        let mut load = |name: &str, entry: &str| -> CudaFunction {
            let module = modules.entry(name.to_owned()).or_insert_with(|| {
                let source = embedded_ptx(name)
                    .unwrap_or_else(|| panic!("PTX module was not embedded: {name}"));
                let ptx = Ptx::from_src(source);
                ctx.load_module(ptx)
                    .unwrap_or_else(|e| panic!("Failed to load {name}.ptx: {e}"))
            });
            module
                .load_function(entry)
                .unwrap_or_else(|e| panic!("Failed to load {entry} from {name}.ptx: {e}"))
        };

        // Profiling inserts eight CUDA events per face and an intentional
        // synchronization every report window. Keep it entirely off the
        // production hot path unless explicitly requested.
        let profiling = if profiling_requested(std::env::var_os("NOPERSON_PROFILE_GPU").as_deref())
        {
            let mut events = Vec::with_capacity(PROFILE_STAGES.len());
            for _ in 0..PROFILE_STAGES.len() {
                events.push(ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?);
            }
            Some(RefCell::new(ProfilingState {
                events,
                frame_count: 0,
                report_every: 30,
                last_active_idx: 0,
            }))
        } else {
            None
        };

        Ok(Self {
            stream,
            normalize_fn: load("normalize", "normalize_kernel"),
            interlace_extract_fn: load("interlace", "interlace_extract_normalized_kernel"),
            interlace_scatter_fn: load("interlace", "interlace_scatter_denormalized_kernel"),
            enhancer_pack_tiles_fn: load("enhancer_tiles", "enhancer_pack_tiles_kernel"),
            enhancer_scatter_tiles_fn: load("enhancer_tiles", "enhancer_scatter_tiles_kernel"),
            resize_fn: load("warp_affine", "resize_chw_kernel"),
            paste_back_fn: load("paste_back", "paste_back_kernel"),
            border_oval_mask_fn: load("border_mask", "border_oval_mask_kernel"),
            alpha_blend_fn: load("alpha_blend", "alpha_blend_kernel"),
            scalar_blend_inplace_fn: load("alpha_blend", "scalar_blend_inplace_kernel"),
            hwc_u8_to_chw_f32_fn: load("frame_convert", "hwc_u8_to_chw_f32_kernel"),
            chw_f32_to_hwc_u8_fn: load("frame_convert", "chw_f32_to_hwc_u8_kernel"),
            chw_f32_to_rgba_u8_pitched_fn: load(
                "frame_convert",
                "chw_f32_to_rgba_u8_pitched_kernel",
            ),
            nv12_to_chw_f32_fn: load("frame_convert", "nv12_to_chw_f32_kernel"),
            chw_f32_to_nv12_scaled_fn: load("frame_convert", "chw_f32_to_nv12_scaled_kernel"),
            letterbox_fn: load("frame_convert", "letterbox_resize_kernel"),
            rotate_quadrants_fn: load("rotate", "rotate_quadrants_chw_kernel"),
            affine_scale_fn: load("frame_convert", "affine_scale_kernel"),
            affine_scale_copy_fn: load("frame_convert", "affine_scale_copy_kernel"),
            yolo_face_compact_fn: load("detector_decode", "yolo_face_compact_kernel"),
            anchor_face_compact_fn: load("detector_decode", "anchor_face_compact_kernel"),
            chw_rgb_to_nhwc_bgr_unit_fn: load("layout_convert", "chw_rgb_to_nhwc_bgr_unit_kernel"),
            nhwc_bgr_unit_to_chw_rgb_fn: load("layout_convert", "nhwc_bgr_unit_to_chw_rgb_kernel"),
            dfm_rct_stats_fn: load("dfm_color", "dfm_rct_stats_kernel"),
            dfm_rct_apply_fn: load("dfm_color", "dfm_rct_apply_kernel"),
            auto_color_dfl_stats_fn: load("dfm_color", "auto_color_dfl_stats_kernel"),
            auto_color_dfl_apply_fn: load("dfm_color", "auto_color_dfl_apply_kernel"),
            color_adjust_prep_fn: load("color_adjust", "color_adjust_prep_kernel"),
            color_contrast_saturation_fn: load("color_adjust", "color_contrast_saturation_kernel"),
            color_sharpness_hue_noise_fn: load("color_adjust", "color_sharpness_hue_noise_kernel"),
            calc_latent_512_fn: load("matmul_512", "calc_latent_512_kernel"),
            blur_h_fn: load("gaussian_blur", "gaussian_blur_h_kernel"),
            blur_v_fn: load("gaussian_blur", "gaussian_blur_v_kernel"),
            blur_chw_h_fn: load("gaussian_blur", "gaussian_blur_chw_h_kernel"),
            blur_chw_v_fn: load("gaussian_blur", "gaussian_blur_chw_v_kernel"),
            mask_resize_fn: load("gaussian_blur", "mask_resize_kernel"),
            mask_mul_fn: load("gaussian_blur", "mask_mul_kernel"),
            occluder_threshold_fn: load("mask_postprocess", "occluder_threshold_kernel"),
            xseg_postprocess_fn: load("mask_postprocess", "xseg_postprocess_kernel"),
            imagenet_normalize_copy_fn: load("mask_postprocess", "imagenet_normalize_copy_kernel"),
            landmark_normalize_fn: load("mask_postprocess", "landmark_normalize_kernel"),
            parser_argmax_fn: load("mask_postprocess", "parser_argmax_kernel"),
            parser_class_mask_fn: load("mask_postprocess", "parser_class_mask_kernel"),
            parser_makeup_fn: load("mask_postprocess", "parser_makeup_kernel"),
            mask_invert_fn: load("mask_postprocess", "mask_invert_kernel"),
            restore_ellipse_mask_fn: load("mask_postprocess", "restore_ellipse_mask_kernel"),
            fake_diff_mask_fn: load("mask_postprocess", "fake_diff_mask_kernel"),
            fake_diff_composite_fn: load("mask_postprocess", "fake_diff_composite_kernel"),
            fake_diff_composite_direct_fn: load(
                "mask_postprocess",
                "fake_diff_composite_direct_kernel",
            ),
            semantic_region_mask_fn: load("mask_postprocess", "semantic_region_mask_kernel"),
            semantic_temporal_mask_fn: load("mask_postprocess", "semantic_temporal_mask_kernel"),
            semantic_mark_valid_fn: load("mask_postprocess", "semantic_mark_valid_kernel"),
            semantic_region_stats_fn: load("mask_postprocess", "semantic_region_stats_kernel"),
            semantic_composite_fn: load("mask_postprocess", "semantic_composite_kernel"),
            profiling,
        })
    }

    // ── Profiling ───────────────────────────────────────────────────

    /// Record a CUDA event at this point in the stream. Cheap (~µs).
    pub fn profile_mark(&self, idx: usize) -> Result<(), DriverError> {
        let Some(profiling) = &self.profiling else {
            return Ok(());
        };
        let prof = profiling.borrow();
        if idx < prof.events.len() {
            prof.events[idx].record(&self.stream)?;
        }
        Ok(())
    }

    /// Update last_active_idx (must be called with highest recorded index).
    pub fn profile_set_active(&self, idx: usize) {
        if let Some(profiling) = &self.profiling {
            profiling.borrow_mut().last_active_idx = idx;
        }
    }

    /// Tick frame counter; every `report_every` frames, synchronize the last
    /// event, compute elapsed times between consecutive events, and log them.
    pub fn profile_tick(&self) -> Result<(), DriverError> {
        let Some(profiling) = &self.profiling else {
            return Ok(());
        };
        let mut prof = profiling.borrow_mut();
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

    /// destination[i] = source[i] * mul + add
    pub fn affine_scale_copy(
        &self,
        source: &CudaSlice<f32>,
        destination: &mut CudaSlice<f32>,
        len: usize,
        mul: f32,
        add: f32,
    ) -> Result<(), DriverError> {
        assert!(len <= source.len() && len <= destination.len());
        let n = len as u32;
        let mut b = self.stream.launch_builder(&self.affine_scale_copy_fn);
        b.arg(source);
        b.arg(destination);
        b.arg(&n);
        b.arg(&mul);
        b.arg(&add);
        unsafe { b.launch(LaunchConfig::for_num_elems(n)) }?;
        Ok(())
    }

    pub fn compact_yolo_faces(
        &self,
        output: &CudaSlice<f32>,
        candidates: &mut CudaSlice<f32>,
        count: &mut CudaSlice<u32>,
        threshold: f32,
        scale: f32,
    ) -> Result<(), DriverError> {
        let anchors = 8400u32;
        debug_assert!(candidates.len() >= anchors as usize * 15);
        let mut b = self.stream.launch_builder(&self.yolo_face_compact_fn);
        b.arg(output);
        b.arg(candidates);
        b.arg(count);
        b.arg(&anchors);
        b.arg(&threshold);
        b.arg(&scale);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        Ok(())
    }

    pub fn compact_anchor_faces(
        &self,
        output: &CudaSlice<f32>,
        candidates: &mut CudaSlice<f32>,
        count: &mut CudaSlice<u32>,
        threshold: f32,
        scale: f32,
    ) -> Result<(), DriverError> {
        debug_assert!(candidates.len() >= 16_800 * 15);
        let mut b = self.stream.launch_builder(&self.anchor_face_compact_fn);
        b.arg(output);
        b.arg(candidates);
        b.arg(count);
        b.arg(&threshold);
        b.arg(&scale);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        Ok(())
    }

    /// Compute `L2norm(L2norm(embedding) @ emap)` in one CUDA block.
    pub fn calc_latent_512(
        &self,
        embedding: &mut CudaSlice<f32>,
        emap: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), DriverError> {
        debug_assert!(embedding.len() >= 512);
        debug_assert!(emap.len() >= 512 * 512);
        debug_assert!(output.len() >= 512);
        let mut b = self.stream.launch_builder(&self.calc_latent_512_fn);
        b.arg(embedding);
        b.arg(emap);
        b.arg(output);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (512, 1, 1),
                shared_mem_bytes: 0,
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

    /// CHW f32 [3,H,W] in [0,255] -> pitch-linear RGBA8.
    pub fn chw_f32_to_rgba_u8_pitched<D>(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut D,
        h: u32,
        w: u32,
        row_bytes: u32,
    ) -> Result<(), DriverError>
    where
        D: cudarc::driver::DevicePtrMut<u8>,
    {
        let total = h * w;
        let mut builder = self
            .stream
            .launch_builder(&self.chw_f32_to_rgba_u8_pitched_fn);
        let (dst_ptr, _dst_guard) = dst.device_ptr_mut(&self.stream);
        builder.arg(src);
        builder.arg(&dst_ptr);
        builder.arg(&h);
        builder.arg(&w);
        builder.arg(&row_bytes);
        unsafe { builder.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    /// Convert a pitch-linear NVDEC NV12/P010 surface directly into CHW f32.
    ///
    /// # Safety
    /// `source_device_ptr` must remain mapped and readable on this CUDA context
    /// until work submitted to this stream has completed.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn nv12_device_to_chw_f32(
        &self,
        source_device_ptr: u64,
        pitch: u32,
        destination: &mut CudaSlice<f32>,
        height: u32,
        width: u32,
        pixel_format: crate::io::native_video::PixelFormat,
        matrix: crate::io::native_video::ColorMatrix,
        range: crate::io::native_video::ColorRange,
    ) -> Result<(), DriverError> {
        use crate::io::native_video::{ColorMatrix, ColorRange, PixelFormat};

        assert!(source_device_ptr != 0 && pitch >= width);
        assert!(height > 0 && width > 0);
        assert!(destination.len() >= 3 * height as usize * width as usize);
        let matrix = match matrix {
            ColorMatrix::Unspecified if width < 1280 => ColorMatrix::Bt601,
            ColorMatrix::Unspecified => ColorMatrix::Bt709,
            matrix => matrix,
        };
        let limited = !matches!(range, ColorRange::Full);
        let (y_offset, y_scale) = if limited {
            (16.0_f32, 255.0 / 219.0)
        } else {
            (0.0_f32, 1.0_f32)
        };
        let (rv, gu, gv, bu): (f32, f32, f32, f32) = match (matrix, limited) {
            (ColorMatrix::Bt601, true) => (1.596, -0.392, -0.813, 2.017),
            (ColorMatrix::Bt709, true) => (1.793, -0.213, -0.533, 2.112),
            (ColorMatrix::Bt2020NonConstantLuminance, true) => (1.679, -0.187, -0.650, 2.142),
            (ColorMatrix::Bt601, false) => (1.402, -0.344_136, -0.714_136, 1.772),
            (ColorMatrix::Bt709, false) => (1.574_8, -0.187_324, -0.468_124, 1.855_6),
            (ColorMatrix::Bt2020NonConstantLuminance, false) => {
                (1.474_6, -0.164_553, -0.571_353, 1.881_4)
            }
            (ColorMatrix::Unspecified, _) => unreachable!(),
        };
        let p010 = u32::from(matches!(pixel_format, PixelFormat::P010));
        let total = height * width;
        let mut launch = self.stream.launch_builder(&self.nv12_to_chw_f32_fn);
        launch.arg(&source_device_ptr);
        launch.arg(&pitch);
        launch.arg(destination);
        launch.arg(&height);
        launch.arg(&width);
        launch.arg(&p010);
        launch.arg(&y_offset);
        launch.arg(&y_scale);
        launch.arg(&rv);
        launch.arg(&gu);
        launch.arg(&gv);
        launch.arg(&bu);
        unsafe { launch.launch(LaunchConfig::for_num_elems(total)) }?;
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
        self.chw_f32_to_nv12_scaled_color(
            src,
            dst,
            src_h,
            src_w,
            dst_h,
            dst_w,
            crate::io::native_video::ColorMatrix::Bt601,
            crate::io::native_video::ColorRange::Limited,
            crate::io::native_video::PixelFormat::Nv12,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chw_f32_to_nv12_scaled_color(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<u8>,
        src_h: u32,
        src_w: u32,
        dst_h: u32,
        dst_w: u32,
        matrix: crate::io::native_video::ColorMatrix,
        range: crate::io::native_video::ColorRange,
        pixel_format: crate::io::native_video::PixelFormat,
    ) -> Result<(), DriverError> {
        use crate::io::native_video::PixelFormat;

        assert!(dst.len() >= dst_h as usize * dst_w as usize * 3 / 2);
        let pitch = dst_w
            * if matches!(pixel_format, PixelFormat::P010) {
                2
            } else {
                1
            };
        let (device_ptr, _write_guard) = dst.device_ptr_mut(&self.stream);
        unsafe {
            self.chw_f32_to_pitched_nv12_scaled_color(
                src,
                device_ptr,
                pitch,
                src_h,
                src_w,
                dst_h,
                dst_w,
                matrix,
                range,
                pixel_format,
            )
        }
    }

    /// Scale CHW RGB directly into an externally-owned pitch-linear NV12/P010 surface.
    ///
    /// # Safety
    /// `dst_device_ptr` must refer to at least `dst_h * 3 / 2` rows of
    /// `dst_pitch` bytes on this CUDA context and must outlive all queued work.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn chw_f32_to_pitched_nv12_scaled_color(
        &self,
        src: &CudaSlice<f32>,
        dst_device_ptr: u64,
        dst_pitch: u32,
        src_h: u32,
        src_w: u32,
        dst_h: u32,
        dst_w: u32,
        matrix: crate::io::native_video::ColorMatrix,
        range: crate::io::native_video::ColorRange,
        pixel_format: crate::io::native_video::PixelFormat,
    ) -> Result<(), DriverError> {
        use crate::io::native_video::{ColorMatrix, ColorRange, PixelFormat};

        assert!(src_h > 0 && src_w > 0);
        assert!(dst_h > 0 && dst_w > 0 && dst_h.is_multiple_of(2) && dst_w.is_multiple_of(2));
        assert!(src.len() >= 3 * src_h as usize * src_w as usize);
        let row_bytes = dst_w
            * if matches!(pixel_format, PixelFormat::P010) {
                2
            } else {
                1
            };
        assert!(dst_device_ptr != 0 && dst_pitch >= row_bytes);
        let matrix = match matrix {
            ColorMatrix::Unspecified if dst_w < 1280 => ColorMatrix::Bt601,
            ColorMatrix::Unspecified => ColorMatrix::Bt709,
            matrix => matrix,
        };
        let full = matches!(range, ColorRange::Full);
        let coefficients: [f32; 11] = match (matrix, full) {
            (ColorMatrix::Bt601, false) => [
                0.257, 0.504, 0.098, 16.0, -0.148, -0.291, 0.439, 0.439, -0.368, -0.071, 128.0,
            ],
            (ColorMatrix::Bt709, false) => [
                0.183, 0.614, 0.062, 16.0, -0.101, -0.339, 0.439, 0.439, -0.399, -0.040, 128.0,
            ],
            (ColorMatrix::Bt2020NonConstantLuminance, false) => [
                0.2256, 0.5823, 0.0509, 16.0, -0.1227, -0.3166, 0.4392, 0.4392, -0.4039, -0.0353,
                128.0,
            ],
            (ColorMatrix::Bt601, true) => [
                0.299, 0.587, 0.114, 0.0, -0.168_736, -0.331_264, 0.5, 0.5, -0.418_688, -0.081_312,
                128.0,
            ],
            (ColorMatrix::Bt709, true) => [
                0.2126, 0.7152, 0.0722, 0.0, -0.114_572, -0.385_428, 0.5, 0.5, -0.454_153,
                -0.045_847, 128.0,
            ],
            (ColorMatrix::Bt2020NonConstantLuminance, true) => [
                0.2627, 0.678, 0.0593, 0.0, -0.139_63, -0.360_37, 0.5, 0.5, -0.459_786, -0.040_214,
                128.0,
            ],
            (ColorMatrix::Unspecified, _) => unreachable!(),
        };
        let total = dst_h * dst_w;
        let mut b = self.stream.launch_builder(&self.chw_f32_to_nv12_scaled_fn);
        b.arg(src);
        b.arg(&dst_device_ptr);
        b.arg(&dst_pitch);
        b.arg(&src_h);
        b.arg(&src_w);
        b.arg(&dst_h);
        b.arg(&dst_w);
        let p010 = u32::from(matches!(pixel_format, PixelFormat::P010));
        b.arg(&p010);
        for coefficient in &coefficients {
            b.arg(coefficient);
        }
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

    /// Exact CHW rotation by a multiple of 90 degrees.
    pub fn rotate_quadrants(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        src_h: u32,
        src_w: u32,
        quarter_turns: u32,
    ) -> Result<(), DriverError> {
        let turns = quarter_turns & 3;
        let total = 3 * src_h * src_w;
        let mut builder = self.stream.launch_builder(&self.rotate_quadrants_fn);
        builder.arg(src);
        builder.arg(dst);
        builder.arg(&src_h);
        builder.arg(&src_w);
        builder.arg(&turns);
        unsafe { builder.launch(LaunchConfig::for_num_elems(total)) }?;
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

    pub fn chw_rgb_to_nhwc_bgr_unit(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        pixels: u32,
    ) -> Result<(), DriverError> {
        let total = pixels * 3;
        let mut b = self
            .stream
            .launch_builder(&self.chw_rgb_to_nhwc_bgr_unit_fn);
        b.arg(src);
        b.arg(dst);
        b.arg(&pixels);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    pub fn nhwc_bgr_unit_to_chw_rgb(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        pixels: u32,
    ) -> Result<(), DriverError> {
        let total = pixels * 3;
        let mut b = self
            .stream
            .launch_builder(&self.nhwc_bgr_unit_to_chw_rgb_fn);
        b.arg(src);
        b.arg(dst);
        b.arg(&pixels);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    pub fn dfm_rct(
        &self,
        source_nhwc: &mut CudaSlice<f32>,
        like_nhwc: &CudaSlice<f32>,
        mask: &CudaSlice<f32>,
        stats: &mut CudaSlice<f32>,
        pixels: u32,
        cutoff: f32,
    ) -> Result<(), DriverError> {
        self.stream.memcpy_htod(&[0.0f32; 12], stats)?;
        let mut reduce = self.stream.launch_builder(&self.dfm_rct_stats_fn);
        reduce.arg(&*source_nhwc);
        reduce.arg(like_nhwc);
        reduce.arg(mask);
        reduce.arg(&mut *stats);
        reduce.arg(&pixels);
        reduce.arg(&cutoff);
        unsafe { reduce.launch(LaunchConfig::for_num_elems(pixels)) }?;

        let mut apply = self.stream.launch_builder(&self.dfm_rct_apply_fn);
        apply.arg(source_nhwc);
        apply.arg(&*stats);
        apply.arg(&pixels);
        unsafe { apply.launch(LaunchConfig::for_num_elems(pixels)) }?;
        Ok(())
    }

    pub fn auto_color_dfl(
        &self,
        original_chw: &CudaSlice<f32>,
        swapped_chw: &mut CudaSlice<f32>,
        mask: &CudaSlice<f32>,
        stats: &mut CudaSlice<f32>,
        pixels: u32,
        use_mask: bool,
        blend: f32,
    ) -> Result<(), DriverError> {
        self.stream.memcpy_htod(&[0.0f32; 13], stats)?;
        let use_mask = u32::from(use_mask);
        let mut reduce = self.stream.launch_builder(&self.auto_color_dfl_stats_fn);
        reduce.arg(original_chw);
        reduce.arg(&*swapped_chw);
        reduce.arg(mask);
        reduce.arg(&mut *stats);
        reduce.arg(&pixels);
        reduce.arg(&use_mask);
        unsafe { reduce.launch(LaunchConfig::for_num_elems(pixels)) }?;

        let mut apply = self.stream.launch_builder(&self.auto_color_dfl_apply_fn);
        apply.arg(swapped_chw);
        apply.arg(&*stats);
        apply.arg(&pixels);
        apply.arg(&blend);
        unsafe { apply.launch(LaunchConfig::for_num_elems(pixels)) }?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn adjust_color(
        &self,
        image: &mut CudaSlice<f32>,
        scratch: &mut CudaSlice<f32>,
        gray_sum: &mut CudaSlice<u32>,
        width: u32,
        height: u32,
        gamma: f32,
        offsets: [f32; 3],
        brightness: f32,
        contrast: f32,
        saturation: f32,
        sharpness: f32,
        hue: f32,
        noise: f32,
        seed: u32,
    ) -> Result<(), DriverError> {
        let pixels = width * height;
        self.stream.memcpy_htod(&[0u32], gray_sum)?;
        let mut prep = self.stream.launch_builder(&self.color_adjust_prep_fn);
        prep.arg(&mut *image);
        prep.arg(&mut *gray_sum);
        prep.arg(&pixels);
        prep.arg(&gamma);
        prep.arg(&offsets[0]);
        prep.arg(&offsets[1]);
        prep.arg(&offsets[2]);
        prep.arg(&brightness);
        unsafe { prep.launch(LaunchConfig::for_num_elems(pixels)) }?;

        let mut color = self
            .stream
            .launch_builder(&self.color_contrast_saturation_fn);
        color.arg(&mut *image);
        color.arg(&*gray_sum);
        color.arg(&pixels);
        color.arg(&contrast);
        color.arg(&saturation);
        unsafe { color.launch(LaunchConfig::for_num_elems(pixels)) }?;

        let mut finish = self
            .stream
            .launch_builder(&self.color_sharpness_hue_noise_fn);
        finish.arg(&*image);
        finish.arg(&mut *scratch);
        finish.arg(&width);
        finish.arg(&height);
        finish.arg(&sharpness);
        finish.arg(&hue);
        finish.arg(&noise);
        finish.arg(&seed);
        unsafe { finish.launch(LaunchConfig::for_num_elems(pixels)) }?;
        self.stream.memcpy_dtod(scratch, image)?;
        Ok(())
    }

    /// Interlace extract: face [3, dim*T, dim*T] → tiles [dim², 3, T, T]
    pub fn interlace_extract_normalized(
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
    pub fn interlace_scatter_denormalized(
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
        self.gaussian_blur_mask_with_border(mask, tmp, h, w, kernel, ks, 0)
    }

    /// `border_mode`: 0 = torchvision reflect, 1 = conv2d zero padding.
    pub fn gaussian_blur_mask_with_border(
        &self,
        mask: &mut CudaSlice<f32>,
        tmp: &mut CudaSlice<f32>,
        h: u32,
        w: u32,
        kernel: &CudaSlice<f32>,
        ks: u32,
        border_mode: u32,
    ) -> Result<(), DriverError> {
        if ks <= 1 {
            return Ok(());
        }
        debug_assert!(ks <= 65);
        let grid = (w.div_ceil(32), h.div_ceil(8), 1);

        // Horizontal pass: mask (read) → tmp (write)
        {
            let mut b = self.stream.launch_builder(&self.blur_h_fn);
            b.arg(&*mask);
            b.arg(&mut *tmp);
            b.arg(kernel);
            b.arg(&h);
            b.arg(&w);
            b.arg(&ks);
            b.arg(&border_mode);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: grid,
                    block_dim: (32, 8, 1),
                    shared_mem_bytes: (32 + ks - 1) * 8 * 4,
                })
            }?;
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
            b.arg(&border_mode);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: grid,
                    block_dim: (32, 8, 1),
                    shared_mem_bytes: 32 * (8 + ks - 1) * 4,
                })
            }?;
        }
        Ok(())
    }

    /// Separable torchvision-compatible Gaussian blur on a three-channel CHW image.
    pub fn gaussian_blur_chw(
        &self,
        image: &mut CudaSlice<f32>,
        tmp: &mut CudaSlice<f32>,
        h: u32,
        w: u32,
        kernel: &CudaSlice<f32>,
        ks: u32,
    ) -> Result<(), DriverError> {
        if ks <= 1 {
            return Ok(());
        }
        debug_assert!(ks <= 65);
        let grid = (w.div_ceil(32), h.div_ceil(8), 3);
        let mut horizontal = self.stream.launch_builder(&self.blur_chw_h_fn);
        horizontal.arg(&*image);
        horizontal.arg(&mut *tmp);
        horizontal.arg(kernel);
        horizontal.arg(&h);
        horizontal.arg(&w);
        horizontal.arg(&ks);
        unsafe {
            horizontal.launch(LaunchConfig {
                grid_dim: grid,
                block_dim: (32, 8, 1),
                shared_mem_bytes: (32 + ks - 1) * 8 * 4,
            })
        }?;

        let mut vertical = self.stream.launch_builder(&self.blur_chw_v_fn);
        vertical.arg(&*tmp);
        vertical.arg(&mut *image);
        vertical.arg(kernel);
        vertical.arg(&h);
        vertical.arg(&w);
        vertical.arg(&ks);
        unsafe {
            vertical.launch(LaunchConfig {
                grid_dim: grid,
                block_dim: (32, 8, 1),
                shared_mem_bytes: 32 * (8 + ks - 1) * 4,
            })
        }?;
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

    pub fn imagenet_normalize_copy_512(
        &self,
        source: &CudaSlice<f32>,
        destination: &mut CudaSlice<f32>,
    ) -> Result<(), DriverError> {
        let plane = 512 * 512u32;
        let total = 3 * plane;
        let mut b = self.stream.launch_builder(&self.imagenet_normalize_copy_fn);
        b.arg(source);
        b.arg(destination);
        b.arg(&plane);
        b.arg(&total);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    pub fn landmark_normalize(
        &self,
        image: &mut CudaSlice<f32>,
        pixels: u32,
        mode: u32,
    ) -> Result<(), DriverError> {
        let total = pixels * 3;
        let mut b = self.stream.launch_builder(&self.landmark_normalize_fn);
        b.arg(image);
        b.arg(&pixels);
        b.arg(&mode);
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

    #[allow(clippy::too_many_arguments)]
    pub fn parser_makeup(
        &self,
        image: &mut CudaSlice<f32>,
        classes: &CudaSlice<u8>,
        hair_enabled: bool,
        hair_color: [f32; 3],
        hair_blend: f32,
        lips_enabled: bool,
        lips_color: [f32; 3],
        lips_blend: f32,
    ) -> Result<(), DriverError> {
        let pixels = 512 * 512u32;
        let hair_enabled = u32::from(hair_enabled);
        let lips_enabled = u32::from(lips_enabled);
        let hair_blend = hair_blend.clamp(0.0, 1.0);
        let lips_blend = lips_blend.clamp(0.0, 1.0);
        let mut b = self.stream.launch_builder(&self.parser_makeup_fn);
        b.arg(image);
        b.arg(classes);
        b.arg(&pixels);
        b.arg(&hair_enabled);
        b.arg(&hair_color[0]);
        b.arg(&hair_color[1]);
        b.arg(&hair_color[2]);
        b.arg(&hair_blend);
        b.arg(&lips_enabled);
        b.arg(&lips_color[0]);
        b.arg(&lips_color[1]);
        b.arg(&lips_color[2]);
        b.arg(&lips_blend);
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

    #[allow(clippy::too_many_arguments)]
    pub fn restore_ellipse_mask(
        &self,
        mask: &mut CudaSlice<f32>,
        width: u32,
        height: u32,
        center_x: i32,
        center_y: i32,
        radius_x: i32,
        radius_y: i32,
        blend: f32,
        feather: u32,
    ) -> Result<(), DriverError> {
        let pixels = width * height;
        let blend = blend.clamp(0.0, 1.0);
        let mut b = self.stream.launch_builder(&self.restore_ellipse_mask_fn);
        b.arg(mask);
        b.arg(&width);
        b.arg(&height);
        b.arg(&center_x);
        b.arg(&center_y);
        b.arg(&radius_x);
        b.arg(&radius_y);
        b.arg(&blend);
        b.arg(&feather);
        unsafe { b.launch(LaunchConfig::for_num_elems(pixels)) }?;
        Ok(())
    }

    pub fn fake_diff_mask(
        &self,
        swapped: &CudaSlice<f32>,
        original: &CudaSlice<f32>,
        mask: &mut CudaSlice<f32>,
        pixels: u32,
        amount: u32,
    ) -> Result<(), DriverError> {
        let threshold = amount as f32 * 2.55;
        let mut b = self.stream.launch_builder(&self.fake_diff_mask_fn);
        b.arg(swapped);
        b.arg(original);
        b.arg(mask);
        b.arg(&pixels);
        b.arg(&threshold);
        unsafe { b.launch(LaunchConfig::for_num_elems(pixels)) }?;
        Ok(())
    }

    pub fn fake_diff_composite(
        &self,
        swapped: &mut CudaSlice<f32>,
        original: &CudaSlice<f32>,
        mask: &CudaSlice<f32>,
        pixels: u32,
    ) -> Result<(), DriverError> {
        let total = pixels * 3;
        let mut b = self.stream.launch_builder(&self.fake_diff_composite_fn);
        b.arg(swapped);
        b.arg(original);
        b.arg(mask);
        b.arg(&pixels);
        b.arg(&total);
        unsafe { b.launch(LaunchConfig::for_num_elems(total)) }?;
        Ok(())
    }

    pub fn fake_diff_composite_direct(
        &self,
        swapped: &mut CudaSlice<f32>,
        original: &CudaSlice<f32>,
        pixels: u32,
        amount: u32,
    ) -> Result<(), DriverError> {
        let threshold = amount as f32 * 2.55;
        let mut b = self
            .stream
            .launch_builder(&self.fake_diff_composite_direct_fn);
        b.arg(swapped);
        b.arg(original);
        b.arg(&pixels);
        b.arg(&threshold);
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

#[cfg(test)]
mod tests {
    use super::profiling_requested;
    use std::ffi::OsStr;

    #[test]
    fn gpu_profiling_is_explicitly_opt_in() {
        assert!(!profiling_requested(None));
        assert!(!profiling_requested(Some(OsStr::new("0"))));
        assert!(!profiling_requested(Some(OsStr::new("false"))));
        assert!(profiling_requested(Some(OsStr::new("1"))));
        assert!(profiling_requested(Some(OsStr::new("TRUE"))));
    }
}
