//! Per-frame processing orchestrator (GPU-native).
//!
//! Coordinates: detect → recognize → swap → restore → mask → blend.
//! Everything runs on a persistent CudaSlice<f32> frame buffer — no CPU loops.

use cudarc::driver::CudaSlice;

use crate::config::parameters::{FaceSwapParams, RestorerMode, RestorerSize};
use crate::gpu::ops::GpuOps;
use crate::gpu::unified;
use crate::math::affine;
use crate::models::manager::ModelManager;
use crate::pipeline::face_detector::FaceDetectorBackend;
use crate::pipeline::face_mask;
use crate::pipeline::face_recognizer::FaceRecognizer;
use crate::pipeline::face_swapper::FaceSwapper;
use crate::pipeline::workspace::GpuWorkspace;

/// Source face identity: pre-computed embedding for matching.
#[derive(Clone)]
pub struct SourceFace {
    pub embedding: Vec<f32>,
    /// Inswapper latent is invariant for an identity. Compute it once when the
    /// source is loaded instead of doing a 512x512 CPU matmul for every face.
    pub latent: Vec<f32>,
    pub threshold: f32,
}

/// Result of processing a single frame.
pub struct FrameResult {
    pub faces_detected: usize,
    pub faces_swapped: usize,
}

/// Process a single frame fully on the GPU.
///
/// `frame_chw` holds the current frame [3, H, W] in [0, 255] and is mutated
/// in place to contain the swapped frame.
pub fn process_frame_gpu<D: FaceDetectorBackend + ?Sized>(
    gpu: &GpuOps,
    manager: &mut ModelManager,
    detector: &D,
    frame_chw: &mut CudaSlice<f32>,
    frame_h: u32,
    frame_w: u32,
    ws: &mut GpuWorkspace,
    sources: &[SourceFace],
    params: &FaceSwapParams,
) -> anyhow::Result<FrameResult> {
    if !params.enabled || sources.is_empty() {
        return Ok(FrameResult {
            faces_detected: 0,
            faces_swapped: 0,
        });
    }

    // Profiling: record start
    let _ = gpu.profile_mark(0);

    // 1. Detect faces (letterbox on GPU, ort inference, decode on CPU)
    let (faces, _det_scale) = detector.detect_gpu(manager, gpu, frame_chw, ws, frame_h, frame_w)?;
    let _ = gpu.profile_mark(1); // after_detect

    let mut faces_swapped = 0;

    let dim = (params.dim as u32).max(1);
    let swap_size = dim * 128; // 128 or 256
    let pipeline_size = 512u32;

    for face in &faces {
        // 2a. Match against sources
        let source = if sources.len() == 1 && sources[0].threshold <= 0.0 {
            &sources[0]
        } else {
            let embedding = FaceRecognizer::recognize_gpu(
                manager,
                gpu,
                frame_chw,
                frame_h,
                frame_w,
                &face.kps_5,
                ws,
            )?;
            match sources.iter().find(|src| {
                FaceRecognizer::cosine_similarity(&embedding, &src.embedding) >= src.threshold
            }) {
                Some(s) => s,
                None => continue,
            }
        };
        let _ = gpu.profile_mark(2); // after_recognize

        // 2b. Latent from source embedding
        // 2c. Match Crosswap exactly: align once in canonical 512-space, then
        // resize that crop for the requested Inswapper dimension. Re-estimating
        // a separate transform at every resolution creates sub-pixel drift
        // between the generated face and the paste-back transform.
        let face_template = scaled_arcface_template(pipeline_size);
        let face_affine = affine::estimate_face_affine(&face.kps_5, &face_template);
        let aligned_landmarks = std::array::from_fn(|index| {
            let x = face.kps_5[index][0] as f64;
            let y = face.kps_5[index][1] as f64;
            [
                face_affine[0][0] * x + face_affine[0][1] * y + face_affine[0][2],
                face_affine[1][0] * x + face_affine[1][1] * y + face_affine[1][2],
            ]
        });

        gpu.warp_affine_npp(
            frame_chw,
            &mut ws.face_512,
            frame_h,
            frame_w,
            pipeline_size,
            pipeline_size,
            &face_affine,
        )?;
        gpu.stream
            .memcpy_dtod(&ws.face_512, &mut ws.face_512_original)?;
        gpu.resize_npp(
            &ws.face_512,
            &mut ws.face_256,
            pipeline_size,
            pipeline_size,
            swap_size,
            swap_size,
        )?;
        let _ = gpu.profile_mark(3); // after_warp_swap

        // 2d. Swap on GPU: face_256 → swap_batch_in → ort → scatter back
        FaceSwapper::swap_gpu(manager, gpu, ws, &source.latent, dim)?;
        let _ = gpu.profile_mark(4); // after_swap_ort

        // 2e. Resize swapped face to pipeline_size (512) via NPP
        gpu.resize_npp(
            &ws.face_256,
            &mut ws.face_512,
            swap_size,
            swap_size,
            pipeline_size,
            pipeline_size,
        )?;
        gpu.stream
            .memcpy_dtod(&ws.face_512, &mut ws.face_512_pre_restorer)?;

        // 2f. Optional GPEN restoration. The original swapped face remains in
        // face_512; inference input/output and alpha blending stay on GPU.
        if params.restorer_enabled {
            let values = 3 * pipeline_size as usize * pipeline_size as usize;
            let (restorer_name, restorer_size) = restorer_contract(params.restorer_size)?;
            // Refresh the temporally stable GPEN-512 output every second frame
            // and reuse the device-resident result in between.
            // A single cache is safe only for the common one-face live path.
            // With multiple faces refresh each aligned crop to avoid leaking
            // one person's expression/pose into the next face.
            let refresh = params.restorer_mode == RestorerMode::Quality
                || faces.len() != 1
                || !ws.restorer_cache_valid
                || ws.restorer_frame.is_multiple_of(2);
            if refresh {
                let session = manager
                    .get_mut(restorer_name)
                    .ok_or_else(|| anyhow::anyhow!("{restorer_name} is not loaded"))?;
                if restorer_size == 256 {
                    gpu.resize_npp(
                        &ws.face_512,
                        &mut ws.restorer_256_input,
                        pipeline_size,
                        pipeline_size,
                        256,
                        256,
                    )?;
                    gpu.affine_scale(&mut ws.restorer_256_input, 1.0 / 127.5, -1.0)?;
                    unified::run_gpen(
                        session,
                        &gpu.stream,
                        &mut ws.restorer_256_input,
                        &mut ws.restorer_256_output,
                        256,
                    )?;
                    gpu.affine_scale(&mut ws.restorer_256_output, 127.5, 127.5)?;
                    gpu.resize_npp(
                        &ws.restorer_256_output,
                        &mut ws.restorer_cache,
                        256,
                        256,
                        pipeline_size,
                        pipeline_size,
                    )?;
                } else {
                    gpu.stream
                        .memcpy_dtod(&ws.face_512, &mut ws.face_512_scratch)?;
                    // GPEN expects [-1, 1] and returns [-1, 1].
                    gpu.affine_scale(&mut ws.face_512_scratch, 1.0 / 127.5, -1.0)?;
                    unified::run_gpen(
                        session,
                        &gpu.stream,
                        &mut ws.face_512_scratch,
                        &mut ws.restorer_cache,
                        restorer_size,
                    )?;
                    gpu.affine_scale(&mut ws.restorer_cache, 127.5, 127.5)?;
                }
                ws.restorer_cache_valid = true;
            }
            gpu.scalar_blend_inplace(
                &ws.restorer_cache,
                &mut ws.face_512,
                values,
                params.restorer_alpha,
            )?;
            ws.restorer_frame += 1;
        }
        let _ = gpu.profile_mark(5); // after_resize_512

        // 2g. Paste back with the exact inverse of the alignment transform.
        let paste_inv = affine::invert_2x3(&face_affine);

        // 2g. Generate blurred 512² mask on GPU
        let blur_ks = params.border_blur * 2 + 1;
        let blur_sigma = (params.border_blur as f32 + 1.0) * 0.2;
        let learned_mask = face_mask::gpu_generate_learned_mask_128(gpu, manager, ws, params)?;
        face_mask::gpu_apply_landmark_restore_mask(gpu, ws, params, &aligned_landmarks)?;
        face_mask::gpu_restore_semantic_regions(gpu, ws, params)?;
        face_mask::gpu_apply_fake_diff(gpu, ws, params)?;
        face_mask::gpu_generate_mask_512(
            gpu,
            ws,
            params.border_top.min(100),
            params.border_bottom.min(100),
            params.border_left.min(100),
            params.border_right.min(100),
            blur_ks,
            blur_sigma,
            params.restorer_enabled,
            learned_mask,
        )?;
        let _ = gpu.profile_mark(6); // after_mask_gen

        // 2h. Bbox in frame space via inverse affine (face corners → frame)
        let fs = pipeline_size as f64;
        let corners = [(0.0, 0.0), (fs, 0.0), (0.0, fs), (fs, fs)];
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        for (cx, cy) in corners {
            let fx = paste_inv[0][0] * cx + paste_inv[0][1] * cy + paste_inv[0][2];
            let fy = paste_inv[1][0] * cx + paste_inv[1][1] * cy + paste_inv[1][2];
            min_x = min_x.min(fx);
            min_y = min_y.min(fy);
            max_x = max_x.max(fx);
            max_y = max_y.max(fy);
        }
        let left = (min_x.floor() as i32).max(0) as u32;
        let top = (min_y.floor() as i32).max(0) as u32;
        let right = (max_x.ceil() as i32)
            .min(frame_w as i32)
            .max(left as i32 + 1) as u32;
        let bottom = (max_y.ceil() as i32)
            .min(frame_h as i32)
            .max(top as i32 + 1) as u32;

        // 2i. GPU paste-back: frame[bbox] = swap*mask + frame*(1-mask)
        gpu.paste_back(
            frame_chw,
            &ws.face_512,
            &ws.mask_512,
            frame_h,
            frame_w,
            pipeline_size,
            [left, top, right, bottom],
            &face_affine,
        )?;
        let _ = gpu.profile_mark(7); // after_paste_back
        gpu.profile_set_active(7);

        faces_swapped += 1;
    }

    // Tick profiler — every N frames this syncs the last event and logs stage times.
    let _ = gpu.profile_tick();

    // No sync() here — caller's final download will implicitly synchronize
    // the stream. Inserting barriers after every face is the fastest way
    // to kill GPU pipelining.
    Ok(FrameResult {
        faces_detected: faces.len(),
        faces_swapped,
    })
}

fn restorer_contract(size: RestorerSize) -> anyhow::Result<(&'static str, usize)> {
    match size {
        RestorerSize::Gpen256 => Ok(("GPENBFR256", 256)),
        RestorerSize::Gpen512 => Ok(("GPENBFR512", 512)),
        RestorerSize::Gpen1024 => anyhow::bail!("GPEN-1024 is excluded from the runtime"),
    }
}

/// Compute face alignment template at target size.
/// Match Crosswap's actual `get_arcface_template(..., mode="arcface128")`
/// output, including its `(1, 5, 2)` NumPy indexing behavior. `template[:, 0]`
/// selects both coordinates of the first landmark, so the offset is applied to
/// the first eye's X and Y rather than every landmark's X.
fn scaled_arcface_template(target_size: u32) -> [[f32; 2]; 5] {
    use crate::math::constants::ARCFACE_DST;
    let factor = target_size as f32 / 128.0;
    let mut dst = ARCFACE_DST;
    for pt in dst.iter_mut() {
        pt[0] *= factor;
        pt[1] *= factor;
    }
    let offset = factor * 8.0;
    dst[0][0] += offset;
    dst[0][1] += offset;
    dst
}

#[cfg(test)]
mod tests {
    use super::restorer_contract;
    use crate::config::parameters::RestorerSize;

    #[test]
    fn restorer_contract_maps_only_supported_runtime_sizes() {
        assert_eq!(
            restorer_contract(RestorerSize::Gpen256).unwrap(),
            ("GPENBFR256", 256)
        );
        assert_eq!(
            restorer_contract(RestorerSize::Gpen512).unwrap(),
            ("GPENBFR512", 512)
        );
        assert!(restorer_contract(RestorerSize::Gpen1024).is_err());
    }
}
