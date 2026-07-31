//! Per-frame processing orchestrator (GPU-native).
//!
//! Coordinates: detect → recognize → swap → restore → mask → blend.
//! Everything runs on a persistent CudaSlice<f32> frame buffer — no CPU loops.

use cudarc::driver::CudaSlice;

use crate::config::parameters::{FaceSwapParams, RestorerAlignment, RestorerSize, SwapperModel};
use crate::gpu::ops::GpuOps;
use crate::math::affine;
use crate::models::manager::ModelManager;
use crate::pipeline::dfm::DfmContract;
use crate::pipeline::face_detector::FaceDetectorBackend;
use crate::pipeline::face_landmark::LandmarkModel;
use crate::pipeline::face_mask;
use crate::pipeline::face_recognizer::FaceRecognizer;
use crate::pipeline::face_swapper::FaceSwapper;
use crate::pipeline::ort_binding::run_bound_f32;
use crate::pipeline::workspace::GpuWorkspace;

/// Source face identity: pre-computed embedding for matching.
#[derive(Clone)]
pub struct SourceFace {
    /// Reference target identity. `None` is the explicit swap-all shortcut.
    pub target_embedding: Option<Vec<f32>>,
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
    dfm: Option<DfmContract>,
) -> anyhow::Result<FrameResult> {
    if !params.enabled || (params.swapper_model == SwapperModel::Inswapper128 && sources.is_empty())
    {
        return Ok(FrameResult {
            faces_detected: 0,
            faces_swapped: 0,
        });
    }

    // Profiling: record start
    let _ = gpu.profile_mark(0);

    // 1. Detect faces (letterbox on GPU, ort inference, decode on CPU)
    let (faces, _det_scale) = if params.auto_rotation {
        detector.detect_gpu_auto_rotation(manager, gpu, frame_chw, ws, frame_h, frame_w)?
    } else {
        detector.detect_gpu(manager, gpu, frame_chw, ws, frame_h, frame_w)?
    };
    let _ = gpu.profile_mark(1); // after_detect

    let mut faces_swapped = 0;

    let dim = (params.dim as u32).max(1);
    let swap_size = dim * 128; // 128 or 256
    let pipeline_size = 512u32;

    for face in &faces {
        let refined = if params.landmark_enabled {
            LandmarkModel::from(params.landmark_mode).detect_gpu(
                manager,
                gpu,
                ws,
                frame_chw,
                frame_h,
                frame_w,
                face.bbox,
                &face.kps_5,
                params.landmark_from_points,
                params.landmark_score,
            )?
        } else {
            None
        };
        let refined_is_better = refined
            .as_ref()
            .is_some_and(|landmarks| landmarks.is_preferred_to(face.score));
        let effective_kps = refined
            .as_ref()
            .filter(|_| refined_is_better)
            .map_or(face.kps_5, |landmarks| landmarks.five);

        // 2a. Match against sources
        let source = if params.swapper_model == SwapperModel::Inswapper128 {
            Some(
                if sources.len() == 1 && sources[0].target_embedding.is_none() {
                    &sources[0]
                } else {
                    let embedding = FaceRecognizer::recognize_gpu(
                        manager,
                        gpu,
                        frame_chw,
                        frame_h,
                        frame_w,
                        &effective_kps,
                        ws,
                    )?;
                    match sources
                        .iter()
                        .find(|src| assignment_matches(src, &embedding))
                    {
                        Some(source) => source,
                        None => continue,
                    }
                },
            )
        } else {
            None
        };
        let _ = gpu.profile_mark(2); // after_recognize

        let effective_kps = adjusted_keypoints(effective_kps, params);

        // 2b. Latent from source embedding
        // 2c. Match Crosswap exactly: align once in canonical 512-space, then
        // resize that crop for the requested Inswapper dimension. Re-estimating
        // a separate transform at every resolution creates sub-pixel drift
        // between the generated face and the paste-back transform.
        let face_template = scaled_arcface_template(pipeline_size);
        let face_affine = affine::estimate_face_affine(&effective_kps, &face_template);
        let aligned_landmarks = std::array::from_fn(|index| {
            let x = effective_kps[index][0] as f64;
            let y = effective_kps[index][1] as f64;
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
        let _ = gpu.profile_mark(3); // after_warp_swap

        match params.swapper_model {
            SwapperModel::Inswapper128 => {
                let (iterations, fractional_blend) = strength_plan(params.strength);
                if iterations == 0 {
                    gpu.stream
                        .memcpy_dtod(&ws.face_512_original, &mut ws.face_512)?;
                } else {
                    gpu.resize_npp(
                        &ws.face_512,
                        &mut ws.face_256,
                        pipeline_size,
                        pipeline_size,
                        swap_size,
                        swap_size,
                    )?;
                    if params.geometry.enabled && params.geometry.face_scale != 0.0 {
                        let scale = 1.0 + f64::from(params.geometry.face_scale) / 100.0;
                        let center = f64::from(swap_size) / 2.0;
                        let transform = [
                            [scale, 0.0, center * (1.0 - scale)],
                            [0.0, scale, center * (1.0 - scale)],
                        ];
                        gpu.warp_affine_npp(
                            &ws.face_256,
                            &mut ws.face_512_scratch,
                            swap_size,
                            swap_size,
                            swap_size,
                            swap_size,
                            &transform,
                        )?;
                        gpu.stream
                            .memcpy_dtod(&ws.face_512_scratch, &mut ws.face_256)?;
                    }
                    for iteration in 0..iterations {
                        gpu.stream
                            .memcpy_dtod(&ws.face_256, &mut ws.face_512_scratch)?;
                        FaceSwapper::swap_gpu_cached(
                            manager,
                            gpu,
                            ws,
                            &source.expect("Inswapper source selected above").latent,
                            dim,
                            iteration == 0,
                        )?;
                    }
                    if fractional_blend < 1.0 {
                        gpu.scalar_blend_inplace(
                            &ws.face_512_scratch,
                            &mut ws.face_256,
                            3 * swap_size as usize * swap_size as usize,
                            1.0 - fractional_blend,
                        )?;
                    }
                    gpu.resize_npp(
                        &ws.face_256,
                        &mut ws.face_512,
                        swap_size,
                        swap_size,
                        pipeline_size,
                        pipeline_size,
                    )?;
                }
            }
            SwapperModel::Dfm => {
                let (iterations, fractional_blend) = strength_plan(params.strength);
                if iterations == 0 {
                    gpu.stream
                        .memcpy_dtod(&ws.face_512_original, &mut ws.face_512)?;
                } else {
                    dfm.ok_or_else(|| anyhow::anyhow!("DFM contract is not loaded"))?
                        .convert_gpu(manager, gpu, ws, params.dfm_morph, params.dfm_rct)?;
                    if fractional_blend < 1.0 {
                        gpu.scalar_blend_inplace(
                            &ws.face_512_original,
                            &mut ws.face_512,
                            3 * pipeline_size as usize * pipeline_size as usize,
                            1.0 - fractional_blend,
                        )?;
                    }
                }
            }
        }
        let _ = gpu.profile_mark(4); // after_swap_ort
        gpu.stream
            .memcpy_dtod(&ws.face_512, &mut ws.face_512_pre_restorer)?;

        // 2f. Optional GPEN restoration. The original swapped face remains in
        // face_512; inference input/output and alpha blending stay on GPU.
        if params.restorer_enabled {
            let (session_name, size) = restorer_contract(params.restorer_size)?;
            if let Some(alignment) = restorer_alignment_plan(
                gpu,
                manager,
                ws,
                params.restorer_alignment,
                params.detector_score,
            )? {
                apply_restorer_gpu(
                    gpu,
                    manager,
                    ws,
                    session_name,
                    size,
                    params.restorer_alpha,
                    alignment,
                )?;
            }
        }
        if params.restorer2_enabled {
            let (_, size) = restorer_contract(params.restorer2_size)?;
            let session_name = match params.restorer2_size {
                RestorerSize::Gpen256 => "GPENBFR256_2",
                RestorerSize::Gpen512 => "GPENBFR512_2",
                RestorerSize::Gpen1024 => unreachable!("rejected by restorer_contract"),
            };
            if let Some(alignment) = restorer_alignment_plan(
                gpu,
                manager,
                ws,
                params.restorer2_alignment,
                params.detector_score,
            )? {
                apply_restorer_gpu(
                    gpu,
                    manager,
                    ws,
                    session_name,
                    size,
                    params.restorer2_alpha,
                    alignment,
                )?;
            }
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
        crate::pipeline::color::apply_auto_color_gpu(gpu, ws, params, learned_mask)?;
        crate::pipeline::color::apply_color_adjust_gpu(gpu, ws, params)?;
        crate::pipeline::color::apply_jpeg_compression(gpu, ws, params)?;
        crate::pipeline::color::apply_final_blur_gpu(gpu, ws, params)?;
        face_mask::gpu_generate_mask_512(
            gpu,
            ws,
            params.border_top.min(100),
            params.border_bottom.min(100),
            params.border_left.min(100),
            params.border_right.min(100),
            blur_ks,
            blur_sigma,
            params.restorer_enabled || params.restorer2_enabled,
            learned_mask,
            params.overall_mask_blur,
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

fn assignment_matches(assignment: &SourceFace, detected_embedding: &[f32]) -> bool {
    assignment.target_embedding.as_ref().is_none_or(|target| {
        let cosine = FaceRecognizer::cosine_similarity(detected_embedding, target);
        crosswap_similarity(cosine) >= assignment.threshold
    })
}

fn crosswap_similarity(cosine: f32) -> f32 {
    (1.0 + cosine) * 0.5
}

fn adjusted_keypoints(mut keypoints: [[f32; 2]; 5], params: &FaceSwapParams) -> [[f32; 2]; 5] {
    if params.geometry.enabled {
        let scale = 1.0 + params.geometry.keypoints_scale / 100.0;
        for point in &mut keypoints {
            point[0] = (point[0] + params.geometry.keypoints_x - 255.0) * scale + 255.0;
            point[1] = (point[1] + params.geometry.keypoints_y - 255.0) * scale + 255.0;
        }
    }
    if params.geometry.landmark_offsets_enabled {
        for (point, offset) in keypoints.iter_mut().zip(params.geometry.landmark_offsets) {
            point[0] += offset[0];
            point[1] += offset[1];
        }
    }
    keypoints
}

#[derive(Clone, Copy)]
enum RestorerAlignmentPlan {
    Original,
    Aligned([[f64; 3]; 2]),
}

fn blend_restorer_affine() -> [[f64; 3]; 2] {
    affine::estimate_face_affine(
        &scaled_arcface_template(512),
        &crate::math::constants::FFHQ_KPS,
    )
}

fn restorer_alignment_plan(
    gpu: &GpuOps,
    manager: &mut ModelManager,
    ws: &mut GpuWorkspace,
    alignment: RestorerAlignment,
    score_threshold: f32,
) -> anyhow::Result<Option<RestorerAlignmentPlan>> {
    Ok(match alignment {
        RestorerAlignment::Original => Some(RestorerAlignmentPlan::Original),
        RestorerAlignment::Blend => Some(RestorerAlignmentPlan::Aligned(blend_restorer_affine())),
        RestorerAlignment::Reference => LandmarkModel::Points5
            .detect_restorer_reference_gpu(manager, gpu, ws, score_threshold)?
            .map(|result| {
                RestorerAlignmentPlan::Aligned(affine::estimate_face_affine(
                    &result.five,
                    &crate::math::constants::FFHQ_KPS,
                ))
            }),
    })
}

fn apply_restorer_gpu(
    gpu: &GpuOps,
    manager: &mut ModelManager,
    ws: &mut GpuWorkspace,
    session_name: &str,
    size: usize,
    alpha: f32,
    alignment: RestorerAlignmentPlan,
) -> anyhow::Result<()> {
    if let RestorerAlignmentPlan::Aligned(matrix) = alignment {
        gpu.warp_affine_npp(
            &ws.face_512,
            &mut ws.face_512_scratch,
            512,
            512,
            512,
            512,
            &matrix,
        )?;
    }

    if size == 256 {
        let source = match alignment {
            RestorerAlignmentPlan::Original => &ws.face_512,
            RestorerAlignmentPlan::Aligned(_) => &ws.face_512_scratch,
        };
        gpu.resize_npp(source, &mut ws.restorer_256_input, 512, 512, 256, 256)?;
        gpu.affine_scale(&mut ws.restorer_256_input, 1.0 / 127.5, -1.0)?;
        run_bound_f32(
            manager,
            &gpu.stream,
            session_name,
            "input",
            &ws.restorer_256_input,
            &[1, 3, 256, 256],
            "output",
            &mut ws.restorer_256_output,
            &[1, 3, 256, 256],
        )?;
        gpu.affine_scale(&mut ws.restorer_256_output, 127.5, 127.5)?;
        gpu.resize_npp(
            &ws.restorer_256_output,
            &mut ws.restorer_cache,
            256,
            256,
            512,
            512,
        )?;
    } else {
        if matches!(alignment, RestorerAlignmentPlan::Original) {
            gpu.stream
                .memcpy_dtod(&ws.face_512, &mut ws.face_512_scratch)?;
        }
        gpu.affine_scale(&mut ws.face_512_scratch, 1.0 / 127.5, -1.0)?;
        run_bound_f32(
            manager,
            &gpu.stream,
            session_name,
            "input",
            &ws.face_512_scratch,
            &[1, 3, size as i64, size as i64],
            "output",
            &mut ws.restorer_cache,
            &[1, 3, size as i64, size as i64],
        )?;
        gpu.affine_scale(&mut ws.restorer_cache, 127.5, 127.5)?;
    }

    match alignment {
        RestorerAlignmentPlan::Original => {
            gpu.scalar_blend_inplace(&ws.restorer_cache, &mut ws.face_512, 3 * 512 * 512, alpha)?
        }
        RestorerAlignmentPlan::Aligned(matrix) => {
            gpu.warp_affine_npp(
                &ws.restorer_cache,
                &mut ws.face_512_scratch,
                512,
                512,
                512,
                512,
                &affine::invert_2x3(&matrix),
            )?;
            gpu.scalar_blend_inplace(&ws.face_512_scratch, &mut ws.face_512, 3 * 512 * 512, alpha)?;
        }
    }
    Ok(())
}

fn restorer_contract(size: RestorerSize) -> anyhow::Result<(&'static str, usize)> {
    match size {
        RestorerSize::Gpen256 => Ok(("GPENBFR256", 256)),
        RestorerSize::Gpen512 => Ok(("GPENBFR512", 512)),
        RestorerSize::Gpen1024 => anyhow::bail!("GPEN-1024 is excluded from the runtime"),
    }
}

fn strength_plan(amount: f32) -> (u32, f32) {
    let amount = amount.clamp(0.0, 5.0);
    if amount == 0.0 {
        return (0, 0.0);
    }
    let iterations = amount.ceil() as u32;
    let fraction = amount.fract();
    (iterations, if fraction == 0.0 { 1.0 } else { fraction })
}

/// Compute face alignment template at target size.
/// Match Crosswap's actual `get_arcface_template(..., mode="arcface128")`
/// output: scale both coordinates, then shift the X coordinate of all points.
fn scaled_arcface_template(target_size: u32) -> [[f32; 2]; 5] {
    use crate::math::constants::ARCFACE_DST;
    let factor = target_size as f32 / 128.0;
    let mut dst = ARCFACE_DST;
    for pt in dst.iter_mut() {
        pt[0] = pt[0] * factor + factor * 8.0;
        pt[1] *= factor;
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::{
        SourceFace, adjusted_keypoints, assignment_matches, blend_restorer_affine,
        crosswap_similarity, restorer_contract, scaled_arcface_template, strength_plan,
    };
    use crate::config::parameters::FaceSwapParams;
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

    #[test]
    fn blend_restorer_alignment_matches_crossswap_skimage_oracle() {
        let actual = blend_restorer_affine();
        let expected = [
            [0.850048963, -0.000129492873, 38.9094548],
            [0.000129492873, 0.850048963, 62.8344332],
        ];
        for row in 0..2 {
            for column in 0..3 {
                assert!(
                    (actual[row][column] - expected[row][column]).abs() < 5e-5,
                    "matrix[{row}][{column}]={} expected {}",
                    actual[row][column],
                    expected[row][column]
                );
            }
        }
    }

    #[test]
    fn arcface128_template_offsets_every_x_coordinate() {
        let template = scaled_arcface_template(128);
        assert_eq!(template[0], [46.2946, 51.6963]);
        assert_eq!(template[1], [81.5318, 51.5014]);
        assert_eq!(template[4], [78.7299, 92.2041]);
    }

    #[test]
    fn strength_plan_matches_crossswap_iteration_and_fractional_blend() {
        assert_eq!(strength_plan(0.0), (0, 0.0));
        assert_eq!(strength_plan(1.0), (1, 1.0));
        assert_eq!(strength_plan(1.25), (2, 0.25));
        assert_eq!(strength_plan(2.0), (2, 1.0));
        assert_eq!(strength_plan(5.0), (5, 1.0));
    }

    #[test]
    fn similarity_uses_crossswap_score_scale() {
        assert_eq!(crosswap_similarity(1.0), 1.0);
        assert_eq!(crosswap_similarity(0.0), 0.5);
        assert_eq!(crosswap_similarity(-1.0), 0.0);
    }

    #[test]
    fn target_assignment_is_separate_from_source_latent() {
        let mut target = vec![0.0; 512];
        target[0] = 1.0;
        let mut same = vec![0.0; 512];
        same[0] = 1.0;
        let mut different = vec![0.0; 512];
        different[1] = 1.0;
        let scoped = SourceFace {
            target_embedding: Some(target),
            latent: vec![9.0, 8.0],
            threshold: 0.75,
        };
        assert!(assignment_matches(&scoped, &same));
        assert!(!assignment_matches(&scoped, &different));

        let swap_all = SourceFace {
            target_embedding: None,
            latent: vec![9.0, 8.0],
            threshold: 1.0,
        };
        assert!(assignment_matches(&swap_all, &different));
    }

    #[test]
    fn keypoint_adjustments_follow_crossswap_order_and_fixed_center() {
        let mut params = FaceSwapParams::default();
        params.geometry.enabled = true;
        params.geometry.keypoints_x = 10.0;
        params.geometry.keypoints_y = -5.0;
        params.geometry.keypoints_scale = 10.0;
        params.geometry.landmark_offsets_enabled = true;
        params.geometry.landmark_offsets[0] = [3.0, -2.0];
        let input = [[100.0, 200.0]; 5];
        let output = adjusted_keypoints(input, &params);
        assert!((output[0][0] - 98.5).abs() < 1e-5);
        assert!((output[0][1] - 187.0).abs() < 1e-5);
        assert!((output[1][0] - 95.5).abs() < 1e-5);
        assert!((output[1][1] - 189.0).abs() < 1e-5);
    }
}
