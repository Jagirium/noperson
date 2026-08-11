//! Example: full face swap pipeline with paste-back.
//!
//! Usage:
//!   cargo run --example swap_to_vcam -- source.jpg target.jpg output.png
//!   cargo run --example swap_to_vcam -- source.jpg target.jpg output.png --vcam
//!
//! Pipeline: detect → recognize → swap → mask → paste-back → /dev/video10

use std::sync::Arc;

use image::GenericImageView;
use noperson::backend::ComputeContext;

use noperson::backend::ComputeOps;
use noperson::config::parameters::{FaceSwapParams, SwapDim};
use noperson::io::VirtualCamera;
use noperson::models::live_catalog::CANONICAL_SWAPPER_FILENAME;
use noperson::models::manager::ModelManager;
use noperson::pipeline::face_detector::YoloFaceDetector;
use noperson::pipeline::face_recognizer::FaceRecognizer;
use noperson::pipeline::face_tracker::{TemporalFaceTracker, TrackerPolicy};
use noperson::pipeline::frame_processor::{AssignmentBackend, SourceFace, process_frame_gpu};
use noperson::pipeline::workspace::GpuWorkspace;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let source_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/photos/face.jpg".to_string());
    let target_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "assets/photos/face.jpg".to_string());
    let output_path = std::env::args()
        .nth(3)
        .filter(|argument| argument != "--vcam")
        .unwrap_or_else(|| "swapped_output.png".to_string());
    let publish_to_vcam = std::env::args().any(|argument| argument == "--vcam");

    // Init CUDA
    let ctx = Arc::new(ComputeContext::new(0)?);
    let stream = ctx.new_stream()?;
    let gpu = ComputeOps::new(&ctx, stream.clone())?;
    let mut manager = ModelManager::new("models");
    manager.set_compute_stream(stream.cu_stream() as *mut ())?;

    // Load models
    manager.load("YoloFace8n", "yoloface_8n.onnx")?;
    manager.load("Inswapper128ArcFace", "w600k_r50.onnx")?;
    manager.load("Inswapper128", CANONICAL_SWAPPER_FILENAME)?;
    manager.load_emap(CANONICAL_SWAPPER_FILENAME)?;
    tracing::info!("Models loaded");

    let mut ws = GpuWorkspace::new(&stream)?;
    let mut face_tracker = TemporalFaceTracker::new(TrackerPolicy::offline_recovery());

    // ── Load target face → extract embedding → compute latent ──
    let target_img = image::open(&target_path)?;
    let (tw, th) = target_img.dimensions();
    let target_rgb = target_img.to_rgb8();
    let d_target_u8 = gpu.upload_u8(target_rgb.as_raw())?;
    let mut d_target_chw = gpu.alloc_zeros(3 * tw as usize * th as usize)?;
    gpu.hwc_u8_to_chw_f32(&d_target_u8, &mut d_target_chw, th, tw)?;

    let detector = YoloFaceDetector::new(0.5);
    let (target_faces, _) =
        detector.detect_gpu(&mut manager, &gpu, &d_target_chw, &mut ws, th, tw)?;
    anyhow::ensure!(!target_faces.is_empty(), "No face in target image");
    let target_kps = target_faces[0].kps_5;

    let target_emb = FaceRecognizer::recognize_gpu(
        &mut manager,
        &gpu,
        &d_target_chw,
        th,
        tw,
        &target_kps,
        &mut ws,
    )?;
    let emap = manager.emap.as_ref().unwrap();
    let latent = FaceRecognizer::calc_latent(&target_emb, emap);
    tracing::info!("Target face: embedding + latent computed");

    let sources = vec![SourceFace {
        target_embedding: None,
        backend: AssignmentBackend::Inswapper { latent },
        threshold: 0.0,
        params: None,
    }];

    // ── Load source image ──
    let source_img = image::open(&source_path)?;
    let (sw, sh) = source_img.dimensions();
    let source_rgb = source_img.to_rgb8();
    let d_source_u8 = gpu.upload_u8(source_rgb.as_raw())?;
    let mut d_source_chw = gpu.alloc_zeros(3 * sw as usize * sh as usize)?;
    gpu.hwc_u8_to_chw_f32(&d_source_u8, &mut d_source_chw, sh, sw)?;

    // ── Run full pipeline: detect → recognize → swap → paste-back ──
    let params = FaceSwapParams {
        enabled: true,
        dim: SwapDim::Dim1,
        ..Default::default()
    };

    let result = process_frame_gpu(
        &gpu,
        &mut manager,
        &detector,
        &mut d_source_chw,
        sh,
        sw,
        &mut ws,
        &sources,
        &params,
        &mut face_tracker,
    )?;
    tracing::info!(
        "Pipeline done: {} detected, {} swapped",
        result.faces_detected,
        result.faces_swapped
    );

    // ── Download result frame ──
    let mut frame_out = vec![0.0f32; 3 * sw as usize * sh as usize];
    gpu.download_into(&d_source_chw, &mut frame_out)?;

    // ── Convert CHW f32 → HWC u8 ──
    let mut output_hwc = vec![0u8; (sw * sh * 3) as usize];
    for c in 0..3usize {
        for y in 0..sh as usize {
            for x in 0..sw as usize {
                let chw_idx = c * sw as usize * sh as usize + y * sw as usize + x;
                let hwc_idx = (y * sw as usize + x) * 3 + c;
                output_hwc[hwc_idx] = frame_out[chw_idx].clamp(0.0, 255.0) as u8;
            }
        }
    }

    // Save to file
    image::save_buffer(&output_path, &output_hwc, sw, sh, image::ColorType::Rgb8)?;
    tracing::info!("Saved {output_path}");

    if publish_to_vcam {
        let (vcam_w, vcam_h) = (sw.max(640), sh.max(480));
        let mut vcam = VirtualCamera::open(10, vcam_w, vcam_h, 30)?;
        tracing::info!("Opened {}", vcam.device_path());
        for _ in 0..60 {
            vcam.send_frame(&output_hwc)?;
        }
        tracing::info!("Sent 60 frames to {}", vcam.device_path());
    }

    Ok(())
}
