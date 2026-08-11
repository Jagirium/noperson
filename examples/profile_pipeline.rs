//! Profiling: run the full pipeline on face.jpg for N frames, report per-stage timings.
//!
//! Usage:
//!   cargo run --example profile_pipeline -- source.jpg target.jpg [frames=60] [dim=1] [restore=0]

use std::sync::Arc;
use std::time::Instant;

use image::GenericImageView;
use noperson::backend::ComputeContext;

use noperson::backend::ComputeOps;
use noperson::config::parameters::{FaceSwapParams, RestorerMode, SwapDim};
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
    let n_frames: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let dim = match std::env::args().nth(4).as_deref() {
        Some("2") => SwapDim::Dim2,
        Some("3") => SwapDim::Dim3,
        Some("4") => SwapDim::Dim4,
        _ => SwapDim::Dim1,
    };
    let restore = matches!(
        std::env::args().nth(5).as_deref(),
        Some("1" | "true" | "restore")
    );
    let restorer_mode = if matches!(std::env::args().nth(6).as_deref(), Some("quality")) {
        RestorerMode::Quality
    } else {
        RestorerMode::Realtime
    };

    // Init CUDA
    let ctx = Arc::new(ComputeContext::new(0)?);
    let stream = ctx.new_stream()?;
    let gpu = ComputeOps::new(&ctx, stream.clone())?;
    let mut manager = ModelManager::new("models");
    manager.set_compute_stream(stream.cu_stream() as *mut ())?;
    manager.load("YoloFace8n", "yoloface_8n.onnx")?;
    manager.load("Inswapper128ArcFace", "w600k_r50.onnx")?;
    manager.load("Inswapper128", CANONICAL_SWAPPER_FILENAME)?;
    manager.load_emap(CANONICAL_SWAPPER_FILENAME)?;
    if restore {
        manager.load("GPENBFR512", "GPEN-BFR-512.onnx")?;
    }
    eprintln!("Models loaded");

    let mut ws = GpuWorkspace::new(&stream)?;
    let mut face_tracker = TemporalFaceTracker::new(TrackerPolicy::offline_recovery());

    // Load target face → embedding → latent
    let target_img = image::open(&target_path)?;
    let (tw, th) = target_img.dimensions();
    let target_rgb = target_img.to_rgb8();
    let d_target_u8 = gpu.upload_u8(target_rgb.as_raw())?;
    let mut d_target_chw = gpu.alloc_zeros(3 * tw as usize * th as usize)?;
    gpu.hwc_u8_to_chw_f32(&d_target_u8, &mut d_target_chw, th, tw)?;

    let detector = YoloFaceDetector::new(0.5);
    let (target_faces, _) =
        detector.detect_gpu(&mut manager, &gpu, &d_target_chw, &mut ws, th, tw)?;
    anyhow::ensure!(!target_faces.is_empty(), "No face in target");
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
    let sources = vec![SourceFace {
        target_embedding: None,
        backend: AssignmentBackend::Inswapper { latent },
        threshold: 0.0,
        params: None,
    }];

    // Load source image
    let source_img = image::open(&source_path)?;
    let (sw, sh) = source_img.dimensions();
    let source_rgb = source_img.to_rgb8();
    let d_source_u8 = gpu.upload_u8(source_rgb.as_raw())?;
    let mut d_source_chw = gpu.alloc_zeros(3 * sw as usize * sh as usize)?;
    gpu.hwc_u8_to_chw_f32(&d_source_u8, &mut d_source_chw, sh, sw)?;

    let params = FaceSwapParams {
        enabled: true,
        dim,
        restorer_enabled: restore,
        restorer_mode,
        ..Default::default()
    };

    // Three passes capture both TensorRT graphs and settle CUDA allocators.
    eprintln!("Warmup (3 frames)...");
    for _ in 0..3 {
        gpu.hwc_u8_to_chw_f32(&d_source_u8, &mut d_source_chw, sh, sw)?;
        process_frame_gpu(
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
    }
    gpu.sync()?;

    // Re-upload the frame (process_frame_gpu modifies it in-place)
    gpu.hwc_u8_to_chw_f32(&d_source_u8, &mut d_source_chw, sh, sw)?;

    // Benchmark
    eprintln!("Benchmarking {n_frames} frames...");
    let wall_start = Instant::now();
    for i in 0..n_frames {
        // Re-upload the original frame each iteration (process_frame_gpu mutates it)
        gpu.hwc_u8_to_chw_f32(&d_source_u8, &mut d_source_chw, sh, sw)?;
        process_frame_gpu(
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
        if (i + 1) % 10 == 0 {
            eprintln!("  {}/{} frames", i + 1, n_frames);
        }
    }
    gpu.sync()?;
    let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    let per_frame_ms = wall_ms / n_frames as f64;
    let fps = 1000.0 / per_frame_ms;

    eprintln!();
    eprintln!("══════════════════════════════════════════════");
    eprintln!("  Resolution: {sw}x{sh}");
    eprintln!("  Frames:     {n_frames}");
    eprintln!("  Total:      {wall_ms:.1} ms");
    eprintln!("  Per frame:  {per_frame_ms:.2} ms");
    eprintln!("  FPS:        {fps:.1}");
    eprintln!("══════════════════════════════════════════════");

    // The per-stage profile_tick logs every 30 frames — set report_every to 1
    // for a single-frame breakdown, but the wall-clock above is the real number.
    // For per-stage breakdown on the last frame, we force a manual sync.
    eprintln!();
    eprintln!("Per-stage breakdown (from CUDA events, logged every 30 frames):");
    eprintln!("Run with n_frames >= 30 to see the [profile] log line from gpu.profile_tick().");

    Ok(())
}
