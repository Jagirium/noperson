//! Integration test: verify model loading and basic detection.
//!
//! This test requires:
//!   - CUDA device
//!   - models/yoloface_8n.onnx
//!   - assets/photos/face.jpg
//!
//! Run: cargo test --test pipeline_smoke -- --nocapture --ignored

use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use image::GenericImageView;

use noperson::gpu::ops::GpuOps;
use noperson::models::manager::ModelManager;
use noperson::models::registry::MODELS;
use noperson::pipeline::face_detector::YoloFaceDetector;
use noperson::pipeline::workspace::GpuWorkspace;

/// Verify that model files exist in models/.
#[test]
fn test_models_present() {
    let models_dir = PathBuf::from("models");
    for entry in MODELS {
        let path = models_dir.join(entry.filename);
        if !path.exists() {
            eprintln!("Missing model: {} ({})", entry.name, path.display());
        }
    }
}

/// Verify that the face test image exists.
#[test]
fn test_face_image_present() {
    let face = PathBuf::from("assets/photos/face.jpg");
    assert!(face.exists(), "assets/photos/face.jpg not found");
}

/// Smoke test: load YoloFace8n model and run detection on the face fixture.
///
/// Requires CUDA + models/yoloface_8n.onnx + assets/photos/face.jpg.
/// Marked `#[ignore]` — run with `cargo test -- --ignored`.
#[test]
#[ignore = "requires CUDA + ONNX models"]
fn test_detect_faces_smoke() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    // Load test image
    let img = image::open("assets/photos/face.jpg")?;
    let (width, height) = img.dimensions();
    let rgb = img.to_rgb8();
    eprintln!("Loaded face.jpg: {width}x{height}");

    // Init CUDA
    let ctx = Arc::new(CudaContext::new(0)?);
    let stream = ctx.default_stream().clone();
    let gpu = GpuOps::new(&ctx, stream.clone())?;
    eprintln!("CUDA initialized");

    // Init model manager
    let mut manager = ModelManager::new("models");
    manager.set_compute_stream(stream.cu_stream() as *mut ());

    // Load detector
    manager.load("YoloFace8n", "yoloface_8n.onnx")?;
    eprintln!("YoloFace8n loaded");

    let detector = YoloFaceDetector::new(0.5);

    // Upload frame
    let d_frame_u8 = gpu.upload_u8(rgb.as_raw())?;
    let mut d_frame_chw = gpu.alloc_zeros(3 * width as usize * height as usize)?;
    gpu.hwc_u8_to_chw_f32(&d_frame_u8, &mut d_frame_chw, height, width)?;

    // Detect
    let mut ws = GpuWorkspace::new(&stream)?;
    let (faces, _scale) =
        detector.detect_gpu(&mut manager, &gpu, &d_frame_chw, &mut ws, height, width)?;

    eprintln!("Detected {} faces", faces.len());
    assert!(!faces.is_empty(), "Should detect at least 1 face");

    for (i, face) in faces.iter().enumerate() {
        eprintln!("  Face {i}: bbox={:?} score={:.3}", face.bbox, face.score);
    }

    Ok(())
}
