//! Example: detect faces in an image using YoloFace8n via ort + CUDA.
//!
//! Usage:
//!   cargo run --example detect_faces -- face.jpg
//!
//! Loads the YoloFace8n ONNX model, runs detection, prints results.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use image::GenericImageView;
use noperson::backend::ComputeContext;

use noperson::backend::ComputeOps;
use noperson::models::manager::ModelManager;
use noperson::pipeline::face_detector::YoloFaceDetector;
use noperson::pipeline::workspace::GpuWorkspace;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let img_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("assets/photos/face.jpg"));

    tracing::info!("Loading image: {}", img_path.display());
    let img = image::open(&img_path)
        .with_context(|| format!("Failed to open image: {}", img_path.display()))?;
    let (width, height) = img.dimensions();
    let rgb = img.to_rgb8();
    tracing::info!("Image: {}x{}", width, height);

    // Init CUDA
    let ctx = Arc::new(ComputeContext::new(0)?);
    let stream = ctx.new_stream()?;
    tracing::info!("CUDA context initialized on device 0");

    let gpu = ComputeOps::new(&ctx, stream.clone())?;
    let mut manager = ModelManager::new("models");
    manager.set_compute_stream(stream.cu_stream() as *mut ())?;

    // Load detector
    manager.load("YoloFace8n", "yoloface_8n.onnx")?;
    let detector = YoloFaceDetector::new(0.5);

    // Upload HWC u8 frame → GPU, convert to CHW f32
    let d_frame_u8 = gpu.upload_u8(rgb.as_raw())?;
    let mut d_frame_chw = gpu.alloc_zeros(3 * width as usize * height as usize)?;
    gpu.hwc_u8_to_chw_f32(&d_frame_u8, &mut d_frame_chw, height, width)?;

    // Detect
    let mut ws = GpuWorkspace::new(&stream)?;
    let (faces, scale) =
        detector.detect_gpu(&mut manager, &gpu, &d_frame_chw, &mut ws, height, width)?;

    tracing::info!("Detected {} faces (scale={:.3})", faces.len(), scale);

    for (i, face) in faces.iter().enumerate() {
        println!(
            "Face {i}: bbox=[{:.0}, {:.0}, {:.0}, {:.0}] score={:.3} kps=[{:.0},{:.0}] [{:.0},{:.0}] [{:.0},{:.0}] [{:.0},{:.0}] [{:.0},{:.0}]",
            face.bbox[0],
            face.bbox[1],
            face.bbox[2],
            face.bbox[3],
            face.score,
            face.kps_5[0][0],
            face.kps_5[0][1],
            face.kps_5[1][0],
            face.kps_5[1][1],
            face.kps_5[2][0],
            face.kps_5[2][1],
            face.kps_5[3][0],
            face.kps_5[3][1],
            face.kps_5[4][0],
            face.kps_5[4][1],
        );
    }

    Ok(())
}
