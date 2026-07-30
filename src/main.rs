//! noperson — pure Rust GPU-accelerated face swap.
//!
//! Backend: ort (ONNX Runtime) + CUDA EP + cudarc (GPU compute)
//! UI: egui (immediate mode, native)
//! I/O: nokhwa (webcam), v4l2loopback (virtual camera)

fn main() -> eframe::Result {
    // Init logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    tracing::info!("noperson v{} starting up", env!("CARGO_PKG_VERSION"));

    // Check models directory
    let models_dir = std::path::PathBuf::from("models");
    if !models_dir.exists() {
        std::fs::create_dir_all(&models_dir).ok();
        tracing::warn!("Created models/ directory — place .onnx files here");
    }

    // List .onnx files
    if models_dir.exists() {
        let onnx_files: Vec<_> = std::fs::read_dir(&models_dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "onnx"))
            .collect();
        if onnx_files.is_empty() {
            tracing::warn!("No .onnx models in models/");
            tracing::warn!("Required: yoloface_8n.onnx, w600k_r50.onnx, inswapper_128.fp16.onnx");
        } else {
            for f in &onnx_files {
                tracing::info!("Found model: {}", f.file_name().to_string_lossy());
            }
        }
    }

    // Launch egui app
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([666.0, 839.0])
            .with_min_inner_size([620.0, 700.0])
            .with_title("noperson"),
        ..Default::default()
    };

    eframe::run_native(
        "noperson",
        options,
        Box::new(move |_cc| Ok(Box::new(noperson::app::App::new(models_dir)))),
    )
}
