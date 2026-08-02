//! noperson — pure Rust GPU-accelerated face swap.
//!
//! Backend: ort (ONNX Runtime) + CUDA EP + cudarc (GPU compute)
//! UI: egui (immediate mode, native)
//! I/O: nokhwa (webcam), v4l2loopback (virtual camera)

fn main() -> anyhow::Result<()> {
    let launch_mode = noperson::launch::LaunchMode::parse(std::env::args_os().skip(1))?;

    // Init logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,ort=warn")),
        )
        .init();

    tracing::info!("noperson v{} starting up", env!("CARGO_PKG_VERSION"));

    let runtime = match noperson::runtime::prepare()? {
        noperson::runtime::BootstrapOutcome::Ready(layout) => layout,
        noperson::runtime::BootstrapOutcome::Reexec(layout) => {
            return Err(noperson::runtime::reexec(&layout).into());
        }
    };
    tracing::info!("GPU runtime ready: {}", runtime.root().display());

    let models_dir = std::path::PathBuf::from("models");
    if launch_mode == noperson::launch::LaunchMode::RuntimeCheck {
        let probe_model = models_dir.join("yoloface_8n.onnx");
        if probe_model.is_file() {
            let mut manager = noperson::models::manager::ModelManager::new(&models_dir);
            manager.load("RuntimeCheck", "yoloface_8n.onnx")?;
            tracing::info!("CUDA execution provider session check passed");

            let mut manager = noperson::models::manager::ModelManager::with_provider(
                &models_dir,
                noperson::config::settings::ExecutionProvider::TensorRT,
            );
            manager.load("RuntimeCheck", "yoloface_8n.onnx")?;
            tracing::info!("TensorRT execution provider session check passed");
        } else {
            tracing::warn!(
                "CUDA session check skipped because {} is missing",
                probe_model.display()
            );
        }
        tracing::info!("GPU runtime bootstrap check passed");
        return Ok(());
    }

    // Check models directory
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

    match launch_mode {
        noperson::launch::LaunchMode::Realtime => noperson::app::launch(models_dir)?,
        noperson::launch::LaunchMode::ExtraGui => noperson::extra_gui::launch(models_dir)?,
        noperson::launch::LaunchMode::RuntimeCheck => unreachable!("handled above"),
    }
    Ok(())
}
