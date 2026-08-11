//! noperson — pure Rust GPU-accelerated face swap.
//!
//! Backend: ort (ONNX Runtime) + CUDA EP + cudarc (GPU compute)
//! UI: egui (immediate mode, native)
//! I/O: nokhwa (webcam), v4l2loopback (virtual camera)

fn main() -> anyhow::Result<()> {
    let options = noperson::launch::LaunchOptions::parse(std::env::args_os().skip(1))?;
    if options.help {
        if options.mode == noperson::launch::LaunchMode::Headless {
            print!("{}", noperson::launch::headless_help_text());
        } else {
            print!("{}", noperson::launch::help_text());
        }
        return Ok(());
    }
    let launch_mode = options.mode;
    let headless_plan = options
        .headless
        .as_ref()
        .map(noperson::headless::prepare)
        .transpose()?;

    // Init logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,ort=warn")),
        )
        .init();

    let runtime = match noperson::runtime::prepare()? {
        noperson::runtime::BootstrapOutcome::Ready(layout) => layout,
        noperson::runtime::BootstrapOutcome::Reexec(layout) => {
            return Err(noperson::runtime::reexec(&layout).into());
        }
    };
    tracing::info!("noperson v{} starting up", env!("CARGO_PKG_VERSION"));
    tracing::info!("GPU runtime ready: {}", runtime.root().display());

    let explicit_models_dir = options.models_dir;
    let models_dir = explicit_models_dir
        .or_else(|| std::env::var_os("NOPERSON_MODELS_DIR").map(std::path::PathBuf::from))
        .unwrap_or_else(|| std::path::PathBuf::from("models"));
    tracing::info!("Models directory: {}", models_dir.display());

    let async_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    async_runtime.block_on(noperson::models::install::ensure_required_models(
        &models_dir,
    ))?;

    if launch_mode == noperson::launch::LaunchMode::RuntimeCheck {
        let probe_model = models_dir.join("yoloface_8n.onnx");
        if probe_model.is_file() {
            for provider in noperson::backend::CompiledCapabilities::current().inference_providers {
                let mut manager =
                    noperson::models::manager::ModelManager::with_provider(&models_dir, *provider);
                manager.load("RuntimeCheck", "yoloface_8n.onnx")?;
                tracing::info!(
                    provider = provider.display_name(),
                    "execution provider session check passed"
                );
            }
        } else {
            tracing::warn!(
                "CUDA session check skipped because {} is missing",
                probe_model.display()
            );
        }
        tracing::info!("GPU runtime bootstrap check passed");
        return Ok(());
    }

    match launch_mode {
        noperson::launch::LaunchMode::Realtime => noperson::app::launch(models_dir)?,
        noperson::launch::LaunchMode::ExtraGui => noperson::extra_gui::launch(models_dir)?,
        noperson::launch::LaunchMode::Headless => noperson::headless::run_plan(
            headless_plan
                .as_ref()
                .expect("headless plan exists for headless-run"),
            &models_dir,
        )?,
        noperson::launch::LaunchMode::RuntimeCheck => unreachable!("handled above"),
    }
    Ok(())
}
