//! FaceFusion-compatible file processing without constructing a GUI.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::CudaContext;
use indicatif::{ProgressBar, ProgressStyle};

use crate::config::parameters::{FaceSwapParams, SwapDim};
use crate::config::settings::{DetectorModel, ExecutionProvider};
use crate::extra_gui::{EditorEngineRequest, EditorRuntimeEvent, EditorRuntimeHandle};
use crate::gpu::ops::GpuOps;
use crate::launch::HeadlessOptions;
use crate::live::{
    AtomicLiveEngine, FaceAssignmentInputs, FaceIdentityInput, LiveShadowBuilder, build_live_spec,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileWorkflow {
    Image,
    Video,
}

#[derive(Debug, Clone)]
pub struct HeadlessPlan {
    pub workflow: FileWorkflow,
    pub source_path: PathBuf,
    pub target_path: PathBuf,
    pub output_path: PathBuf,
    pub provider: ExecutionProvider,
    pub device_id: i32,
    pub worker_threads: usize,
    pub params: FaceSwapParams,
}

pub fn classify_workflow(
    source_path: &Path,
    target_path: &Path,
    output_path: &Path,
) -> anyhow::Result<FileWorkflow> {
    anyhow::ensure!(is_image(source_path), "source face must be an image");
    if is_image(target_path) {
        anyhow::ensure!(
            is_image(output_path),
            "image target requires an image output"
        );
        return Ok(FileWorkflow::Image);
    }
    if is_video(target_path) {
        anyhow::ensure!(
            is_video(output_path),
            "video target requires a video output"
        );
        return Ok(FileWorkflow::Video);
    }
    anyhow::bail!("unsupported target media: {}", target_path.display())
}

pub fn build_plan(options: &HeadlessOptions) -> anyhow::Result<HeadlessPlan> {
    let workflow = classify_workflow(
        &options.source_path,
        &options.target_path,
        &options.output_path,
    )?;
    let provider = match options.execution_provider.as_str() {
        "cuda" => ExecutionProvider::Cuda,
        "tensorrt" => ExecutionProvider::TensorRT,
        value => anyhow::bail!("unsupported execution provider {value}"),
    };
    let dim = match options.swap_resolution {
        128 => SwapDim::Dim1,
        256 => SwapDim::Dim2,
        384 => SwapDim::Dim3,
        512 => SwapDim::Dim4,
        value => anyhow::bail!("unsupported swap resolution {value}"),
    };
    let mut params = FaceSwapParams::default();
    params.dim = dim;
    params.detector_score = options.face_detector_score;
    params.max_faces = options.max_faces;
    Ok(HeadlessPlan {
        workflow,
        source_path: options.source_path.clone(),
        target_path: options.target_path.clone(),
        output_path: options.output_path.clone(),
        provider,
        device_id: options.device_id,
        worker_threads: options.worker_threads.clamp(1, 32),
        params,
    })
}

pub fn prepare(options: &HeadlessOptions) -> anyhow::Result<HeadlessPlan> {
    let plan = build_plan(options)?;
    validate_files(&plan)?;
    Ok(plan)
}

pub fn run_plan(plan: &HeadlessPlan, models_dir: &Path) -> anyhow::Result<()> {
    if let Some(parent) = plan
        .output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    match plan.workflow {
        FileWorkflow::Image => process_image(plan, models_dir),
        FileWorkflow::Video => process_video(plan, models_dir),
    }
}

fn validate_files(plan: &HeadlessPlan) -> anyhow::Result<()> {
    anyhow::ensure!(
        plan.source_path.is_file(),
        "source file not found: {}",
        plan.source_path.display()
    );
    anyhow::ensure!(
        plan.target_path.is_file(),
        "target file not found: {}",
        plan.target_path.display()
    );
    let source = std::fs::canonicalize(&plan.source_path)?;
    let target = std::fs::canonicalize(&plan.target_path)?;
    if plan.output_path.exists() {
        let output = std::fs::canonicalize(&plan.output_path)?;
        anyhow::ensure!(
            output != source && output != target,
            "output path must not overwrite an input"
        );
    }
    Ok(())
}

fn engine_request(plan: &HeadlessPlan, models_dir: &Path) -> anyhow::Result<EditorEngineRequest> {
    let spec = build_live_spec(
        models_dir,
        &plan.source_path,
        plan.params.clone(),
        plan.provider,
        DetectorModel::YoloFace8n,
        plan.device_id,
    )?;
    let assignments = vec![FaceAssignmentInputs {
        source: FaceIdentityInput::Image(plan.source_path.clone()),
        target: None,
        similarity_threshold: plan.params.similarity_threshold,
        params: None,
        models: BTreeMap::new(),
    }];
    Ok(EditorEngineRequest {
        spec,
        assignments,
        worker_threads: plan.worker_threads,
    })
}

fn process_image(plan: &HeadlessPlan, models_dir: &Path) -> anyhow::Result<()> {
    let request = engine_request(plan, models_dir)?;
    let context = Arc::new(CudaContext::new(plan.device_id as usize)?);
    let stream = context.default_stream().clone();
    let gpu = Arc::new(GpuOps::new(&context, Arc::clone(&stream))?);
    let builder = LiveShadowBuilder::new(gpu, models_dir.to_path_buf(), stream);
    let mut engine =
        AtomicLiveEngine::bootstrap_inputs(builder, &request.assignments, request.spec, 1)?;
    let target = image::open(&plan.target_path)?.to_rgb8();
    let output = engine.process_rgb(target.as_raw(), target.width(), target.height())?;
    let output = image::RgbImage::from_raw(output.width, output.height, output.data)
        .ok_or_else(|| anyhow::anyhow!("engine returned invalid RGB image geometry"))?;
    output.save(&plan.output_path)?;
    tracing::info!("Saved {}", plan.output_path.display());
    Ok(())
}

fn process_video(plan: &HeadlessPlan, models_dir: &Path) -> anyhow::Result<()> {
    let request = engine_request(plan, models_dir)?;
    let runtime = EditorRuntimeHandle::spawn_headless(models_dir.to_path_buf(), plan.device_id)?;
    runtime.record(
        plan.target_path.clone(),
        plan.output_path.clone(),
        request,
        BTreeMap::new(),
    )?;
    let progress = ProgressBar::new_spinner();
    progress.set_style(
        ProgressStyle::with_template(
            "{spinner:.cyan} {msg} [{bar:40.cyan/blue}] {pos}/{len} {per_sec} ETA {eta}",
        )?
        .progress_chars("━━╸"),
    );
    progress.set_message("Processing video");
    loop {
        match runtime.recv_event()? {
            EditorRuntimeEvent::Progress { processed, total } => {
                if let Some(total) = total {
                    progress.set_length(total);
                }
                progress.set_position(processed);
            }
            EditorRuntimeEvent::Completed(path) => {
                progress.finish_with_message(format!("Saved {}", path.display()));
                return Ok(());
            }
            EditorRuntimeEvent::Failed(error) => {
                progress.abandon_with_message("Video processing failed");
                anyhow::bail!(error);
            }
            _ => {}
        }
    }
}

fn is_image(path: &Path) -> bool {
    matches!(
        extension(path).as_deref(),
        Some("jpg" | "jpeg" | "png" | "bmp" | "webp")
    )
}

fn is_video(path: &Path) -> bool {
    matches!(
        extension(path).as_deref(),
        Some("mp4" | "mkv" | "mov" | "avi" | "webm")
    )
}

fn extension(path: &Path) -> Option<String> {
    path.extension()?.to_str().map(str::to_ascii_lowercase)
}
