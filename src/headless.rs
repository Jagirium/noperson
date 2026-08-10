//! FaceFusion-compatible file processing without constructing a GUI.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::CudaContext;
use indicatif::{ProgressBar, ProgressStyle};

use crate::config::parameters::{FaceSwapParams, SwapDim};
use crate::config::settings::{DetectorModel, ExecutionProvider};
use crate::extra_gui::{EditorEngineRequest, EditorRuntimeEvent, EditorRuntimeHandle};
use crate::gpu::ops::GpuOps;
#[cfg(target_os = "linux")]
use crate::io::native_video::NativeDemuxer;
use crate::launch::{HeadlessOptions, PredecodeMode};
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
    pub predecode: PredecodeMode,
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
    let params = FaceSwapParams {
        dim,
        detector_score: options.face_detector_score,
        max_faces: options.max_faces,
        ..FaceSwapParams::default()
    };
    Ok(HeadlessPlan {
        workflow,
        source_path: options.source_path.clone(),
        target_path: options.target_path.clone(),
        output_path: options.output_path.clone(),
        provider,
        device_id: options.device_id,
        worker_threads: options.worker_threads.clamp(1, 32),
        predecode: options.predecode,
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
        predecode: plan.predecode,
    })
}

fn process_image(plan: &HeadlessPlan, models_dir: &Path) -> anyhow::Result<()> {
    let request = engine_request(plan, models_dir)?;
    let context = Arc::new(CudaContext::new(plan.device_id as usize)?);
    let stream = context.new_stream()?;
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
    #[cfg(target_os = "linux")]
    let preflight = {
        let source = NativeDemuxer::open(&plan.target_path)?;
        let info = source.video_stream();
        Some((info.frame_count, info.duration_seconds(), info.fps()))
    };
    #[cfg(not(target_os = "linux"))]
    let preflight: Option<(Option<u64>, Option<f64>, Option<f64>)> = None;

    let request = engine_request(plan, models_dir)?;
    let runtime = EditorRuntimeHandle::spawn_headless(models_dir.to_path_buf(), plan.device_id)?;
    runtime.record(
        plan.target_path.clone(),
        plan.output_path.clone(),
        request,
        BTreeMap::new(),
    )?;
    let known_total = preflight.and_then(|(total, _, _)| total);
    let progress = known_total.map_or_else(ProgressBar::new_spinner, ProgressBar::new);
    let waiting_style =
        ProgressStyle::with_template("{spinner:.cyan} {msg} [{bar:40.cyan/blue}] {pos}/{len}")?
            .progress_chars("━━╸");
    let running_style = ProgressStyle::with_template(
        "{spinner:.cyan} {msg} [{bar:40.cyan/blue}] {pos}/{len} {per_sec} ETA {eta}",
    )?
    .progress_chars("━━╸");
    let finished_style =
        ProgressStyle::with_template("{spinner:.green} {msg} [{bar:40.green/blue}] {pos}/{len}")?
            .progress_chars("━━╸");
    progress.set_style(waiting_style);
    let details = preflight
        .and_then(|(_, duration, fps)| duration.zip(fps))
        .map(|(duration, fps)| format!("Processing video · {:.2}s · {fps:.3} FPS", duration))
        .unwrap_or_else(|| "Processing video".to_owned());
    progress.set_message(details);
    let mut processing_started = false;
    let mut previous_frame_completed = None;
    let mut frame_intervals = Vec::new();
    loop {
        match runtime.recv_event()? {
            EditorRuntimeEvent::PredecodeProgress { decoded, total } => {
                progress.set_message("Pre-decoding video to VRAM");
                if let Some(total) = total {
                    progress.set_length(total);
                }
                progress.set_position(decoded);
            }
            EditorRuntimeEvent::Progress { processed, total } => {
                let completed_at = Instant::now();
                if let Some(total) = total {
                    progress.set_length(total);
                }
                if processed > 0 && !processing_started {
                    progress.set_style(running_style.clone());
                    progress.set_message("Processing video");
                    progress.set_position(0);
                    processing_started = true;
                }
                if let Some(previous) = previous_frame_completed.replace(completed_at) {
                    frame_intervals.push(completed_at.duration_since(previous));
                }
                progress.set_position(processed);
            }
            EditorRuntimeEvent::Completed(path) => {
                progress.set_style(finished_style);
                progress.finish_with_message(video_completion_message(
                    &path,
                    median_frame_rate(&frame_intervals),
                ));
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

fn median_frame_rate(intervals: &[Duration]) -> Option<f64> {
    let mut frame_times = intervals
        .iter()
        .map(Duration::as_secs_f64)
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
        .collect::<Vec<_>>();
    if frame_times.is_empty() {
        return None;
    }
    frame_times.sort_by(f64::total_cmp);
    let middle = frame_times.len() / 2;
    let median_frame_time = if frame_times.len().is_multiple_of(2) {
        (frame_times[middle - 1] + frame_times[middle]) / 2.0
    } else {
        frame_times[middle]
    };
    Some(1.0 / median_frame_time)
}

fn video_completion_message(path: &Path, median_fps: Option<f64>) -> String {
    median_fps.map_or_else(
        || format!("Saved {}", path.display()),
        |fps| format!("Saved {} · median {fps:.2} FPS", path.display()),
    )
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

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::Duration;

    use super::{median_frame_rate, video_completion_message};

    #[test]
    fn median_frame_rate_uses_completed_frame_intervals() {
        let intervals = [
            Duration::from_millis(10),
            Duration::from_millis(100),
            Duration::from_millis(10),
        ];

        assert_eq!(median_frame_rate(&intervals), Some(100.0));
    }

    #[test]
    fn completion_message_reports_median_instead_of_wall_clock_rate() {
        assert_eq!(
            video_completion_message(Path::new("out.mp4"), Some(89.734)),
            "Saved out.mp4 · median 89.73 FPS"
        );
    }
}
