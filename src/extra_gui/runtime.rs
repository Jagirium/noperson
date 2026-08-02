use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

use cudarc::driver::CudaContext;
use thiserror::Error;

use crate::engine::BuildPhase;
use crate::gpu::ops::GpuOps;
use crate::io::video::{
    FfmpegVideoSink, FfmpegVideoSource, Frame, FrameSink, FrameSource, remux_original_audio,
};
use crate::live::{
    AnalyzedIdentity, AtomicLiveEngine, IdentityAnalyzer, LiveShadowBuilder, ProcessedRgb,
    output_dimensions,
};

use super::bridge::{EditorEngineRequest, EditorRuntimeConfig};
use super::editor::{MediaId, MediaKind};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EditorJobPhase {
    #[default]
    Idle,
    Analyzing,
    Building,
    Previewing,
    Recording,
    Cancelling,
    Failed,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EditorJobState {
    phase: EditorJobPhase,
}

impl EditorJobState {
    pub const fn phase(&self) -> EditorJobPhase {
        self.phase
    }

    pub fn begin(&mut self, phase: EditorJobPhase) -> Result<(), EditorRuntimeError> {
        if self.phase != EditorJobPhase::Idle && self.phase != EditorJobPhase::Failed {
            return Err(EditorRuntimeError::Busy(self.phase));
        }
        self.phase = phase;
        Ok(())
    }

    pub fn cancel(&mut self) -> Result<(), EditorRuntimeError> {
        if !matches!(
            self.phase,
            EditorJobPhase::Analyzing
                | EditorJobPhase::Building
                | EditorJobPhase::Previewing
                | EditorJobPhase::Recording
        ) {
            return Err(EditorRuntimeError::NotRunning);
        }
        self.phase = EditorJobPhase::Cancelling;
        Ok(())
    }

    pub fn settle(&mut self) {
        self.phase = EditorJobPhase::Idle;
    }

    pub fn fail(&mut self) {
        self.phase = EditorJobPhase::Failed;
    }

    pub fn observe(&mut self, phase: EditorJobPhase) {
        self.phase = phase;
    }
}

pub struct AnalyzeRequest {
    pub target_media: MediaId,
    pub target_path: PathBuf,
    pub target_kind: MediaKind,
    pub frame_index: u64,
    pub source_paths: Vec<(String, PathBuf)>,
    pub runtime: EditorRuntimeConfig,
}

pub enum EditorRuntimeEvent {
    Phase(EditorJobPhase),
    Analyzed {
        target_media: MediaId,
        target_faces: Vec<AnalyzedIdentity>,
        source_faces: BTreeMap<String, AnalyzedIdentity>,
        total_frames: u64,
        fps: f32,
    },
    Preview {
        input: EditorPreviewImage,
        output: ProcessedRgb,
        playback: bool,
    },
    Progress {
        processed: u64,
        total: Option<u64>,
    },
    Completed(PathBuf),
    CacheCleared,
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditorPreviewImage {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

impl From<&Frame> for EditorPreviewImage {
    fn from(frame: &Frame) -> Self {
        Self {
            data: frame.data.clone(),
            width: frame.width,
            height: frame.height,
        }
    }
}

enum EditorRuntimeCommand {
    Analyze {
        request: AnalyzeRequest,
        cancellation_epoch: u64,
    },
    Preview {
        path: PathBuf,
        kind: MediaKind,
        frame_index: u64,
        request: EditorEngineRequest,
        cancellation_epoch: u64,
    },
    Playback {
        path: PathBuf,
        start_frame: u64,
        max_fps: Option<f32>,
        request: EditorEngineRequest,
        markers: BTreeMap<u64, EditorEngineRequest>,
        cancellation_epoch: u64,
    },
    Record {
        input: PathBuf,
        output: PathBuf,
        initial: EditorEngineRequest,
        markers: BTreeMap<u64, EditorEngineRequest>,
        cancellation_epoch: u64,
    },
    ClearCache,
    Shutdown,
}

#[derive(Default)]
struct CancellationEpoch(AtomicU64);

impl CancellationEpoch {
    #[cfg(test)]
    fn capture(&self) -> CancellationToken<'_> {
        CancellationToken {
            source: self,
            accepted: self.0.load(Ordering::Acquire),
        }
    }

    fn capture_value(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }

    fn token(&self, accepted: u64) -> CancellationToken<'_> {
        CancellationToken {
            source: self,
            accepted,
        }
    }

    fn cancel(&self) {
        self.0.fetch_add(1, Ordering::AcqRel);
    }
}

struct CancellationToken<'a> {
    source: &'a CancellationEpoch,
    accepted: u64,
}

impl CancellationToken<'_> {
    fn is_cancelled(&self) -> bool {
        self.source.0.load(Ordering::Acquire) != self.accepted
    }

    fn ensure_active(&self, operation: &str) -> anyhow::Result<()> {
        anyhow::ensure!(!self.is_cancelled(), "{operation} cancelled");
        Ok(())
    }
}

pub struct EditorRuntimeHandle {
    commands: Sender<EditorRuntimeCommand>,
    events: Receiver<EditorRuntimeEvent>,
    cancellation: Arc<CancellationEpoch>,
    worker: Option<JoinHandle<()>>,
}

impl EditorRuntimeHandle {
    pub fn spawn(models_dir: PathBuf) -> Result<Self, EditorRuntimeError> {
        let (command_tx, command_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();
        let cancellation = Arc::new(CancellationEpoch::default());
        let worker_cancellation = Arc::clone(&cancellation);
        let worker = std::thread::Builder::new()
            .name("extra-gui-gpu".to_owned())
            .spawn(move || worker_loop(models_dir, command_rx, event_tx, worker_cancellation))?;
        Ok(Self {
            commands: command_tx,
            events: event_rx,
            cancellation,
            worker: Some(worker),
        })
    }

    pub fn analyze(&self, request: AnalyzeRequest) -> Result<(), EditorRuntimeError> {
        let cancellation_epoch = self.cancellation.capture_value();
        self.commands
            .send(EditorRuntimeCommand::Analyze {
                request,
                cancellation_epoch,
            })
            .map_err(|_| EditorRuntimeError::WorkerStopped)
    }

    pub fn preview(
        &self,
        path: PathBuf,
        kind: MediaKind,
        frame_index: u64,
        request: EditorEngineRequest,
    ) -> Result<(), EditorRuntimeError> {
        let cancellation_epoch = self.cancellation.capture_value();
        self.commands
            .send(EditorRuntimeCommand::Preview {
                path,
                kind,
                frame_index,
                request,
                cancellation_epoch,
            })
            .map_err(|_| EditorRuntimeError::WorkerStopped)
    }

    pub fn record(
        &self,
        input: PathBuf,
        output: PathBuf,
        initial: EditorEngineRequest,
        markers: BTreeMap<u64, EditorEngineRequest>,
    ) -> Result<(), EditorRuntimeError> {
        let cancellation_epoch = self.cancellation.capture_value();
        self.commands
            .send(EditorRuntimeCommand::Record {
                input,
                output,
                initial,
                markers,
                cancellation_epoch,
            })
            .map_err(|_| EditorRuntimeError::WorkerStopped)
    }

    pub fn playback(
        &self,
        path: PathBuf,
        start_frame: u64,
        max_fps: Option<f32>,
        request: EditorEngineRequest,
        markers: BTreeMap<u64, EditorEngineRequest>,
    ) -> Result<(), EditorRuntimeError> {
        let cancellation_epoch = self.cancellation.capture_value();
        self.commands
            .send(EditorRuntimeCommand::Playback {
                path,
                start_frame,
                max_fps,
                request,
                markers,
                cancellation_epoch,
            })
            .map_err(|_| EditorRuntimeError::WorkerStopped)
    }

    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    pub fn clear_cache(&self) -> Result<(), EditorRuntimeError> {
        self.commands
            .send(EditorRuntimeCommand::ClearCache)
            .map_err(|_| EditorRuntimeError::WorkerStopped)
    }

    pub fn try_events(&self) -> impl Iterator<Item = EditorRuntimeEvent> + '_ {
        self.events.try_iter()
    }
}

impl Drop for EditorRuntimeHandle {
    fn drop(&mut self) {
        self.cancellation.cancel();
        let _ = self.commands.send(EditorRuntimeCommand::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn worker_loop(
    models_dir: PathBuf,
    commands: Receiver<EditorRuntimeCommand>,
    events: Sender<EditorRuntimeEvent>,
    cancellation: Arc<CancellationEpoch>,
) {
    let mut gpu = None;
    let mut engine = None;
    while let Ok(command) = commands.recv() {
        if matches!(command, EditorRuntimeCommand::Shutdown) {
            break;
        }
        let result = match command {
            EditorRuntimeCommand::Analyze {
                request,
                cancellation_epoch,
            } => {
                let token = cancellation.token(cancellation_epoch);
                analyze_job(&models_dir, request, &events, &token, &mut gpu)
            }
            EditorRuntimeCommand::Preview {
                path,
                kind,
                frame_index,
                request,
                cancellation_epoch,
            } => {
                let token = cancellation.token(cancellation_epoch);
                preview_job(
                    &models_dir,
                    path,
                    kind,
                    frame_index,
                    request,
                    &events,
                    &token,
                    &mut gpu,
                    &mut engine,
                )
            }
            EditorRuntimeCommand::Playback {
                path,
                start_frame,
                max_fps,
                request,
                markers,
                cancellation_epoch,
            } => {
                let token = cancellation.token(cancellation_epoch);
                playback_job(
                    &models_dir,
                    path,
                    start_frame,
                    max_fps,
                    request,
                    markers,
                    &events,
                    &token,
                    &mut gpu,
                    &mut engine,
                )
            }
            EditorRuntimeCommand::Record {
                input,
                output,
                initial,
                markers,
                cancellation_epoch,
            } => {
                let token = cancellation.token(cancellation_epoch);
                record_job(
                    &models_dir,
                    input,
                    output,
                    initial,
                    markers,
                    &events,
                    &token,
                    &mut gpu,
                    &mut engine,
                )
            }
            EditorRuntimeCommand::ClearCache => {
                engine.take();
                gpu.take();
                events
                    .send(EditorRuntimeEvent::CacheCleared)
                    .map_err(anyhow::Error::from)
            }
            EditorRuntimeCommand::Shutdown => unreachable!(),
        };
        if let Err(error) = result {
            let _ = events.send(EditorRuntimeEvent::Phase(EditorJobPhase::Failed));
            let _ = events.send(EditorRuntimeEvent::Failed(error.to_string()));
        } else {
            let _ = events.send(EditorRuntimeEvent::Phase(EditorJobPhase::Idle));
        }
    }
}

fn ensure_gpu(gpu: &mut Option<Arc<GpuOps>>) -> anyhow::Result<Arc<GpuOps>> {
    if let Some(gpu) = gpu {
        return Ok(Arc::clone(gpu));
    }
    let context = Arc::new(CudaContext::new(0)?);
    let stream = context.default_stream().clone();
    let initialized = Arc::new(GpuOps::new(&context, stream)?);
    *gpu = Some(Arc::clone(&initialized));
    Ok(initialized)
}

fn analyze_job(
    models_dir: &std::path::Path,
    request: AnalyzeRequest,
    events: &Sender<EditorRuntimeEvent>,
    cancel: &CancellationToken<'_>,
    gpu: &mut Option<Arc<GpuOps>>,
) -> anyhow::Result<()> {
    cancel.ensure_active("analysis")?;
    events.send(EditorRuntimeEvent::Phase(EditorJobPhase::Analyzing))?;
    let gpu_ops = ensure_gpu(gpu)?;
    let worker_threads = request.runtime.worker_threads;
    let mut analyzer = IdentityAnalyzer::new(
        gpu_ops,
        models_dir,
        request.runtime.params,
        request.runtime.provider,
        request.runtime.detector,
        0,
    )?;
    let (target_frame, total_frames, fps) = read_frame(
        &request.target_path,
        request.target_kind,
        request.frame_index,
        worker_threads,
    )?;
    let target_faces =
        analyzer.analyze_rgb(&target_frame.data, target_frame.width, target_frame.height)?;
    let mut source_faces = BTreeMap::new();
    for (id, path) in request.source_paths {
        anyhow::ensure!(!cancel.is_cancelled(), "analysis cancelled");
        let faces = analyzer.analyze(&path)?;
        let face = faces
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no source face detected in {}", path.display()))?;
        source_faces.insert(id, face);
    }
    events.send(EditorRuntimeEvent::Analyzed {
        target_media: request.target_media,
        target_faces,
        source_faces,
        total_frames,
        fps,
    })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn preview_job(
    models_dir: &std::path::Path,
    path: PathBuf,
    kind: MediaKind,
    frame_index: u64,
    request: EditorEngineRequest,
    events: &Sender<EditorRuntimeEvent>,
    cancel: &CancellationToken<'_>,
    gpu: &mut Option<Arc<GpuOps>>,
    engine: &mut Option<AtomicLiveEngine>,
) -> anyhow::Result<()> {
    cancel.ensure_active("preview")?;
    events.send(EditorRuntimeEvent::Phase(EditorJobPhase::Building))?;
    let gpu_ops = ensure_gpu(gpu)?;
    let worker_threads = request.worker_threads;
    configure_engine(models_dir, gpu_ops, engine, request, cancel)?;
    anyhow::ensure!(!cancel.is_cancelled(), "preview cancelled");
    events.send(EditorRuntimeEvent::Phase(EditorJobPhase::Previewing))?;
    let (frame, _, _) = read_frame(&path, kind, frame_index, worker_threads)?;
    let output = engine
        .as_mut()
        .expect("engine was configured")
        .process_rgb(&frame.data, frame.width, frame.height)?;
    events.send(EditorRuntimeEvent::Preview {
        input: EditorPreviewImage::from(&frame),
        output,
        playback: false,
    })?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record_job(
    models_dir: &std::path::Path,
    input: PathBuf,
    output: PathBuf,
    initial: EditorEngineRequest,
    mut markers: BTreeMap<u64, EditorEngineRequest>,
    events: &Sender<EditorRuntimeEvent>,
    cancel: &CancellationToken<'_>,
    gpu: &mut Option<Arc<GpuOps>>,
    engine: &mut Option<AtomicLiveEngine>,
) -> anyhow::Result<()> {
    cancel.ensure_active("recording")?;
    events.send(EditorRuntimeEvent::Phase(EditorJobPhase::Building))?;
    let gpu_ops = ensure_gpu(gpu)?;
    let worker_threads = initial.worker_threads;
    let output_params = initial.spec.params.clone();
    let initial = take_effective_request(initial, &mut markers, 0);
    configure_engine(models_dir, gpu_ops, engine, initial, cancel)?;
    events.send(EditorRuntimeEvent::Phase(EditorJobPhase::Recording))?;

    let mut source = FfmpegVideoSource::open_with_threads(&input, worker_threads)?;
    let (input_width, input_height) = source.dimensions();
    let (output_width, output_height) =
        output_dimensions(&output_params, input_width, input_height)?;
    let temporary = PathBuf::from(format!("{}.video-only.mp4", output.display()));
    let result = (|| -> anyhow::Result<()> {
        let mut sink = FfmpegVideoSink::create_with_threads(
            &temporary,
            output_width,
            output_height,
            source.fps(),
            worker_threads,
        )?;
        let total = source.frame_count();
        let mut frame = Frame::new(input_width, input_height);
        let mut index = 0_u64;
        while source.next_frame_into(&mut frame)? {
            if cancel.is_cancelled() {
                anyhow::bail!("recording cancelled");
            }
            if let Some(request) = markers.remove(&index) {
                configure_engine(models_dir, ensure_gpu(gpu)?, engine, request, cancel)?;
            }
            let processed = engine
                .as_mut()
                .expect("engine was configured")
                .process_rgb(&frame.data, frame.width, frame.height)?;
            sink.write_frame(&processed.data, processed.width, processed.height)?;
            index += 1;
            events.send(EditorRuntimeEvent::Progress {
                processed: index,
                total,
            })?;
        }
        sink.finish()?;
        remux_original_audio(&temporary, &input, &output)?;
        Ok(())
    })();
    let _ = std::fs::remove_file(&temporary);
    result?;
    events.send(EditorRuntimeEvent::Completed(output))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn playback_job(
    models_dir: &std::path::Path,
    path: PathBuf,
    start_frame: u64,
    max_fps: Option<f32>,
    request: EditorEngineRequest,
    mut markers: BTreeMap<u64, EditorEngineRequest>,
    events: &Sender<EditorRuntimeEvent>,
    cancel: &CancellationToken<'_>,
    gpu: &mut Option<Arc<GpuOps>>,
    engine: &mut Option<AtomicLiveEngine>,
) -> anyhow::Result<()> {
    cancel.ensure_active("playback")?;
    events.send(EditorRuntimeEvent::Phase(EditorJobPhase::Building))?;
    let worker_threads = request.worker_threads;
    let request = take_effective_request(request, &mut markers, start_frame);
    configure_engine(models_dir, ensure_gpu(gpu)?, engine, request, cancel)?;
    let mut source = FfmpegVideoSource::open_with_threads(&path, worker_threads)?;
    let total = source.frame_count();
    let source_fps = source.fps().max(1.0);
    let playback_fps = max_fps.unwrap_or(source_fps).min(source_fps).max(1.0);
    let frame_duration = Duration::from_secs_f32(1.0 / playback_fps);
    let (width, height) = source.dimensions();
    let mut frame = Frame::new(width, height);
    for _ in 0..start_frame {
        anyhow::ensure!(
            source.next_frame_into(&mut frame)?,
            "video frame is missing"
        );
    }
    events.send(EditorRuntimeEvent::Phase(EditorJobPhase::Previewing))?;
    let mut index = start_frame;
    while source.next_frame_into(&mut frame)? {
        if cancel.is_cancelled() {
            return Ok(());
        }
        if let Some(request) = markers.remove(&index) {
            configure_engine(models_dir, ensure_gpu(gpu)?, engine, request, cancel)?;
        }
        let started = std::time::Instant::now();
        let output = engine
            .as_mut()
            .expect("engine was configured")
            .process_rgb(&frame.data, frame.width, frame.height)?;
        events.send(EditorRuntimeEvent::Preview {
            input: EditorPreviewImage::from(&frame),
            output,
            playback: true,
        })?;
        events.send(EditorRuntimeEvent::Progress {
            processed: index.saturating_add(1),
            total,
        })?;
        index += 1;
        if let Some(remaining) = frame_duration.checked_sub(started.elapsed()) {
            std::thread::sleep(remaining);
        }
    }
    Ok(())
}

fn take_effective_request<T>(fallback: T, markers: &mut BTreeMap<u64, T>, frame: u64) -> T {
    let effective_frame = markers.range(..=frame).next_back().map(|(frame, _)| *frame);
    let effective = effective_frame
        .and_then(|effective_frame| markers.remove(&effective_frame))
        .unwrap_or(fallback);
    markers.retain(|marker_frame, _| *marker_frame > frame);
    effective
}

#[cfg(test)]
mod tests {
    use super::{CancellationEpoch, take_effective_request};
    use std::collections::BTreeMap;

    #[test]
    fn playback_starts_from_latest_marker_at_or_before_seek_frame() {
        let mut markers = BTreeMap::from([(10, "ten"), (20, "twenty"), (30, "thirty")]);
        assert_eq!(take_effective_request("base", &mut markers, 25), "twenty");
        assert_eq!(markers, BTreeMap::from([(30, "thirty")]));
    }

    #[test]
    fn playback_uses_base_before_first_marker() {
        let mut markers = BTreeMap::from([(10, "ten")]);
        assert_eq!(take_effective_request("base", &mut markers, 4), "base");
        assert_eq!(markers, BTreeMap::from([(10, "ten")]));
    }

    #[test]
    fn cancellation_epoch_never_revives_an_older_job() {
        let cancellation = CancellationEpoch::default();
        let running = cancellation.capture();
        assert!(!running.is_cancelled());

        cancellation.cancel();
        assert!(running.is_cancelled());
        assert!(running.ensure_active("running job").is_err());

        let queued_after_cancel = cancellation.capture();
        assert!(!queued_after_cancel.is_cancelled());
        assert!(queued_after_cancel.ensure_active("new job").is_ok());
        assert!(running.is_cancelled());
    }
}

fn configure_engine(
    models_dir: &std::path::Path,
    gpu: Arc<GpuOps>,
    engine: &mut Option<AtomicLiveEngine>,
    request: EditorEngineRequest,
    cancel: &CancellationToken<'_>,
) -> anyhow::Result<()> {
    cancel.ensure_active("engine build")?;
    if let Some(engine) = engine {
        engine.request_inputs(request.spec, &request.assignments)?;
        if engine.build_snapshot().phase == BuildPhase::Idle {
            return Ok(());
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(120);
        let snapshot = loop {
            if cancel.is_cancelled() {
                engine.cancel_pending_build();
                anyhow::bail!("engine build cancelled");
            }
            let snapshot = engine.wait_for_build(Duration::from_millis(100));
            if matches!(snapshot.phase, BuildPhase::Ready | BuildPhase::Failed)
                || std::time::Instant::now() >= deadline
            {
                break snapshot;
            }
        };
        anyhow::ensure!(
            snapshot.phase != BuildPhase::Failed,
            "shadow generation failed: {}",
            snapshot
                .last_failure
                .unwrap_or_else(|| "unknown error".to_owned())
        );
        anyhow::ensure!(
            matches!(snapshot.phase, BuildPhase::Ready | BuildPhase::Idle),
            "shadow generation timed out in {:?}",
            snapshot.phase
        );
        engine.poll_activation()?;
        return Ok(());
    }
    let builder = LiveShadowBuilder::new(
        Arc::clone(&gpu),
        models_dir.to_path_buf(),
        gpu.stream.clone(),
    );
    *engine = Some(AtomicLiveEngine::bootstrap_inputs(
        builder,
        &request.assignments,
        request.spec,
        1,
    )?);
    Ok(())
}

fn read_frame(
    path: &std::path::Path,
    kind: MediaKind,
    frame_index: u64,
    worker_threads: usize,
) -> anyhow::Result<(Frame, u64, f32)> {
    match kind {
        MediaKind::Image => {
            let image = image::open(path)?.to_rgb8();
            let (width, height) = image.dimensions();
            Ok((Frame::from_data(image.into_raw(), width, height), 1, 30.0))
        }
        MediaKind::Video => {
            let mut source = FfmpegVideoSource::open_with_threads(path, worker_threads)?;
            let total = source
                .frame_count()
                .unwrap_or(frame_index.saturating_add(1));
            let fps = source.fps();
            let (width, height) = source.dimensions();
            let mut frame = Frame::new(width, height);
            for _ in 0..=frame_index {
                anyhow::ensure!(
                    source.next_frame_into(&mut frame)?,
                    "video frame is missing"
                );
            }
            Ok((frame, total, fps))
        }
    }
}

#[derive(Debug, Error)]
pub enum EditorRuntimeError {
    #[error("editor runtime is busy in {0:?}")]
    Busy(EditorJobPhase),
    #[error("editor runtime has no cancellable job")]
    NotRunning,
    #[error("editor runtime worker stopped")]
    WorkerStopped,
    #[error("could not start editor runtime worker: {0}")]
    Spawn(#[from] std::io::Error),
}
