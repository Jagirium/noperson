use std::collections::BTreeMap;
#[cfg(target_os = "linux")]
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

use thiserror::Error;

#[cfg(target_os = "linux")]
use crate::backend::cuda::driver_sys as sys;
#[cfg(target_os = "linux")]
use crate::backend::{Buffer, ComputeEvent, DevicePtr, DevicePtrMut};
use crate::backend::{ComputeContext, ComputeOps};
use crate::engine::BuildPhase;
#[cfg(target_os = "linux")]
use crate::io::native_video::{
    MappedVideoSurface, NativeDemuxer, NativeMuxer, NvDecoder, NvEncoder, NvEncoderConfig,
    NvEncoderInputSurface, PixelFormat, VideoCodec, remux_audio,
};
#[cfg(not(target_os = "linux"))]
use crate::io::video::{FfmpegVideoSink, FrameSink, remux_original_audio};
use crate::io::video::{FfmpegVideoSource, Frame, FrameSource};
#[cfg(target_os = "linux")]
use crate::launch::PredecodeMode;
use crate::live::{
    AnalyzedIdentity, AtomicLiveEngine, IdentityAnalyzer, LiveShadowBuilder, ProcessedRgb,
    output_dimensions,
};
#[cfg(target_os = "linux")]
use crate::pipeline::workspace::FrameRing;

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
    #[cfg(target_os = "linux")]
    GpuPreview {
        input_bridge: Arc<crate::gpu_preview::LinuxPreviewBridge>,
        output_bridge: Arc<crate::gpu_preview::LinuxPreviewBridge>,
        faces_detected: usize,
        faces_swapped: usize,
        playback: bool,
    },
    Progress {
        processed: u64,
        total: Option<u64>,
    },
    PredecodeProgress {
        decoded: u64,
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

#[cfg(target_os = "linux")]
struct DeferredVideoSurface {
    completion: ComputeEvent,
    surface: Option<MappedVideoSurface>,
}

#[cfg(target_os = "linux")]
const FULL_PREDECODE_MIN_FREE_BYTES: usize = 16 * 1024 * 1024 * 1024;
#[cfg(target_os = "linux")]
const PREDECODE_HEADROOM_BYTES: usize = 2 * 1024 * 1024 * 1024;

#[cfg(target_os = "linux")]
fn full_predecode_allowed(
    mode: PredecodeMode,
    free_bytes: usize,
    required_bytes: usize,
) -> anyhow::Result<bool> {
    if mode == PredecodeMode::Off {
        return Ok(false);
    }
    let enough_free = free_bytes >= FULL_PREDECODE_MIN_FREE_BYTES;
    let fits = required_bytes
        .checked_add(PREDECODE_HEADROOM_BYTES)
        .is_some_and(|with_headroom| with_headroom <= free_bytes);
    if mode == PredecodeMode::Full {
        anyhow::ensure!(
            enough_free,
            "--predecode full requires at least 16 GiB of free VRAM (available: {:.2} GiB)",
            free_bytes as f64 / 1024_f64.powi(3)
        );
        anyhow::ensure!(
            fits,
            "--predecode full needs {:.2} GiB plus 2 GiB headroom (available: {:.2} GiB)",
            required_bytes as f64 / 1024_f64.powi(3),
            free_bytes as f64 / 1024_f64.powi(3)
        );
    }
    Ok(enough_free && fits)
}

#[cfg(target_os = "linux")]
impl DeferredVideoSurface {
    fn new(completion: ComputeEvent, surface: MappedVideoSurface) -> Self {
        Self {
            completion,
            surface: Some(surface),
        }
    }

    fn is_complete(&self) -> bool {
        self.completion.is_complete()
    }

    fn picture_index(&self) -> i32 {
        self.surface
            .as_ref()
            .expect("deferred NVDEC surface is present")
            .picture_index()
    }

    fn release(mut self) -> anyhow::Result<()> {
        self.completion.synchronize()?;
        self.surface.take();
        Ok(())
    }
}

#[cfg(target_os = "linux")]
impl Drop for DeferredVideoSurface {
    fn drop(&mut self) {
        if self.surface.is_some() {
            let _ = self.completion.synchronize();
            self.surface.take();
        }
    }
}

#[cfg(target_os = "linux")]
struct GpuFrameArchive {
    storage: Buffer<u8>,
    frame_stride: usize,
    row_bytes: usize,
    height: usize,
    pixel_format: PixelFormat,
}

#[cfg(target_os = "linux")]
impl GpuFrameArchive {
    fn required_bytes(
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
        frames: u64,
    ) -> anyhow::Result<usize> {
        let bytes_per_sample = match pixel_format {
            PixelFormat::Nv12 => 1_usize,
            PixelFormat::P010 => 2_usize,
        };
        let row_bytes = width as usize * bytes_per_sample;
        let frame_stride = row_bytes
            .checked_mul(height as usize + height as usize / 2)
            .ok_or_else(|| anyhow::anyhow!("decoded frame size overflow"))?;
        frame_stride
            .checked_mul(usize::try_from(frames)?)
            .ok_or_else(|| anyhow::anyhow!("decoded video archive size overflow"))
    }

    fn new(
        gpu: &Arc<ComputeOps>,
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
        frames: u64,
    ) -> anyhow::Result<Self> {
        let required = Self::required_bytes(width, height, pixel_format, frames)?;
        let row_bytes = width as usize
            * match pixel_format {
                PixelFormat::Nv12 => 1,
                PixelFormat::P010 => 2,
            };
        let frame_stride = row_bytes * (height as usize + height as usize / 2);
        let storage = unsafe { gpu.stream.alloc::<u8>(required)? };
        Ok(Self {
            storage,
            frame_stride,
            row_bytes,
            height: height as usize,
            pixel_format,
        })
    }

    fn copy_from(
        &mut self,
        gpu: &ComputeOps,
        index: u64,
        surface: &MappedVideoSurface,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            surface.pixel_format() == self.pixel_format,
            "NVDEC pixel format changed during predecode"
        );
        let offset = self
            .frame_stride
            .checked_mul(usize::try_from(index)?)
            .ok_or_else(|| anyhow::anyhow!("predecode frame offset overflow"))?;
        anyhow::ensure!(
            offset + self.frame_stride <= self.storage.len(),
            "decoded frame count exceeded metadata estimate"
        );
        let (base, _write) = self.storage.device_ptr_mut(&gpu.stream);
        let copy_plane = |src_device: u64, dst_device: u64, height: usize| -> anyhow::Result<()> {
            let copy = sys::CUDA_MEMCPY2D {
                srcXInBytes: 0,
                srcY: 0,
                srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                srcHost: std::ptr::null(),
                srcDevice: src_device,
                srcArray: std::ptr::null_mut(),
                srcPitch: surface.pitch() as usize,
                dstXInBytes: 0,
                dstY: 0,
                dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
                dstHost: std::ptr::null_mut(),
                dstDevice: dst_device,
                dstArray: std::ptr::null_mut(),
                dstPitch: self.row_bytes,
                WidthInBytes: self.row_bytes,
                Height: height,
            };
            unsafe { sys::cuMemcpy2DAsync_v2(&copy, gpu.stream.cu_stream()) }.result()?;
            Ok(())
        };
        copy_plane(surface.device_ptr(), base + offset as u64, self.height)?;
        copy_plane(
            surface.device_ptr() + u64::from(surface.pitch()) * self.height as u64,
            base + (offset + self.row_bytes * self.height) as u64,
            self.height / 2,
        )?;
        Ok(())
    }

    fn frame_ptr(&self, gpu: &ComputeOps, index: u64) -> anyhow::Result<u64> {
        let offset = self.frame_stride * usize::try_from(index)?;
        anyhow::ensure!(
            offset + self.frame_stride <= self.storage.len(),
            "predecoded frame index is out of bounds"
        );
        let (base, _read) = self.storage.device_ptr(&gpu.stream);
        Ok(base + offset as u64)
    }
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
    pub fn spawn_with_render_state(
        models_dir: PathBuf,
        render_state: Option<egui_wgpu::RenderState>,
    ) -> Result<Self, EditorRuntimeError> {
        Self::spawn(models_dir, render_state, 0)
    }

    pub fn spawn_headless(models_dir: PathBuf, device_id: i32) -> Result<Self, EditorRuntimeError> {
        Self::spawn(models_dir, None, device_id)
    }

    fn spawn(
        models_dir: PathBuf,
        render_state: Option<egui_wgpu::RenderState>,
        device_id: i32,
    ) -> Result<Self, EditorRuntimeError> {
        let (command_tx, command_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();
        let cancellation = Arc::new(CancellationEpoch::default());
        let worker_cancellation = Arc::clone(&cancellation);
        let worker = std::thread::Builder::new()
            .name("extra-gui-gpu".to_owned())
            .spawn(move || {
                worker_loop(
                    models_dir,
                    command_rx,
                    event_tx,
                    worker_cancellation,
                    render_state,
                    device_id,
                )
            })?;
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

    pub fn recv_event(&self) -> Result<EditorRuntimeEvent, EditorRuntimeError> {
        self.events
            .recv()
            .map_err(|_| EditorRuntimeError::WorkerStopped)
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
    render_state: Option<egui_wgpu::RenderState>,
    device_id: i32,
) {
    let mut gpu = None;
    let mut engine = None;
    #[cfg(target_os = "linux")]
    let mut gpu_input_preview = None;
    #[cfg(target_os = "linux")]
    let mut gpu_output_preview = None;
    #[cfg(target_os = "linux")]
    let mut preview_frames = None;
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
                analyze_job(&models_dir, request, &events, &token, &mut gpu, device_id)
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
                    device_id,
                    #[cfg(target_os = "linux")]
                    render_state.as_ref(),
                    #[cfg(target_os = "linux")]
                    &mut gpu_input_preview,
                    #[cfg(target_os = "linux")]
                    &mut gpu_output_preview,
                    #[cfg(target_os = "linux")]
                    &mut preview_frames,
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
                    device_id,
                    #[cfg(target_os = "linux")]
                    render_state.as_ref(),
                    #[cfg(target_os = "linux")]
                    &mut gpu_input_preview,
                    #[cfg(target_os = "linux")]
                    &mut gpu_output_preview,
                    #[cfg(target_os = "linux")]
                    &mut preview_frames,
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
                    device_id,
                )
            }
            EditorRuntimeCommand::ClearCache => {
                engine.take();
                gpu.take();
                #[cfg(target_os = "linux")]
                {
                    gpu_input_preview.take();
                    gpu_output_preview.take();
                    preview_frames.take();
                }
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

fn ensure_gpu(
    gpu: &mut Option<Arc<ComputeOps>>,
    device_id: i32,
) -> anyhow::Result<Arc<ComputeOps>> {
    if let Some(gpu) = gpu {
        return Ok(Arc::clone(gpu));
    }
    let context = Arc::new(ComputeContext::new(device_id as usize)?);
    // A real stream handle is required for NvEncSetIOCudaStreams. CUDA's
    // legacy default stream is represented by NULL, which made the native
    // shim skip NVENC ordering and race the final RGB -> NV12 kernel.
    let stream = context.new_stream()?;
    let initialized = Arc::new(ComputeOps::new(&context, stream)?);
    *gpu = Some(Arc::clone(&initialized));
    Ok(initialized)
}

fn analyze_job(
    models_dir: &std::path::Path,
    request: AnalyzeRequest,
    events: &Sender<EditorRuntimeEvent>,
    cancel: &CancellationToken<'_>,
    gpu: &mut Option<Arc<ComputeOps>>,
    device_id: i32,
) -> anyhow::Result<()> {
    cancel.ensure_active("analysis")?;
    events.send(EditorRuntimeEvent::Phase(EditorJobPhase::Analyzing))?;
    let gpu_ops = ensure_gpu(gpu, device_id)?;
    let worker_threads = request.runtime.worker_threads;
    let mut analyzer = IdentityAnalyzer::new(
        gpu_ops,
        models_dir,
        request.runtime.params,
        request.runtime.provider,
        request.runtime.detector,
        device_id,
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
    gpu: &mut Option<Arc<ComputeOps>>,
    engine: &mut Option<AtomicLiveEngine>,
    device_id: i32,
    #[cfg(target_os = "linux")] render_state: Option<&egui_wgpu::RenderState>,
    #[cfg(target_os = "linux")] gpu_input_preview: &mut Option<
        Arc<crate::gpu_preview::LinuxPreviewBridge>,
    >,
    #[cfg(target_os = "linux")] gpu_output_preview: &mut Option<
        Arc<crate::gpu_preview::LinuxPreviewBridge>,
    >,
    #[cfg(target_os = "linux")] preview_frames: &mut Option<(u32, u32, FrameRing)>,
) -> anyhow::Result<()> {
    cancel.ensure_active("preview")?;
    events.send(EditorRuntimeEvent::Phase(EditorJobPhase::Building))?;
    let gpu_ops = ensure_gpu(gpu, device_id)?;
    let worker_threads = request.worker_threads;
    configure_engine(models_dir, Arc::clone(&gpu_ops), engine, request, cancel)?;
    anyhow::ensure!(!cancel.is_cancelled(), "preview cancelled");
    events.send(EditorRuntimeEvent::Phase(EditorJobPhase::Previewing))?;
    let (frame, _, _) = read_frame(&path, kind, frame_index, worker_threads)?;
    #[cfg(target_os = "linux")]
    if let Some(render_state) = render_state {
        let (input_bridge, output_bridge, result) = process_gpu_preview(
            &gpu_ops,
            engine.as_mut().expect("engine was configured"),
            &frame,
            render_state,
            gpu_input_preview,
            gpu_output_preview,
            preview_frames,
        )?;
        events.send(EditorRuntimeEvent::GpuPreview {
            input_bridge,
            output_bridge,
            faces_detected: result.faces_detected,
            faces_swapped: result.faces_swapped,
            playback: false,
        })?;
        return Ok(());
    }
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
    markers: BTreeMap<u64, EditorEngineRequest>,
    events: &Sender<EditorRuntimeEvent>,
    cancel: &CancellationToken<'_>,
    gpu: &mut Option<Arc<ComputeOps>>,
    engine: &mut Option<AtomicLiveEngine>,
    device_id: i32,
) -> anyhow::Result<()> {
    #[cfg(target_os = "linux")]
    {
        record_job_native(
            models_dir, input, output, initial, markers, events, cancel, gpu, engine, device_id,
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        record_job_pipe(
            models_dir, input, output, initial, markers, events, cancel, gpu, engine, device_id,
        )
    }
}

#[cfg(not(target_os = "linux"))]
#[allow(clippy::too_many_arguments)]
fn record_job_pipe(
    models_dir: &std::path::Path,
    input: PathBuf,
    output: PathBuf,
    initial: EditorEngineRequest,
    mut markers: BTreeMap<u64, EditorEngineRequest>,
    events: &Sender<EditorRuntimeEvent>,
    cancel: &CancellationToken<'_>,
    gpu: &mut Option<Arc<ComputeOps>>,
    engine: &mut Option<AtomicLiveEngine>,
    device_id: i32,
) -> anyhow::Result<()> {
    cancel.ensure_active("recording")?;
    events.send(EditorRuntimeEvent::Phase(EditorJobPhase::Building))?;
    let gpu_ops = ensure_gpu(gpu, device_id)?;
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
                configure_engine(
                    models_dir,
                    ensure_gpu(gpu, device_id)?,
                    engine,
                    request,
                    cancel,
                )?;
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

#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn record_job_native(
    models_dir: &std::path::Path,
    input: PathBuf,
    output: PathBuf,
    initial: EditorEngineRequest,
    mut markers: BTreeMap<u64, EditorEngineRequest>,
    events: &Sender<EditorRuntimeEvent>,
    cancel: &CancellationToken<'_>,
    gpu: &mut Option<Arc<ComputeOps>>,
    engine: &mut Option<AtomicLiveEngine>,
    device_id: i32,
) -> anyhow::Result<()> {
    cancel.ensure_active("recording")?;
    events.send(EditorRuntimeEvent::Phase(EditorJobPhase::Building))?;
    let gpu_ops = ensure_gpu(gpu, device_id)?;
    let output_params = initial.spec.params.clone();
    let predecode_mode = initial.predecode;
    let initial = take_effective_request(initial, &mut markers, 0);
    configure_engine(models_dir, Arc::clone(&gpu_ops), engine, initial, cancel)?;
    events.send(EditorRuntimeEvent::Phase(EditorJobPhase::Recording))?;

    let mut source = NativeDemuxer::open(&input)?;
    let source_info = source.video_stream().clone();
    let (output_width, output_height) =
        output_dimensions(&output_params, source_info.width, source_info.height)?;
    anyhow::ensure!(
        output_width.is_multiple_of(2) && output_height.is_multiple_of(2),
        "NVENC output dimensions must be even: {output_width}x{output_height}"
    );
    let frame_rate_num = source_info.frame_rate_num.max(1);
    let frame_rate_den = source_info.frame_rate_den.max(1);
    let frame_duration = ((i128::from(source_info.time_base_den) * i128::from(frame_rate_den)
        + i128::from(source_info.time_base_num) * i128::from(frame_rate_num) / 2)
        / (i128::from(source_info.time_base_num) * i128::from(frame_rate_num)))
    .max(1) as i64;

    let context = gpu_ops.stream.context();
    let mut decoder = unsafe {
        NvDecoder::open(
            context.cu_ctx() as *mut std::ffi::c_void,
            gpu_ops.stream.cu_stream() as *mut std::ffi::c_void,
            source_info.codec,
        )?
    };
    let output_pixel_format = if source_info.bit_depth > 8 {
        PixelFormat::P010
    } else {
        PixelFormat::Nv12
    };
    let mut config = NvEncoderConfig::h264_quality(
        output_width,
        output_height,
        frame_rate_num,
        frame_rate_den,
        source_info.time_base_num,
        source_info.time_base_den,
    )
    .with_color(source_info.color);
    if output_pixel_format == PixelFormat::P010 {
        config.codec = VideoCodec::Hevc;
        config.pixel_format = PixelFormat::P010;
    }
    let encode_surfaces = (0..5)
        .map(|_| {
            NvEncoderInputSurface::new(
                Arc::clone(&gpu_ops.stream),
                output_width,
                output_height,
                output_pixel_format,
            )
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut encoder = unsafe {
        NvEncoder::open(
            context.cu_ctx() as *mut std::ffi::c_void,
            gpu_ops.stream.cu_stream() as *mut std::ffi::c_void,
            config,
        )?
    };
    let temporary = PathBuf::from(format!("{}.video-only.mp4", output.display()));
    let result = (|| -> anyhow::Result<()> {
        let mut muxer = NativeMuxer::create(&temporary, encoder.video_stream())?;
        let mut chw = gpu_ops
            .alloc_zeros(3usize * source_info.width as usize * source_info.height as usize)?;
        let mut index = 0_u64;
        let mut next_pts = 0_i64;
        let mut pending_surfaces: VecDeque<DeferredVideoSurface> = VecDeque::new();

        let estimated_frames = source_info.frame_count;
        let archive_capacity = estimated_frames
            .and_then(|frames| frames.checked_add(64))
            .unwrap_or(0);
        let archive_bytes = GpuFrameArchive::required_bytes(
            source_info.width,
            source_info.height,
            output_pixel_format,
            archive_capacity,
        )?;
        let (free_vram, _) = context.mem_get_info()?;
        let use_full_predecode = estimated_frames.is_some()
            && full_predecode_allowed(predecode_mode, free_vram, archive_bytes)?;

        if predecode_mode == PredecodeMode::Full && estimated_frames.is_none() {
            anyhow::bail!("--predecode full requires duration and FPS metadata");
        }

        if use_full_predecode {
            tracing::info!(
                frames = estimated_frames.unwrap_or_default(),
                archive_gib = archive_bytes as f64 / 1024_f64.powi(3),
                free_gib = free_vram as f64 / 1024_f64.powi(3),
                "Pre-decoding video into contiguous GPU memory"
            );
            let mut archive = GpuFrameArchive::new(
                &gpu_ops,
                source_info.width,
                source_info.height,
                output_pixel_format,
                archive_capacity,
            )?;
            let mut timestamps = Vec::with_capacity(usize::try_from(archive_capacity)?);

            let mut drain_archive = |decoder: &mut NvDecoder| -> anyhow::Result<()> {
                loop {
                    while pending_surfaces
                        .front()
                        .is_some_and(DeferredVideoSurface::is_complete)
                    {
                        pending_surfaces.pop_front();
                    }
                    let Some(next_picture_index) = decoder.next_picture_index()? else {
                        break;
                    };
                    if let Some(position) = pending_surfaces
                        .iter()
                        .position(|surface| surface.picture_index() == next_picture_index)
                    {
                        pending_surfaces
                            .remove(position)
                            .expect("reused NVDEC picture slot is pending")
                            .release()?;
                    }
                    if pending_surfaces.len() >= 6 {
                        pending_surfaces
                            .pop_front()
                            .expect("deferred NVDEC surface queue is not empty")
                            .release()?;
                    }
                    let surface = decoder
                        .next_frame()?
                        .expect("peeked NVDEC picture slot is ready");
                    cancel.ensure_active("pre-decoding")?;
                    let frame_index = timestamps.len() as u64;
                    archive.copy_from(&gpu_ops, frame_index, &surface)?;
                    timestamps.push(surface.timestamp_100ns());
                    let copy_done = gpu_ops.stream.record_event(None)?;
                    pending_surfaces.push_back(DeferredVideoSurface::new(copy_done, surface));
                    events.send(EditorRuntimeEvent::PredecodeProgress {
                        decoded: timestamps.len() as u64,
                        total: estimated_frames,
                    })?;
                }
                Ok(())
            };

            while let Some(packet) = source.next_decode_packet()? {
                cancel.ensure_active("pre-decoding")?;
                decoder.send_packet(
                    &packet,
                    source_info.time_base_num,
                    source_info.time_base_den,
                )?;
                drain_archive(&mut decoder)?;
            }
            decoder.flush()?;
            drain_archive(&mut decoder)?;
            while let Some(surface) = pending_surfaces.pop_front() {
                surface.release()?;
            }

            let total = timestamps.len() as u64;
            for (frame_index, timestamp_100ns) in timestamps.into_iter().enumerate() {
                cancel.ensure_active("recording")?;
                let frame_index = frame_index as u64;
                if let Some(request) = markers.remove(&frame_index) {
                    configure_engine(
                        models_dir,
                        ensure_gpu(gpu, device_id)?,
                        engine,
                        request,
                        cancel,
                    )?;
                }
                unsafe {
                    gpu_ops.nv12_device_to_chw_f32(
                        archive.frame_ptr(&gpu_ops, frame_index)?,
                        archive.row_bytes as u32,
                        &mut chw,
                        source_info.height,
                        source_info.width,
                        output_pixel_format,
                        source_info.color.matrix,
                        source_info.color.range,
                    )?;
                }
                engine
                    .as_mut()
                    .expect("engine was configured")
                    .process_chw_to_pitched_nv12(
                        &mut chw,
                        source_info.height,
                        source_info.width,
                        encode_surfaces[frame_index as usize % 5].device_ptr(),
                        encode_surfaces[frame_index as usize % 5].pitch(),
                        source_info.color.matrix,
                        source_info.color.range,
                        output_pixel_format,
                    )?;
                let timestamp_pts = (i128::from(timestamp_100ns)
                    * i128::from(source_info.time_base_den)
                    / (10_000_000_i128 * i128::from(source_info.time_base_num)))
                    as i64;
                let pts = timestamp_pts.max(next_pts);
                next_pts = pts.saturating_add(frame_duration);
                let encode_surface = &encode_surfaces[frame_index as usize % 5];
                if let Some(packet) = unsafe {
                    encoder.encode_device_frame(
                        encode_surface.device_ptr(),
                        encode_surface.pitch(),
                        pts,
                        frame_duration,
                    )?
                } {
                    muxer.write_video_packet(&packet)?;
                }
                index = frame_index.saturating_add(1);
                events.send(EditorRuntimeEvent::Progress {
                    processed: index,
                    total: Some(total),
                })?;
            }
        } else {
            let mut drain = |decoder: &mut NvDecoder| -> anyhow::Result<()> {
                loop {
                    while pending_surfaces
                        .front()
                        .is_some_and(DeferredVideoSurface::is_complete)
                    {
                        pending_surfaces.pop_front();
                    }
                    let Some(next_picture_index) = decoder.next_picture_index()? else {
                        break;
                    };
                    if let Some(position) = pending_surfaces
                        .iter()
                        .position(|surface| surface.picture_index() == next_picture_index)
                    {
                        pending_surfaces
                            .remove(position)
                            .expect("reused NVDEC picture slot is pending")
                            .release()?;
                    }
                    if pending_surfaces.len() >= 6 {
                        pending_surfaces
                            .pop_front()
                            .expect("deferred NVDEC surface queue is not empty")
                            .release()?;
                    }
                    let surface = decoder
                        .next_frame()?
                        .expect("peeked NVDEC picture slot is ready");
                    cancel.ensure_active("recording")?;
                    if let Some(request) = markers.remove(&index) {
                        configure_engine(
                            models_dir,
                            ensure_gpu(gpu, device_id)?,
                            engine,
                            request,
                            cancel,
                        )?;
                    }
                    let timestamp_100ns = surface.timestamp_100ns();
                    unsafe {
                        gpu_ops.nv12_device_to_chw_f32(
                            surface.device_ptr(),
                            surface.pitch(),
                            &mut chw,
                            surface.height(),
                            surface.width(),
                            surface.pixel_format(),
                            source_info.color.matrix,
                            source_info.color.range,
                        )?;
                    }
                    let conversion_done = gpu_ops.stream.record_event(None)?;
                    pending_surfaces.push_back(DeferredVideoSurface::new(conversion_done, surface));
                    engine
                        .as_mut()
                        .expect("engine was configured")
                        .process_chw_to_pitched_nv12(
                            &mut chw,
                            source_info.height,
                            source_info.width,
                            encode_surfaces[index as usize % 5].device_ptr(),
                            encode_surfaces[index as usize % 5].pitch(),
                            source_info.color.matrix,
                            source_info.color.range,
                            output_pixel_format,
                        )?;
                    let timestamp_pts = (i128::from(timestamp_100ns)
                        * i128::from(source_info.time_base_den)
                        / (10_000_000_i128 * i128::from(source_info.time_base_num)))
                        as i64;
                    let pts = timestamp_pts.max(next_pts);
                    next_pts = pts.saturating_add(frame_duration);
                    let surface = &encode_surfaces[index as usize % 5];
                    if let Some(packet) = unsafe {
                        encoder.encode_device_frame(
                            surface.device_ptr(),
                            surface.pitch(),
                            pts,
                            frame_duration,
                        )?
                    } {
                        muxer.write_video_packet(&packet)?;
                    }
                    index = index.saturating_add(1);
                    events.send(EditorRuntimeEvent::Progress {
                        processed: index,
                        total: source_info.frame_count,
                    })?;
                }
                Ok(())
            };

            while let Some(packet) = source.next_decode_packet()? {
                cancel.ensure_active("recording")?;
                decoder.send_packet(
                    &packet,
                    source_info.time_base_num,
                    source_info.time_base_den,
                )?;
                drain(&mut decoder)?;
            }
            decoder.flush()?;
            drain(&mut decoder)?;
            while let Some(surface) = pending_surfaces.pop_front() {
                surface.release()?;
            }
        }
        for packet in encoder.finish()? {
            muxer.write_video_packet(&packet)?;
        }
        muxer.finish()?;
        remux_audio(&temporary, &input, &output)?;
        Ok(())
    })();
    let _ = std::fs::remove_file(&temporary);
    result?;
    events.send(EditorRuntimeEvent::Completed(output))?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn process_gpu_preview(
    gpu: &Arc<ComputeOps>,
    engine: &mut AtomicLiveEngine,
    frame: &Frame,
    render_state: &egui_wgpu::RenderState,
    input_bridge: &mut Option<Arc<crate::gpu_preview::LinuxPreviewBridge>>,
    output_bridge: &mut Option<Arc<crate::gpu_preview::LinuxPreviewBridge>>,
    frame_ring: &mut Option<(u32, u32, FrameRing)>,
) -> anyhow::Result<(
    Arc<crate::gpu_preview::LinuxPreviewBridge>,
    Arc<crate::gpu_preview::LinuxPreviewBridge>,
    crate::pipeline::frame_processor::FrameResult,
)> {
    let dimensions_changed = frame_ring
        .as_ref()
        .is_none_or(|(width, height, _)| (*width, *height) != (frame.width, frame.height));
    if dimensions_changed {
        *frame_ring = Some((
            frame.width,
            frame.height,
            FrameRing::new_for_dimensions(
                gpu.stream.context(),
                &gpu.stream,
                1,
                frame.width,
                frame.height,
            )?,
        ));
    }
    let geometry_changed = output_bridge.as_ref().is_none_or(|bridge| {
        let geometry = bridge.geometry();
        (geometry.width(), geometry.height()) != (frame.width, frame.height)
    });
    if geometry_changed {
        let geometry = crate::gpu_preview::PreviewGeometry::new(frame.width, frame.height)
            .ok_or_else(|| anyhow::anyhow!("invalid editor preview geometry"))?;
        *input_bridge = Some(crate::gpu_preview::LinuxPreviewBridge::new(
            render_state,
            geometry,
        )?);
        *output_bridge = Some(crate::gpu_preview::LinuxPreviewBridge::new(
            render_state,
            geometry,
        )?);
    }
    let slot = frame_ring
        .as_mut()
        .expect("editor frame ring initialized")
        .2
        .acquire(frame.width, frame.height)?;
    gpu.upload_into_u8(&frame.data, &mut slot.u8_in)?;
    gpu.hwc_u8_to_chw_f32(&slot.u8_in, &mut slot.chw, frame.height, frame.width)?;
    let input_bridge = Arc::clone(
        input_bridge
            .as_ref()
            .expect("editor input GPU preview initialized"),
    );
    let input_event_bridge = Arc::clone(&input_bridge);
    let input_write = input_bridge.stage(gpu, &slot.chw)?;
    let result = engine.process_chw(&mut slot.chw, frame.height, frame.width)?;
    let output_bridge = Arc::clone(
        output_bridge
            .as_ref()
            .expect("editor output GPU preview initialized"),
    );
    let output_event_bridge = Arc::clone(&output_bridge);
    let output_write = output_bridge.stage(gpu, &slot.chw)?;
    gpu.sync()?;
    if let Some(write) = input_write {
        write.commit();
    }
    if let Some(write) = output_write {
        write.commit();
    }
    Ok((input_event_bridge, output_event_bridge, result))
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
    gpu: &mut Option<Arc<ComputeOps>>,
    engine: &mut Option<AtomicLiveEngine>,
    device_id: i32,
    #[cfg(target_os = "linux")] render_state: Option<&egui_wgpu::RenderState>,
    #[cfg(target_os = "linux")] gpu_input_preview: &mut Option<
        Arc<crate::gpu_preview::LinuxPreviewBridge>,
    >,
    #[cfg(target_os = "linux")] gpu_output_preview: &mut Option<
        Arc<crate::gpu_preview::LinuxPreviewBridge>,
    >,
    #[cfg(target_os = "linux")] preview_frames: &mut Option<(u32, u32, FrameRing)>,
) -> anyhow::Result<()> {
    cancel.ensure_active("playback")?;
    events.send(EditorRuntimeEvent::Phase(EditorJobPhase::Building))?;
    let worker_threads = request.worker_threads;
    let request = take_effective_request(request, &mut markers, start_frame);
    configure_engine(
        models_dir,
        ensure_gpu(gpu, device_id)?,
        engine,
        request,
        cancel,
    )?;
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
            configure_engine(
                models_dir,
                ensure_gpu(gpu, device_id)?,
                engine,
                request,
                cancel,
            )?;
        }
        let started = std::time::Instant::now();
        #[cfg(target_os = "linux")]
        if let Some(render_state) = render_state {
            let gpu_ops = ensure_gpu(gpu, device_id)?;
            let (input_bridge, output_bridge, result) = process_gpu_preview(
                &gpu_ops,
                engine.as_mut().expect("engine was configured"),
                &frame,
                render_state,
                gpu_input_preview,
                gpu_output_preview,
                preview_frames,
            )?;
            events.send(EditorRuntimeEvent::GpuPreview {
                input_bridge,
                output_bridge,
                faces_detected: result.faces_detected,
                faces_swapped: result.faces_swapped,
                playback: true,
            })?;
        } else {
            let output = engine
                .as_mut()
                .expect("engine was configured")
                .process_rgb(&frame.data, frame.width, frame.height)?;
            events.send(EditorRuntimeEvent::Preview {
                input: EditorPreviewImage::from(&frame),
                output,
                playback: true,
            })?;
        }
        #[cfg(not(target_os = "linux"))]
        let output = engine
            .as_mut()
            .expect("engine was configured")
            .process_rgb(&frame.data, frame.width, frame.height)?;
        #[cfg(not(target_os = "linux"))]
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
    #[cfg(target_os = "linux")]
    use super::{FULL_PREDECODE_MIN_FREE_BYTES, full_predecode_allowed};
    #[cfg(target_os = "linux")]
    use crate::launch::PredecodeMode;
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

    #[cfg(target_os = "linux")]
    #[test]
    fn full_predecode_is_gated_by_sixteen_gib_free() {
        let below = FULL_PREDECODE_MIN_FREE_BYTES - 1;
        assert!(
            full_predecode_allowed(PredecodeMode::Auto, below, 1).is_ok_and(|use_full| !use_full)
        );
        assert!(full_predecode_allowed(PredecodeMode::Full, below, 1).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn full_predecode_keeps_two_gib_headroom() {
        let free = 20 * 1024 * 1024 * 1024;
        assert!(full_predecode_allowed(PredecodeMode::Full, free, 18 * 1024 * 1024 * 1024).is_ok());
        assert!(
            full_predecode_allowed(PredecodeMode::Full, free, 19 * 1024 * 1024 * 1024).is_err()
        );
    }
}

fn configure_engine(
    models_dir: &std::path::Path,
    gpu: Arc<ComputeOps>,
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
