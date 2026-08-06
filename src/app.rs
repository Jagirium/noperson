//! egui application — face swap UI.
//!
//! Layout (inspired by Deep-Live-Cam):
//!   ┌─ Source/Target row ─── two image drop zones + swap button
//!   ├─ Options card ──────── toggles (enabled, restorer, occluder, xseg, ...)
//!   ├─ Refinement card ───── sliders (borders, blur, strength, threshold)
//!   ├─ Output card ────────── virtual camera / file
//!   ├─ Action row ─────────── Start / Stop / Preview
//!   └─ Status bar ─────────── fps + face count + status text

mod realtime_preview;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

use cudarc::driver::CudaContext;
use eframe::egui::Color32;
use eframe::egui::epaint::text::{FontInsert, FontPriority, InsertFontFamily};
use eframe::egui::{FontData, FontFamily};
use elegance::Theme;
use elegance::{
    Accent as EleganceAccent, Button as EleganceButton, ButtonSize, Card as EleganceCard,
    IndicatorState, SegmentedControl, SegmentedSize, StatusPill, Switch,
};

use crate::config::parameters::{FaceSwapParams, RestorerMode, SwapDim};
use crate::config::settings::ExecutionProvider;
use crate::gpu::ops::GpuOps;
use crate::io::{VirtualCamera, send_virtual_camera_frame};
use crate::live::LiveEngine;
use crate::models::live_catalog::CANONICAL_SWAPPER_FILENAME;
use crate::pipeline::workspace::FrameRing;
use realtime_preview::{RealtimePreviewSnapshot, RealtimePreviewState};

pub fn launch(models_dir: PathBuf) -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([666.0, 839.0])
            .with_min_inner_size([620.0, 700.0])
            .with_title("noperson"),
        ..Default::default()
    };
    eframe::run_native(
        "noperson",
        options,
        Box::new(move |cc| {
            Ok(Box::new(App::new_with_render_state(
                models_dir,
                cc.wgpu_render_state.clone(),
            )))
        }),
    )
}

// ─── Deep-Live-Cam-inspired palette ──────────────────────────────────────
const BG: Color32 = Color32::from_rgb(0x0d, 0x0d, 0x11);
const CARD: Color32 = Color32::from_rgb(0x1c, 0x1e, 0x22);
const CARD_HOVER: Color32 = Color32::from_rgb(0x26, 0x29, 0x2e);
const CARD_BORDER: Color32 = Color32::from_rgb(0x36, 0x3a, 0x42);
const ACCENT: Color32 = Color32::from_rgb(0x34, 0x78, 0xf6);
const DANGER: Color32 = Color32::from_rgb(0xd4, 0x43, 0x32);
const SUCCESS: Color32 = Color32::from_rgb(0x4a, 0xd6, 0x9c);
const TEXT: Color32 = Color32::from_rgb(0xf4, 0xf5, 0xf8);
const TEXT_DIM: Color32 = Color32::from_rgb(0x91, 0x97, 0xa8);
const TEXT_FAINT: Color32 = Color32::from_rgb(0x62, 0x68, 0x78);
const TITLE_BLUE: Color32 = Color32::from_rgb(0x78, 0xa9, 0xff);

/// Input source selection.
#[derive(Clone, PartialEq)]
pub enum InputSource {
    None,
    Webcam(usize),
    Photo(PathBuf),
}

/// Output destination.
#[derive(Clone, PartialEq)]
pub enum OutputDest {
    VirtualCamera(u32),
    File(PathBuf),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StartIntent {
    CameraPreview,
    FaceSwap,
}

fn start_intent(input: &InputSource, target_face: Option<&Path>) -> Option<StartIntent> {
    match (input, target_face) {
        (InputSource::Webcam(_), None) => Some(StartIntent::CameraPreview),
        (InputSource::Webcam(_) | InputSource::Photo(_), Some(_)) => Some(StartIntent::FaceSwap),
        (InputSource::None, _) | (InputSource::Photo(_), None) => None,
    }
}

/// A frame the worker thread sends back to the UI thread.
enum WorkerMsg {
    /// A processed RGB frame (HWC u8) — for preview.
    Frame(Vec<u8>, u32, u32),
    /// Status update line.
    Status(String),
    /// FPS readout.
    Fps(f32),
    /// Faces detected this tick.
    FaceCount(usize),
    /// Terminal: worker exited (clean or error).
    Done(anyhow::Result<()>),
}

#[derive(Default)]
struct ImagePickerHistory {
    directory: Option<PathBuf>,
}

impl ImagePickerHistory {
    fn directory(&self) -> Option<&Path> {
        self.directory.as_deref()
    }

    fn dialog(&self) -> rfd::FileDialog {
        let dialog =
            rfd::FileDialog::new().add_filter("Image", &["jpg", "jpeg", "png", "bmp", "webp"]);
        match self.directory() {
            Some(directory) => dialog.set_directory(directory),
            None => dialog,
        }
    }

    fn remember_selection(&mut self, path: &Path) {
        if let Some(directory) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            self.directory = Some(directory.to_path_buf());
        }
    }
}

#[derive(Debug)]
struct OutputViewport {
    zoom: f32,
    pan: egui::Vec2,
    fitted: bool,
}

impl Default for OutputViewport {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            fitted: true,
        }
    }
}

impl OutputViewport {
    const MIN_ZOOM: f32 = 0.25;
    const MAX_ZOOM: f32 = 8.0;

    fn fit_scale(viewport_size: egui::Vec2, image_size: egui::Vec2) -> f32 {
        if viewport_size.x <= 0.0
            || viewport_size.y <= 0.0
            || image_size.x <= 0.0
            || image_size.y <= 0.0
        {
            return 1.0;
        }
        (viewport_size.x / image_size.x).min(viewport_size.y / image_size.y)
    }

    fn scale(&self, viewport_size: egui::Vec2, image_size: egui::Vec2) -> f32 {
        if self.fitted {
            Self::fit_scale(viewport_size, image_size)
        } else {
            self.zoom
        }
    }

    fn image_point_at(
        &self,
        viewport: egui::Rect,
        image_size: egui::Vec2,
        pointer: egui::Pos2,
    ) -> egui::Pos2 {
        let scale = self.scale(viewport.size(), image_size);
        let offset = pointer - (viewport.center() + self.pan);
        egui::Pos2::ZERO + offset / scale + image_size * 0.5
    }

    fn set_zoom_around(
        &mut self,
        viewport: egui::Rect,
        image_size: egui::Vec2,
        pointer: egui::Pos2,
        zoom: f32,
    ) {
        let anchor = self.image_point_at(viewport, image_size, pointer);
        self.fitted = false;
        self.zoom = zoom.clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        self.pan = pointer
            - viewport.center()
            - (anchor - egui::Pos2::ZERO - image_size * 0.5) * self.zoom;
    }

    fn toggle_zoom(&mut self, viewport: egui::Rect, image_size: egui::Vec2, pointer: egui::Pos2) {
        if self.fitted {
            self.set_zoom_around(viewport, image_size, pointer, 1.0);
        } else {
            self.reset();
        }
    }

    fn zoom_by(
        &mut self,
        viewport: egui::Rect,
        image_size: egui::Vec2,
        pointer: egui::Pos2,
        factor: f32,
    ) {
        let next = self.scale(viewport.size(), image_size) * factor;
        self.set_zoom_around(viewport, image_size, pointer, next);
    }

    fn pan_by(&mut self, delta: egui::Vec2) {
        if !self.fitted {
            self.pan += delta;
        }
    }

    fn reset(&mut self) {
        self.zoom = 1.0;
        self.pan = egui::Vec2::ZERO;
        self.fitted = true;
    }

    fn image_rect(&self, viewport: egui::Rect, image_size: egui::Vec2) -> egui::Rect {
        let size = image_size * self.scale(viewport.size(), image_size);
        egui::Rect::from_center_size(viewport.center() + self.pan, size)
    }
}

/// Application state.
pub struct App {
    input_source: InputSource,
    input_preview: Option<egui::ColorImage>,
    input_texture: Option<egui::TextureHandle>,
    target_face_path: Option<PathBuf>,
    target_face_preview: Option<egui::ColorImage>,
    target_face_texture: Option<egui::TextureHandle>,
    /// Preview of the swapped output (what the pipeline produces).
    output_preview: Option<egui::ColorImage>,
    output_texture: Option<egui::TextureHandle>,
    output_gpu_texture: Option<egui::load::SizedTexture>,
    output_frame: Option<(Vec<u8>, u32, u32)>,
    output_zoom_open: bool,
    output_viewport: OutputViewport,
    image_picker_history: ImagePickerHistory,
    output_dest: OutputDest,
    provider: ExecutionProvider,
    running: bool,
    fps: f32,
    face_count: usize,
    status: String,
    models_dir: PathBuf,
    models_loaded: bool,
    params: FaceSwapParams,
    /// Lazily-initialized GPU context (lives for the app's lifetime once created).
    gpu: Option<Arc<GpuOps>>,
    #[cfg(target_os = "linux")]
    gpu_preview: Option<Arc<crate::gpu_preview::LinuxPreviewBridge>>,
    #[cfg(target_os = "linux")]
    render_state: Option<egui_wgpu::RenderState>,
    /// Worker thread handle (None when idle).
    worker: Option<thread::JoinHandle<()>>,
    /// Channel from worker → UI.
    msg_rx: Option<Receiver<WorkerMsg>>,
    /// Clone of the sender handed to each new worker.
    msg_tx: Option<std::sync::mpsc::SyncSender<WorkerMsg>>,
    stop_flag: Arc<AtomicBool>,
    realtime_preview: RealtimePreviewState,
    webcam_presets: Vec<crate::io::webcam::WebcamPreset>,
    webcam_preset: usize,
    webcam_probe_rx: Option<Receiver<Vec<crate::io::webcam::WebcamPreset>>>,
}

impl App {
    pub fn new(models_dir: PathBuf) -> Self {
        Self::new_with_render_state(models_dir, None)
    }

    fn new_with_render_state(
        models_dir: PathBuf,
        render_state: Option<egui_wgpu::RenderState>,
    ) -> Self {
        let (webcam_probe_tx, webcam_probe_rx) = mpsc::channel();
        thread::spawn(move || {
            let _ = webcam_probe_tx.send(crate::io::webcam::WebcamSource::probe_all());
        });
        Self {
            input_source: InputSource::None,
            input_preview: None,
            input_texture: None,
            target_face_path: None,
            target_face_preview: None,
            target_face_texture: None,
            output_preview: None,
            output_texture: None,
            output_gpu_texture: None,
            output_frame: None,
            output_zoom_open: false,
            output_viewport: OutputViewport::default(),
            image_picker_history: ImagePickerHistory::default(),
            output_dest: OutputDest::VirtualCamera(10),
            provider: ExecutionProvider::Cuda,
            running: false,
            fps: 0.0,
            face_count: 0,
            status: "Ready".to_string(),
            models_dir,
            models_loaded: false,
            params: FaceSwapParams::default(),
            gpu: None,
            #[cfg(target_os = "linux")]
            gpu_preview: None,
            #[cfg(target_os = "linux")]
            render_state,
            worker: None,
            msg_rx: None,
            msg_tx: None,
            stop_flag: Arc::new(AtomicBool::new(false)),
            realtime_preview: RealtimePreviewState::default(),
            webcam_presets: vec![crate::io::webcam::WebcamPreset {
                camera_index: 0,
                camera_name: "Probing cameras…".to_owned(),
                width: 1280,
                height: 720,
                fps: 30,
            }],
            webcam_preset: 0,
            webcam_probe_rx: Some(webcam_probe_rx),
        }
    }

    fn poll_webcam_probe(&mut self) {
        let Some(receiver) = &self.webcam_probe_rx else {
            return;
        };
        match receiver.try_recv() {
            Ok(presets) => {
                self.webcam_presets = presets;
                self.webcam_preset = 0;
                if matches!(self.input_source, InputSource::Webcam(_))
                    && let Some(preset) = self.webcam_presets.first()
                {
                    self.input_source = InputSource::Webcam(preset.camera_index);
                }
                self.webcam_probe_rx = None;
            }
            Err(mpsc::TryRecvError::Disconnected) => self.webcam_probe_rx = None,
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }

    fn load_models(&mut self) {
        self.status = "Loading models".to_string();
        let required = [
            "yoloface_8n.onnx",
            "w600k_r50.onnx",
            CANONICAL_SWAPPER_FILENAME,
        ];
        let mut missing = Vec::new();
        for name in &required {
            let path = self.models_dir.join(name);
            if !path.exists() {
                missing.push(name.to_string());
            }
        }
        if missing.is_empty() {
            self.models_loaded = true;
            self.status = "Models ready".to_string();
        } else {
            self.status = format!("Missing: {}", missing.join(", "));
        }
    }

    /// Lazily initialize the GPU context + stream. Done once, reused.
    fn ensure_gpu(&mut self) -> anyhow::Result<Arc<GpuOps>> {
        if let Some(g) = &self.gpu {
            return Ok(g.clone());
        }
        let ctx = Arc::new(CudaContext::new(0)?);
        let stream = ctx.default_stream().clone();
        let gpu = Arc::new(GpuOps::new(&ctx, stream.clone())?);
        self.gpu = Some(gpu.clone());
        Ok(gpu)
    }

    fn begin_realtime_preview(&mut self, source: &InputSource) {
        if matches!(source, InputSource::Webcam(_)) {
            self.fps = 0.0;
            self.face_count = 0;
            self.output_preview = None;
            self.output_texture = None;
            self.output_gpu_texture = None;
            self.output_frame = None;
            self.realtime_preview.open();
        }
    }

    fn apply_realtime_preview_close_request(&mut self) {
        self.realtime_preview.apply_close_request();
    }

    /// Start processing. Spawns a worker thread for photo or webcam input.
    fn start(&mut self, ctx: &egui::Context) {
        if self.running {
            return;
        }
        let Some(intent) = start_intent(&self.input_source, self.target_face_path.as_deref())
        else {
            self.status = if matches!(self.input_source, InputSource::None) {
                "Select an input source first".to_string()
            } else {
                "Select a source face first".to_string()
            };
            return;
        };
        if intent == StartIntent::FaceSwap && !self.models_loaded {
            self.load_models();
            if !self.models_loaded {
                return;
            }
        }

        // Set up channel.
        let (tx, rx) = mpsc::sync_channel::<WorkerMsg>(3);
        self.msg_rx = Some(rx);
        self.msg_tx = Some(tx.clone());
        let gpu = match intent {
            StartIntent::FaceSwap => match self.ensure_gpu() {
                Ok(gpu) => Some(gpu),
                Err(error) => {
                    self.status = format!("GPU init failed: {error}");
                    return;
                }
            },
            StartIntent::CameraPreview => None,
        };
        self.stop_flag.store(false, Ordering::Release);
        let stop_flag = self.stop_flag.clone();
        let target_path = self.target_face_path.clone();
        let models_dir = self.models_dir.clone();
        let params = self.params.clone();
        let output_dest = self.output_dest.clone();
        let provider = self.provider;
        let ctx2 = ctx.clone();
        let input = self.input_source.clone();
        let webcam_preset = self.webcam_presets.get(self.webcam_preset).cloned();
        #[cfg(target_os = "linux")]
        let gpu_preview =
            if intent == StartIntent::FaceSwap && matches!(&input, InputSource::Webcam(_)) {
                self.render_state
                    .as_ref()
                    .zip(webcam_preset.as_ref())
                    .and_then(|(render_state, preset)| {
                        let geometry =
                            crate::gpu_preview::PreviewGeometry::new(preset.width, preset.height)?;
                        match crate::gpu_preview::LinuxPreviewBridge::new(render_state, geometry) {
                            Ok(bridge) => Some(bridge),
                            Err(error) => {
                                tracing::warn!("GPU preview disabled: {error:#}");
                                None
                            }
                        }
                    })
            } else {
                None
            };
        #[cfg(target_os = "linux")]
        {
            self.gpu_preview = gpu_preview.clone();
            self.output_gpu_texture = gpu_preview
                .as_ref()
                .map(|bridge| egui::load::SizedTexture::new(bridge.texture_id(), bridge.size()));
        }
        let preview_input = input.clone();
        let handle = match input {
            InputSource::Photo(path) => thread::spawn(move || {
                let gpu = gpu.expect("face-swap intent always initializes CUDA");
                let target_path = target_path.expect("photo face-swap intent requires a face");
                let stream = gpu.stream.clone();
                let result = run_photo_swap(
                    gpu,
                    models_dir,
                    path,
                    target_path,
                    params,
                    provider,
                    output_dest,
                    tx.clone(),
                    stream.clone(),
                );
                send_worker_done(&tx, &ctx2, result);
            }),
            InputSource::Webcam(idx) => thread::spawn(move || {
                let completion_ctx = ctx2.clone();
                let preset = webcam_preset.unwrap_or(crate::io::webcam::WebcamPreset {
                    camera_index: idx,
                    camera_name: format!("Camera {idx}"),
                    width: 1280,
                    height: 720,
                    fps: 30,
                });
                let result = match intent {
                    StartIntent::CameraPreview => run_webcam_preview_loop(
                        preset.camera_index,
                        preset.width,
                        preset.height,
                        preset.fps,
                        output_dest,
                        tx.clone(),
                        ctx2,
                        stop_flag,
                    ),
                    StartIntent::FaceSwap => {
                        let gpu = gpu.expect("face-swap intent always initializes CUDA");
                        let stream = gpu.stream.clone();
                        run_webcam_loop(
                            gpu,
                            models_dir,
                            preset.camera_index,
                            preset.width,
                            preset.height,
                            preset.fps,
                            target_path.expect("webcam face-swap intent requires a face"),
                            params,
                            provider,
                            output_dest,
                            tx.clone(),
                            ctx2,
                            stream,
                            stop_flag,
                            #[cfg(target_os = "linux")]
                            gpu_preview,
                        )
                    }
                };
                send_worker_done(&tx, &completion_ctx, result);
            }),
            InputSource::None => return,
        };

        self.worker = Some(handle);
        self.running = true;
        self.status = "Running".to_string();
        self.begin_realtime_preview(&preview_input);
    }

    fn stop(&mut self, ctx: &egui::Context) {
        self.stop_flag.store(true, Ordering::Release);
        if let Some(_tx) = &self.msg_tx {
            // Drop the sender → worker's rx.recv() returns Err and the loop exits.
            // For webcam the worker checks tx.is_empty periodically.
        }
        self.msg_tx = None; // drop our end → channel disconnects
        self.running = false;
        self.status = "Stopped".to_string();
        self.realtime_preview.close(ctx);
    }

    /// Drain the worker channel and update state. Called every frame.
    fn poll_worker(&mut self, ctx: &egui::Context) {
        #[cfg(target_os = "linux")]
        if let Some(preview) = &self.gpu_preview
            && preview.consume_latest()
        {
            self.output_gpu_texture = Some(egui::load::SizedTexture::new(
                preview.texture_id(),
                preview.size(),
            ));
            self.output_viewport.reset();
        }
        let Some(rx) = &self.msg_rx else { return };
        let mut latest_frame = None;
        while let Ok(msg) = rx.try_recv() {
            match msg {
                WorkerMsg::Frame(data, w, h) => latest_frame = Some((data, w, h)),
                WorkerMsg::Status(s) => self.status = s,
                WorkerMsg::Fps(f) => self.fps = f,
                WorkerMsg::FaceCount(n) => self.face_count = n,
                WorkerMsg::Done(res) => {
                    self.running = false;
                    self.realtime_preview.close(ctx);
                    if let Err(e) = res {
                        self.status = format!("Error: {e}");
                    } else if !self.status.starts_with("Error") {
                        self.status = "Done".to_string();
                    }
                    self.worker = None;
                }
            }
        }
        if let Some((data, width, height)) = latest_frame {
            self.output_preview = color_preview_from_rgb(&data, width, height, 1024).ok();
            self.output_frame = Some((data, width, height));
            self.output_texture = None;
            self.output_gpu_texture = None;
            self.output_viewport.reset();
        }
        ctx.request_repaint();
    }

    fn output_texture(&mut self, ctx: &egui::Context) -> Option<egui::TextureHandle> {
        if self.output_texture.is_none()
            && let Some(preview) = &self.output_preview
        {
            self.output_texture =
                Some(ctx.load_texture("output", preview.clone(), egui::TextureOptions::LINEAR));
        }
        self.output_texture.clone()
    }

    fn show_realtime_preview(&mut self, ctx: &egui::Context) {
        self.apply_realtime_preview_close_request();
        if !self.realtime_preview.is_open() {
            return;
        }
        let snapshot = RealtimePreviewSnapshot {
            texture: self.output_gpu_texture.or_else(|| {
                self.output_texture(ctx)
                    .as_ref()
                    .map(egui::load::SizedTexture::from_handle)
            }),
            fps: self.fps,
            face_count: self.face_count,
            provider: self.provider,
            status: self.status.clone(),
        };
        realtime_preview::show(ctx, &self.realtime_preview, snapshot);
    }

    fn save_output(&mut self) {
        let Some((data, width, height)) = &self.output_frame else {
            self.status = "Process a photo or camera frame first".to_owned();
            return;
        };
        let Some(path) = rfd::FileDialog::new()
            .add_filter("PNG image", &["png"])
            .add_filter("JPEG image", &["jpg", "jpeg"])
            .set_file_name("swapped_output.png")
            .save_file()
        else {
            return;
        };
        match image::save_buffer(&path, data, *width, *height, image::ColorType::Rgb8) {
            Ok(()) => self.status = format!("Saved {}", path.display()),
            Err(error) => self.status = format!("Error: {error}"),
        }
    }

    fn output_image(&mut self, ui: &mut egui::Ui, max_size: egui::Vec2) -> Option<egui::Response> {
        let texture = self.output_texture(ui.ctx())?;
        Some(
            ui.add(
                egui::Image::new(egui::load::SizedTexture::from_handle(&texture))
                    .max_size(max_size)
                    .sense(egui::Sense::click())
                    .alt_text("Output image"),
            )
            .on_hover_cursor(egui::CursorIcon::PointingHand)
            .on_hover_text("Click to inspect"),
        )
    }

    fn output_zoom(&mut self, ctx: &egui::Context) {
        if !self.output_zoom_open {
            return;
        }
        let Some(texture) = self.output_texture(ctx) else {
            self.output_zoom_open = false;
            return;
        };
        let mut open = true;
        let mut close_requested = false;
        egui::Window::new("Output image")
            .open(&mut open)
            .resizable(true)
            .collapsible(false)
            .default_size(egui::vec2(900.0, 650.0))
            .show(ctx, |ui| {
                let available = ui.available_size() - egui::vec2(0.0, 38.0);
                let viewport_size = available.max(egui::vec2(320.0, 240.0));
                let (viewport, response) =
                    ui.allocate_exact_size(viewport_size, egui::Sense::click_and_drag());
                let image_size = texture.size_vec2();

                if response.double_clicked() {
                    self.output_viewport.reset();
                } else if response.clicked()
                    && let Some(pointer) = response.interact_pointer_pos()
                {
                    self.output_viewport
                        .toggle_zoom(viewport, image_size, pointer);
                }
                if response.dragged() {
                    let delta = ui.input(|input| input.pointer.delta());
                    self.output_viewport.pan_by(delta);
                }
                if response.hovered() {
                    let scroll = ui.input(|input| input.smooth_scroll_delta.y);
                    if scroll != 0.0 {
                        let pointer = response.hover_pos().unwrap_or(viewport.center());
                        let factor = (scroll * 0.0025).exp();
                        self.output_viewport
                            .zoom_by(viewport, image_size, pointer, factor);
                        ui.input_mut(|input| input.smooth_scroll_delta = egui::Vec2::ZERO);
                    }
                    ui.output_mut(|output| {
                        output.cursor_icon = if response.dragged() {
                            egui::CursorIcon::Grabbing
                        } else if self.output_viewport.fitted {
                            egui::CursorIcon::ZoomIn
                        } else {
                            egui::CursorIcon::ZoomOut
                        };
                    });
                }

                ui.painter_at(viewport).image(
                    texture.id(),
                    self.output_viewport.image_rect(viewport, image_size),
                    egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                    Color32::WHITE,
                );

                ui.horizontal(|ui| {
                    if ui.button("Close preview").clicked() {
                        close_requested = true;
                    }
                    if ui.button("Fit").clicked() {
                        self.output_viewport.reset();
                    }
                    ui.label(format!(
                        "{:.0}%",
                        self.output_viewport.scale(viewport.size(), image_size) * 100.0
                    ));
                });
            });
        self.output_zoom_open = open && !close_requested;
    }

    #[allow(dead_code)] // Kept for the legacy layout retained below.
    fn swap_paths(&mut self) {
        // Swap source ↔ target if both are image paths. Clone before assign
        // to avoid borrow conflict on self.input_source.
        if let (InputSource::Photo(src), Some(tgt)) = (&self.input_source, &self.target_face_path) {
            let new_target = src.clone();
            let new_source = tgt.clone();
            self.input_source = InputSource::Photo(new_source);
            self.target_face_path = Some(new_target);
            if let InputSource::Photo(path) = &self.input_source
                && let Ok(preview) = load_preview(path)
            {
                self.input_preview = Some(preview);
                self.input_texture = None;
            }
            if let Ok(preview) = load_preview(self.target_face_path.as_ref().unwrap()) {
                self.target_face_preview = Some(preview);
                self.target_face_texture = None;
            }
        }
    }

    fn redesigned_ui(&mut self, ui: &mut egui::Ui) {
        ui.spacing_mut().item_spacing = egui::vec2(8.0, 6.0);
        ui.spacing_mut().button_padding = egui::vec2(10.0, 6.0);

        // Responsive shell: cards consume the current viewport instead of
        // targeting one desktop size. Only controls/previews keep sensible
        // minimum dimensions; the layout itself follows the window.
        let content_width = ui.available_width();
        ui.vertical_centered(|ui| {
            ui.set_width(content_width);

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new("noperson")
                            .color(TEXT)
                            .strong()
                            .size(19.0),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (state, label) = if self.running {
                        (IndicatorState::On, format!("LIVE · {:.1} FPS", self.fps))
                    } else if self.models_loaded {
                        (IndicatorState::On, "READY".to_owned())
                    } else {
                        (
                            IndicatorState::Connecting,
                            match self.provider {
                                ExecutionProvider::Cuda => "NATIVE CUDA · LOCAL",
                                ExecutionProvider::TensorRT => "TENSORRT · LOCAL",
                            }
                            .to_owned(),
                        )
                    };
                    ui.add(StatusPill::new().item(label, state));
                });
            });

            ui.add_space(8.0);
            ui.columns(2, |columns| {
                let selected = self
                    .target_face_path
                    .as_ref()
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().into_owned());
                selector_card(
                    &mut columns[0],
                    "Source face",
                    self.target_face_path.is_some(),
                    |ui| {
                        if let Some(texture) = &self.target_face_texture {
                            ui.add(
                                egui::Image::new(egui::load::SizedTexture::from_handle(texture))
                                    .max_size(egui::vec2(170.0, 94.0)),
                            );
                        } else if let Some(preview) = &self.target_face_preview {
                            let texture = ui.ctx().load_texture(
                                "identity",
                                preview.clone(),
                                egui::TextureOptions::LINEAR,
                            );
                            self.target_face_texture = Some(texture.clone());
                            ui.add(
                                egui::Image::new(egui::load::SizedTexture::from_handle(&texture))
                                    .max_size(egui::vec2(170.0, 94.0)),
                            );
                        } else {
                            empty_preview(ui, "SOURCE FACE");
                        }
                        ui.label(
                            egui::RichText::new(selected.as_deref().unwrap_or("No face selected"))
                                .color(if selected.is_some() { TEXT } else { TEXT_DIM })
                                .size(11.0),
                        );
                        let label = if selected.is_some() {
                            "Change face"
                        } else {
                            "Select a face"
                        };
                        if ui.add(EleganceButton::new(label).full_width()).clicked()
                            && let Some(path) = self.image_picker_history.dialog().pick_file()
                        {
                            self.image_picker_history.remember_selection(&path);
                            self.target_face_path = Some(path.clone());
                            match load_preview(&path) {
                                Ok(preview) => {
                                    self.target_face_preview = Some(preview);
                                    self.target_face_texture = None;
                                }
                                Err(e) => self.status = format!("Error: {e}"),
                            }
                        }
                        if selected.is_some()
                            && ui
                                .add(
                                    EleganceButton::new("Clear face")
                                        .outline()
                                        .full_width()
                                        .enabled(!self.running),
                                )
                                .clicked()
                        {
                            self.target_face_path = None;
                            self.target_face_preview = None;
                            self.target_face_texture = None;
                            self.status = "Camera preview ready".to_owned();
                        }
                    },
                );

                let target_ready = !matches!(self.input_source, InputSource::None);
                let target_label = match &self.input_source {
                    InputSource::None => "No target selected".to_owned(),
                    InputSource::Photo(p) => p
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned(),
                    InputSource::Webcam(i) => format!("Camera {i}"),
                };
                selector_card(&mut columns[1], "Target", target_ready, |ui| {
                    if matches!(self.input_source, InputSource::Photo(_)) {
                        if let Some(texture) = &self.input_texture {
                            ui.add(
                                egui::Image::new(egui::load::SizedTexture::from_handle(texture))
                                    .max_size(egui::vec2(170.0, 94.0)),
                            );
                        } else if let Some(preview) = &self.input_preview {
                            let texture = ui.ctx().load_texture(
                                "target_input",
                                preview.clone(),
                                egui::TextureOptions::LINEAR,
                            );
                            self.input_texture = Some(texture.clone());
                            ui.add(
                                egui::Image::new(egui::load::SizedTexture::from_handle(&texture))
                                    .max_size(egui::vec2(170.0, 94.0)),
                            );
                        } else {
                            empty_preview(ui, "PHOTO");
                        }
                    } else {
                        empty_preview(
                            ui,
                            match self.input_source {
                                InputSource::Webcam(_) => "CAMERA",
                                InputSource::None => "TARGET",
                                InputSource::Photo(_) => unreachable!(),
                            },
                        );
                    }
                    ui.label(
                        egui::RichText::new(target_label)
                            .color(if target_ready { TEXT } else { TEXT_DIM })
                            .size(11.0),
                    );
                    ui.columns(2, |buttons| {
                        if buttons[0]
                            .add(EleganceButton::new("Photo").full_width())
                            .clicked()
                            && let Some(path) = self.image_picker_history.dialog().pick_file()
                        {
                            self.image_picker_history.remember_selection(&path);
                            match load_preview(&path) {
                                Ok(preview) => {
                                    self.input_source = InputSource::Photo(path);
                                    self.output_dest = OutputDest::File(PathBuf::new());
                                    self.input_preview = Some(preview);
                                    self.input_texture = None;
                                }
                                Err(error) => self.status = format!("Error: {error}"),
                            }
                        }
                        if buttons[1]
                            .add(EleganceButton::new("Camera").outline().full_width())
                            .clicked()
                        {
                            self.input_source = InputSource::Webcam(0);
                            self.output_dest = OutputDest::VirtualCamera(10);
                            self.input_preview = None;
                            self.input_texture = None;
                        }
                    });
                });
            });

            ui.add_space(6.0);
            settings_card(ui, "Live setup", |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Provider").color(TEXT_DIM));
                    ui.add_enabled_ui(!self.running, |ui| {
                        let mut provider = match self.provider {
                            ExecutionProvider::Cuda => 0,
                            ExecutionProvider::TensorRT => 1,
                        };
                        if ui
                            .add(
                                SegmentedControl::new(&mut provider, ["Native CUDA", "TensorRT"])
                                    .size(SegmentedSize::Small),
                            )
                            .changed()
                        {
                            self.provider = if provider == 0 {
                                ExecutionProvider::Cuda
                            } else {
                                ExecutionProvider::TensorRT
                            };
                        }
                    });
                    ui.separator();
                    toggle(ui, "Face restorer", &mut self.params.restorer_enabled);
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("Swap resolution").color(TEXT_DIM));
                    let mut resolution = match self.params.dim {
                        SwapDim::Dim1 => 0,
                        SwapDim::Dim2 => 1,
                        SwapDim::Dim3 => 2,
                        SwapDim::Dim4 => 3,
                    };
                    if ui
                        .add(
                            SegmentedControl::new(&mut resolution, ["128", "256", "384", "512"])
                                .size(SegmentedSize::Small),
                        )
                        .changed()
                    {
                        self.params.dim = match resolution {
                            0 => SwapDim::Dim1,
                            1 => SwapDim::Dim2,
                            2 => SwapDim::Dim3,
                            _ => SwapDim::Dim4,
                        };
                    }
                    if self.params.restorer_enabled {
                        ui.separator();
                        ui.label(egui::RichText::new("Hot path").color(TEXT_DIM));
                        ui.selectable_value(
                            &mut self.params.restorer_mode,
                            RestorerMode::Realtime,
                            "Realtime",
                        );
                        ui.selectable_value(
                            &mut self.params.restorer_mode,
                            RestorerMode::Quality,
                            "Quality",
                        );
                    }
                });
                if matches!(self.input_source, InputSource::Webcam(_)) {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Camera mode").color(TEXT_DIM));
                        let selected = self
                            .webcam_presets
                            .get(self.webcam_preset)
                            .map_or_else(|| "Camera 0: 720p30".to_owned(), |preset| preset.label());
                        egui::ComboBox::from_id_salt("realtime-camera-preset")
                            .selected_text(selected)
                            .show_ui(ui, |ui| {
                                for (preset_index, preset) in self.webcam_presets.iter().enumerate()
                                {
                                    if ui
                                        .selectable_value(
                                            &mut self.webcam_preset,
                                            preset_index,
                                            preset.label(),
                                        )
                                        .changed()
                                    {
                                        self.input_source =
                                            InputSource::Webcam(preset.camera_index);
                                    }
                                }
                            });
                        ui.label(
                            egui::RichText::new("Output: /dev/video10")
                                .color(TEXT_FAINT)
                                .size(10.0),
                        );
                    });
                }
            });

            ui.add_space(6.0);
            settings_card(ui, "Refinement", |ui| {
                labeled_slider(ui, "Strength", &mut self.params.strength, 0.0, 1.0);
                labeled_u32_slider(ui, "Edge blur", &mut self.params.border_blur, 0, 100);
                if self.params.restorer_enabled {
                    labeled_slider(
                        ui,
                        "Restorer blend",
                        &mut self.params.restorer_alpha,
                        0.0,
                        1.0,
                    );
                }
                egui::CollapsingHeader::new("Advanced settings")
                    .default_open(false)
                    .show(ui, |ui| {
                        labeled_slider(
                            ui,
                            "Similarity threshold",
                            &mut self.params.similarity_threshold,
                            0.0,
                            1.0,
                        );
                        labeled_u32_slider(ui, "Border top", &mut self.params.border_top, 0, 100);
                        labeled_u32_slider(
                            ui,
                            "Border bottom",
                            &mut self.params.border_bottom,
                            0,
                            100,
                        );
                        labeled_u32_slider(ui, "Border left", &mut self.params.border_left, 0, 100);
                        labeled_u32_slider(
                            ui,
                            "Border right",
                            &mut self.params.border_right,
                            0,
                            100,
                        );
                    });
            });

            ui.add_space(6.0);
            settings_card(ui, "Output preview", |ui| {
                if self.output_preview.is_some() || self.output_texture.is_some() {
                    if self
                        .output_image(ui, egui::vec2(ui.available_width(), 130.0))
                        .is_some_and(|response| response.clicked())
                    {
                        self.output_viewport.reset();
                        self.output_zoom_open = true;
                    }
                } else {
                    let width = ui.available_width();
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(width, 130.0), egui::Sense::hover());
                    ui.painter().rect_filled(rect, 9.0, BG);
                    ui.painter().text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "Processed photo or live camera preview appears here",
                        egui::FontId::proportional(11.0),
                        TEXT_FAINT,
                    );
                }
                let can_save = self.output_frame.is_some();
                if ui
                    .add(
                        EleganceButton::new("Save output")
                            .outline()
                            .full_width()
                            .enabled(can_save),
                    )
                    .clicked()
                {
                    self.save_output();
                }
            });
        });
    }

    fn action_bar(&mut self, ui: &mut egui::Ui) {
        let intent = start_intent(&self.input_source, self.target_face_path.as_deref());
        let ready = intent.is_some();
        let action_label = if self.running {
            "Stop live"
        } else if intent == Some(StartIntent::CameraPreview) {
            "Preview camera"
        } else if matches!(self.input_source, InputSource::Photo(_)) {
            "Process photo"
        } else {
            "Start live"
        };
        let action = ui.add(
            EleganceButton::new(action_label)
                .accent(if self.running {
                    EleganceAccent::Red
                } else {
                    EleganceAccent::Blue
                })
                .size(ButtonSize::Large)
                .full_width()
                .enabled(self.running || ready),
        );
        if action.clicked() {
            if self.running {
                self.stop(ui.ctx());
            } else {
                self.start(ui.ctx());
            }
        }
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(if ready {
                    &self.status
                } else if matches!(self.input_source, InputSource::Webcam(_)) {
                    "Camera preview is available without a source face"
                } else {
                    "Select a source face and target"
                })
                .color(if self.status.starts_with("Error") {
                    DANGER
                } else {
                    TEXT_DIM
                })
                .size(11.0),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(match &self.output_dest {
                        OutputDest::VirtualCamera(device) => format!("/dev/video{device}"),
                        OutputDest::File(_) => "swapped_output.png".to_owned(),
                    })
                    .color(TEXT_FAINT)
                    .size(10.0),
                );
            });
        });
    }

    fn render_ui(&mut self, ui: &mut egui::Ui) {
        install_product_theme(ui.ctx());
        let footer_height = 76.0;
        let body_size = egui::vec2(
            ui.available_width(),
            (ui.available_height() - footer_height).max(240.0),
        );
        ui.allocate_ui(body_size, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| self.redesigned_ui(ui));
        });
        ui.add_space(6.0);
        self.action_bar(ui);
    }
}

pub(crate) fn install_product_theme(ctx: &egui::Context) {
    ctx.add_font(FontInsert::new(
        "Inter",
        FontData::from_static(include_bytes!("../assets/fonts/InterVariable.ttf")),
        vec![InsertFontFamily {
            family: FontFamily::Proportional,
            priority: FontPriority::Highest,
        }],
    ));

    let mut theme = Theme::charcoal();
    theme.palette.bg = BG;
    theme.palette.card = CARD;
    theme.palette.input_bg = BG;
    theme.palette.border = CARD_BORDER;
    theme.palette.text = TEXT;
    theme.palette.text_muted = TEXT_DIM;
    theme.palette.text_faint = TEXT_FAINT;
    theme.palette.blue = ACCENT;
    theme.palette.focus = TITLE_BLUE;
    theme.palette.success = SUCCESS;
    theme.palette.danger = DANGER;
    theme.card_radius = 10.0;
    theme.card_padding = 9.0;
    theme.install(ctx);
}

fn empty_preview(ui: &mut egui::Ui, label: &str) {
    let width = ui.available_width().min(170.0);
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 94.0), egui::Sense::hover());
    ui.painter().rect_filled(rect, 9.0, CARD_HOVER);
    ui.painter().rect_stroke(
        rect,
        9.0,
        egui::Stroke::new(1.0, CARD_BORDER),
        egui::StrokeKind::Inside,
    );
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(10.0),
        TEXT_FAINT,
    );
}

fn selector_card(ui: &mut egui::Ui, title: &str, ready: bool, add: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::NONE
        .fill(CARD)
        .stroke(egui::Stroke::new(
            1.0,
            if ready { ACCENT } else { CARD_BORDER },
        ))
        .corner_radius(10.0)
        .inner_margin(egui::Margin::same(9))
        .show(ui, |ui| {
            ui.set_min_height(176.0);
            ui.label(
                egui::RichText::new(title)
                    .color(TITLE_BLUE)
                    .strong()
                    .size(14.0),
            );
            ui.separator();
            ui.vertical_centered(add);
        });
}

fn settings_card(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    EleganceCard::new()
        .heading(title)
        .padding(9.0)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui);
        });
}

// ─── Card helper ───────────────────────────────────────────────────────────
#[allow(dead_code)] // Kept for the legacy layout retained below.
fn card(ui: &mut egui::Ui, title: &str, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::NONE
        .fill(CARD)
        .stroke(egui::Stroke::new(1.0, CARD_BORDER))
        .corner_radius(10.0)
        .inner_margin(egui::Margin::same(14))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 8.0;
            ui.label(egui::RichText::new(title).color(TITLE_BLUE).strong());
            ui.separator();
            add_contents(ui);
        });
}

/// Styled toggle switch (checkbox replacement).
fn toggle(ui: &mut egui::Ui, label: &str, value: &mut bool) -> egui::Response {
    ui.add(Switch::new(value, label).accent(EleganceAccent::Sky))
}

/// Styled slider with label.
fn labeled_slider(ui: &mut egui::Ui, label: &str, value: &mut f32, min: f32, max: f32) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).color(TEXT_DIM));
        ui.add(
            egui::Slider::new(value, min..=max)
                .text("")
                .fixed_decimals(2),
        );
    });
}

fn labeled_u32_slider(ui: &mut egui::Ui, label: &str, value: &mut u32, min: u32, max: u32) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).color(TEXT_DIM));
        ui.add(egui::Slider::new(value, min..=max).text(""));
    });
}

// ─── Worker functions (run on background threads) ─────────────────────────

/// Photo path: load source image → swap once → send result back for preview.
fn run_photo_swap(
    gpu: Arc<GpuOps>,
    models_dir: PathBuf,
    source_path: PathBuf,
    target_path: PathBuf,
    params: FaceSwapParams,
    provider: ExecutionProvider,
    output_dest: OutputDest,
    tx: mpsc::SyncSender<WorkerMsg>,
    stream: Arc<cudarc::driver::CudaStream>,
) -> anyhow::Result<()> {
    use image::GenericImageView;
    let _ = tx.send(WorkerMsg::Status("Setting up target face".into()));
    let mut engine = LiveEngine::new_with_provider(
        gpu.clone(),
        &models_dir,
        &target_path,
        params,
        provider,
        &stream,
    )?;

    let source_img = image::open(&source_path)?;
    let (sw, sh) = source_img.dimensions();
    let source_rgb = source_img.to_rgb8();

    let _ = tx.send(WorkerMsg::Status("Processing".into()));
    let result = engine.process_rgb(source_rgb.as_raw(), sw, sh)?;
    let _ = tx.send(WorkerMsg::FaceCount(result.faces_detected));
    let (output_width, output_height) = (result.width, result.height);
    let output_hwc = result.data;
    let _ = tx.send(WorkerMsg::Frame(
        output_hwc.clone(),
        output_width,
        output_height,
    ));

    match output_dest {
        OutputDest::VirtualCamera(dev) => {
            let _ = tx.send(WorkerMsg::Status(format!("Opening /dev/video{dev}")));
            let mut vcam =
                VirtualCamera::open(dev, output_width.max(640), output_height.max(480), 30)?;
            for _ in 0..60 {
                if tx.send(WorkerMsg::Status("streaming".into())).is_err() {
                    break;
                }
                vcam.send_frame(&output_hwc)?;
            }
        }
        OutputDest::File(path) => {
            let out_path = if path.as_os_str().is_empty() {
                PathBuf::from("swapped_output.png")
            } else {
                path
            };
            let _ = tx.send(WorkerMsg::Status(format!("Saving {}", out_path.display())));
            image::save_buffer(
                &out_path,
                &output_hwc,
                output_width,
                output_height,
                image::ColorType::Rgb8,
            )?;
        }
    }
    let _ = tx.send(WorkerMsg::Status("Done".into()));
    Ok(())
}

/// Camera-only path used before a source identity is selected. It avoids CUDA
/// and model startup entirely while retaining the same preview and virtual
/// camera lifecycle as the face-swap worker.
#[allow(clippy::too_many_arguments)]
fn run_webcam_preview_loop(
    webcam_idx: usize,
    webcam_width: u32,
    webcam_height: u32,
    webcam_fps: u32,
    output_dest: OutputDest,
    tx: mpsc::SyncSender<WorkerMsg>,
    ctx: egui::Context,
    stop_flag: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    use crate::io::video::FrameSource;

    let _ = tx.send(WorkerMsg::Status("Opening camera preview".into()));
    let mut camera = crate::io::webcam::WebcamSource::new(
        webcam_idx,
        webcam_width,
        webcam_height,
        webcam_fps as f32,
    )?;
    let (width, height) = camera.dimensions();
    let fps = camera.fps().round().max(1.0) as u32;
    let mut virtual_camera_warning = None;
    let mut virtual_camera = if let OutputDest::VirtualCamera(device) = output_dest {
        match VirtualCamera::open(device, width, height, fps) {
            Ok(camera) => Some(camera),
            Err(error) => {
                virtual_camera_warning = Some(format!(
                    "Camera preview only; virtual camera unavailable: {error}"
                ));
                None
            }
        }
    } else {
        None
    };
    let _ = tx
        .send(WorkerMsg::Status(virtual_camera_warning.unwrap_or_else(
            || format!("Camera preview · {width}×{height}@{fps}"),
        )));

    let mut frames = 0_u64;
    let mut sample_started = std::time::Instant::now();
    let mut last_frame_at = std::time::Instant::now();
    while !stop_flag.load(Ordering::Acquire) {
        let frame = match camera.next_frame_result() {
            Ok(frame) => frame,
            Err(error) => {
                anyhow::ensure!(
                    last_frame_at.elapsed() < Duration::from_secs(5),
                    "Webcam stopped delivering frames for 5 seconds: {error}"
                );
                std::thread::sleep(Duration::from_millis(4));
                continue;
            }
        };
        last_frame_at = std::time::Instant::now();
        if let Some(camera) = &mut virtual_camera {
            camera.send_frame(&frame.data)?;
        }
        let _ = tx.try_send(WorkerMsg::Frame(frame.data, frame.width, frame.height));

        frames += 1;
        let elapsed = sample_started.elapsed();
        if elapsed >= Duration::from_millis(500) {
            let _ = tx.try_send(WorkerMsg::Fps(frames as f32 / elapsed.as_secs_f32()));
            frames = 0;
            sample_started = std::time::Instant::now();
        }
        ctx.request_repaint();
    }
    Ok(())
}

/// Webcam live path: open webcam → loop frames → swap each → vcam + preview.
fn run_webcam_loop(
    gpu: Arc<GpuOps>,
    models_dir: PathBuf,
    webcam_idx: usize,
    webcam_width: u32,
    webcam_height: u32,
    webcam_fps: u32,
    target_path: PathBuf,
    params: FaceSwapParams,
    provider: ExecutionProvider,
    output_dest: OutputDest,
    tx: mpsc::SyncSender<WorkerMsg>,
    ctx: egui::Context,
    stream: Arc<cudarc::driver::CudaStream>,
    stop_flag: Arc<AtomicBool>,
    #[cfg(target_os = "linux")] gpu_preview: Option<Arc<crate::gpu_preview::LinuxPreviewBridge>>,
) -> anyhow::Result<()> {
    let _ = tx.send(WorkerMsg::Status("Setting up target face".into()));
    let mut engine = LiveEngine::new_with_provider(
        gpu.clone(),
        &models_dir,
        &target_path,
        params.clone(),
        provider,
        &stream,
    )?;

    let _ = tx.send(WorkerMsg::Status("Opening webcam".into()));
    let (capture_worker, capture_info) = crate::io::webcam::WebcamCaptureWorker::spawn(
        webcam_idx,
        webcam_width,
        webcam_height,
        webcam_fps as f32,
        Arc::clone(&stop_flag),
    )?;
    let (camera_width, camera_height) = (capture_info.width, capture_info.height);
    let camera_fps = capture_info.fps.round().max(1.0) as u32;
    #[cfg(target_os = "linux")]
    let gpu_preview = gpu_preview.filter(|preview| {
        let geometry = preview.geometry();
        let matches = geometry.width() == camera_width && geometry.height() == camera_height;
        if !matches {
            tracing::warn!(
                "GPU preview geometry {}x{} differs from camera {}x{}; using CPU preview",
                geometry.width(),
                geometry.height(),
                camera_width,
                camera_height
            );
        }
        matches
    });
    let mut frames = FrameRing::new_for_dimensions(
        stream.context(),
        &stream,
        FrameRing::DEFAULT_CAPACITY,
        camera_width,
        camera_height,
    )?;
    stream.context().bind_to_thread()?;
    let mut nvjpeg = crate::io::nvjpeg::NvJpegDecoder::load().ok();
    let decoder_name = if nvjpeg.is_some() {
        "nvJPEG"
    } else {
        "CPU JPEG fallback"
    };
    let _ = tx.send(WorkerMsg::Status(format!(
        "Webcam opened · {camera_width}×{camera_height}@{camera_fps} · {decoder_name}"
    )));

    let mut virtual_camera_warning = None;
    let mut vcam = if let OutputDest::VirtualCamera(dev) = &output_dest {
        match VirtualCamera::open(*dev, camera_width, camera_height, camera_fps) {
            Ok(camera) => Some(camera),
            Err(error) => {
                virtual_camera_warning =
                    Some(format!("Preview only; virtual camera unavailable: {error}"));
                None
            }
        }
    } else {
        None
    };
    let save_to_file = matches!(output_dest, OutputDest::File(_));

    let mut last_t = std::time::Instant::now();
    let mut frame_n = 0u64;
    let mut warmed = false;
    let live_profiling = std::env::var_os("NOPERSON_PROFILE_GPU").is_some_and(|value| {
        !matches!(
            value.to_string_lossy().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "off"
        )
    });
    let mut live_profile_samples = 0_u32;
    let mut live_profile_ms = [0.0_f64; 5];
    let _ = tx.send(WorkerMsg::Status(
        virtual_camera_warning.unwrap_or_else(|| "Waiting for first frame".to_owned()),
    ));
    loop {
        let cycle_started = std::time::Instant::now();
        if stop_flag.load(Ordering::Acquire) {
            break;
        }

        let Some(capture) = capture_worker.recv_latest(Duration::from_millis(100))? else {
            continue;
        };
        let capture_finished = std::time::Instant::now();
        let (fw, fh) = match &capture {
            crate::io::webcam::WebcamCapture::Mjpeg { width, height, .. } => (*width, *height),
            crate::io::webcam::WebcamCapture::Rgb(frame) => (frame.width, frame.height),
        };
        let slot = frames.acquire(fw, fh)?;
        let active = 3 * fw as usize * fh as usize;
        match capture {
            crate::io::webcam::WebcamCapture::Mjpeg {
                buffer,
                width,
                height,
            } => {
                let data = buffer.buffer();
                let decode_result = nvjpeg
                    .as_mut()
                    .map(|decoder| decoder.decode_into(data, &mut slot.u8_in, &stream));
                match decode_result {
                    Some(Ok((decoded_width, decoded_height))) => anyhow::ensure!(
                        (decoded_width, decoded_height) == (width, height),
                        "nvJPEG geometry {decoded_width}x{decoded_height} differs from camera {width}x{height}"
                    ),
                    Some(Err(error)) => {
                        nvjpeg = None;
                        let _ = tx.send(WorkerMsg::Status(format!(
                            "nvJPEG disabled; using CPU JPEG fallback: {error}"
                        )));
                        let rgb = image::load_from_memory(data)
                            .map_err(|error| anyhow::anyhow!("MJPEG frame decode failed: {error}"))?
                            .into_rgb8();
                        anyhow::ensure!(
                            rgb.dimensions() == (width, height),
                            "CPU JPEG geometry differs from camera frame"
                        );
                        gpu.upload_into_u8(rgb.as_raw(), &mut slot.u8_in)?;
                    }
                    None => {
                        let rgb = image::load_from_memory(data)
                            .map_err(|error| anyhow::anyhow!("MJPEG frame decode failed: {error}"))?
                            .into_rgb8();
                        anyhow::ensure!(
                            rgb.dimensions() == (width, height),
                            "CPU JPEG geometry differs from camera frame"
                        );
                        gpu.upload_into_u8(rgb.as_raw(), &mut slot.u8_in)?;
                    }
                }
            }
            crate::io::webcam::WebcamCapture::Rgb(frame) => {
                gpu.upload_into_u8(&frame.data, &mut slot.u8_in)?;
            }
        }
        let input_finished = std::time::Instant::now();

        let was_warmed = warmed;
        if !warmed {
            let _ = tx.send(WorkerMsg::Status(format!(
                "Warming up {}x",
                params.dim as u32
            )));
            // Three real-shape passes populate cuDNN/ORT algorithm caches and
            // touch every persistent ring/workspace allocation before live work.
            for _ in 0..3 {
                gpu.hwc_u8_to_chw_f32(&slot.u8_in, &mut slot.chw, fh, fw)?;
                let _ = engine.process_chw(&mut slot.chw, fh, fw)?;
            }
            gpu.sync()?;
            warmed = true;
            let _ = tx.send(WorkerMsg::Status("Running".into()));
            // Restore the first live frame after destructive warm-up passes.
            gpu.hwc_u8_to_chw_f32(&slot.u8_in, &mut slot.chw, fh, fw)?;
            last_t = std::time::Instant::now();
            frame_n = 0;
        } else {
            gpu.hwc_u8_to_chw_f32(&slot.u8_in, &mut slot.chw, fh, fw)?;
        }

        let result = engine.process_chw(&mut slot.chw, fh, fw)?;
        let process_finished = std::time::Instant::now();
        let _ = tx.try_send(WorkerMsg::FaceCount(result.faces_detected));

        #[cfg(target_os = "linux")]
        let gpu_preview_active = gpu_preview.is_some();
        #[cfg(not(target_os = "linux"))]
        let gpu_preview_active = false;

        let needs_host_output = vcam.is_some() || save_to_file || !gpu_preview_active;
        if needs_host_output {
            gpu.chw_f32_to_hwc_u8(&slot.chw, &mut slot.u8_out, fh, fw)?;
            let out_view = slot.u8_out.slice(..active);
            let host_out = slot.host_out.as_mut_slice()?;
            gpu.stream.memcpy_dtoh(&out_view, &mut host_out[..active])?;
        }
        #[cfg(target_os = "linux")]
        if let Some(preview) = &gpu_preview {
            // The bridge sync also completes an optional V4L2/file readback,
            // so the hot path pays for one stream boundary, not two.
            if !preview.publish(&gpu, &slot.chw)? && needs_host_output {
                gpu.sync()?;
            }
        } else if needs_host_output {
            gpu.sync()?;
        }
        #[cfg(not(target_os = "linux"))]
        if needs_host_output {
            gpu.sync()?;
        }
        let out = if needs_host_output {
            let host_out = slot.host_out.as_mut_slice()?;
            host_out[..active].to_vec()
        } else {
            Vec::new()
        };
        if let Some(v) = &mut vcam {
            send_virtual_camera_frame(v, &gpu, &slot.chw, &out, fw, fh)?;
        }
        if save_to_file && let OutputDest::File(p) = &output_dest {
            let name = if p.as_os_str().is_empty() {
                format!("frame_{frame_n:04}.png")
            } else {
                format!(
                    "{}_{}.png",
                    p.file_stem().unwrap_or_default().to_string_lossy(),
                    frame_n
                )
            };
            let _ = image::save_buffer(&name, &out, fw, fh, image::ColorType::Rgb8);
        }
        if !gpu_preview_active {
            let _ = tx.try_send(WorkerMsg::Frame(out, fw, fh));
        }
        let output_finished = std::time::Instant::now();

        if live_profiling && was_warmed {
            for (total, elapsed) in live_profile_ms.iter_mut().zip([
                capture_finished.duration_since(cycle_started),
                input_finished.duration_since(capture_finished),
                process_finished.duration_since(input_finished),
                output_finished.duration_since(process_finished),
                output_finished.duration_since(cycle_started),
            ]) {
                *total += elapsed.as_secs_f64() * 1_000.0;
            }
            live_profile_samples += 1;
            if live_profile_samples == 30 {
                let average = live_profile_ms.map(|total| total / f64::from(live_profile_samples));
                tracing::info!(
                    "[live-profile] total={:.2}ms | capture_wait={:.2}ms | input={:.2}ms | process={:.2}ms | output={:.2}ms",
                    average[4],
                    average[0],
                    average[1],
                    average[2],
                    average[3]
                );
                live_profile_ms = [0.0; 5];
                live_profile_samples = 0;
            }
        }

        frame_n += 1;
        let now = std::time::Instant::now();
        let dt = now.duration_since(last_t);
        if dt >= Duration::from_millis(500) {
            let fps = frame_n as f32 / dt.as_secs_f32();
            let _ = tx.try_send(WorkerMsg::Fps(fps));
            frame_n = 0;
            last_t = now;
        }
        ctx.request_repaint();
    }
    Ok(())
}

impl eframe::App for App {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Dark theme
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = BG;
        visuals.window_fill = CARD;
        visuals.faint_bg_color = CARD;
        visuals.extreme_bg_color = Color32::from_rgb(0x1a, 0x1a, 0x1a);
        ctx.set_visuals(visuals);
        // Dark theme — egui 0.35 has set_visuals but no ctx.set_style();
        // spacing tweaks go through ui.ctx() in ui() instead.
        if self.running {
            ctx.request_repaint_after(Duration::from_millis(16));
        }
        // Drain worker channel every frame.
        self.poll_webcam_probe();
        self.poll_worker(ctx);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.render_ui(ui);
        self.output_zoom(ui.ctx());
        self.show_realtime_preview(ui.ctx());

        #[cfg(any())]
        {
            ui.spacing_mut().item_spacing = egui::vec2(8.0, 10.0);

            // ── Header ──
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("noperson")
                        .color(TEXT)
                        .strong()
                        .size(20.0),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new("CUDA · ort").color(TEXT_DIM));
                });
            });

            // ── Status bar ──
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("{:.1} fps", self.fps)).color(ACCENT));
                ui.separator();
                ui.label(egui::RichText::new(format!("{} faces", self.face_count)).color(TEXT_DIM));
                ui.separator();
                ui.label(egui::RichText::new(&self.status).color(TEXT_DIM));
            });

            ui.add_space(4.0);

            // ── Source / Target row ──
            ui.horizontal(|ui| {
                // Source column
                ui.vertical(|ui| {
                    ui.label(egui::RichText::new("Source face").color(TITLE_BLUE));
                    ui.add_space(2.0);
                    // Drop zone
                    let drop = egui::Frame::NONE
                        .fill(Color32::from_rgb(0x2a, 0x2a, 0x2a))
                        .stroke(egui::Stroke::new(2.0, Color32::from_rgb(0x44, 0x44, 0x44)))
                        .corner_radius(8.0)
                        .inner_margin(egui::Margin::same(12));
                    drop.show(ui, |ui| {
                        ui.set_min_size(egui::vec2(100.0, 80.0));
                        match &self.input_source {
                            InputSource::None => ui.label(
                                egui::RichText::new("Drop or pick\nsource")
                                    .color(TEXT_DIM)
                                    .size(11.0),
                            ),
                            InputSource::Photo(p) => ui.label(
                                egui::RichText::new(format!(
                                    "📷\n{}",
                                    p.file_name().unwrap_or_default().to_string_lossy()
                                ))
                                .color(TEXT)
                                .size(11.0),
                            ),
                            InputSource::Video(p) => ui.label(
                                egui::RichText::new(format!(
                                    "🎬\n{}",
                                    p.file_name().unwrap_or_default().to_string_lossy()
                                ))
                                .color(TEXT)
                                .size(11.0),
                            ),
                            InputSource::Webcam(i) => ui.label(
                                egui::RichText::new(format!("🎥\nWebcam {i}"))
                                    .color(TEXT)
                                    .size(11.0),
                            ),
                        };
                    });
                    ui.add_space(2.0);
                    ui.horizontal(|ui| {
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new("📷 Photo").color(Color32::WHITE),
                                )
                                .fill(ACCENT),
                            )
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Image", &["jpg", "png", "bmp", "webp"])
                                .pick_file()
                            {
                                self.input_source = InputSource::Photo(path);
                            }
                        }
                        if ui
                            .add(
                                egui::Button::new(egui::RichText::new("🎬 Video").color(TEXT))
                                    .fill(Color32::from_rgb(0x3a, 0x3a, 0x3a)),
                            )
                            .clicked()
                        {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Video", &["mp4", "avi", "mkv", "mov", "webm"])
                                .pick_file()
                            {
                                self.input_source = InputSource::Video(path);
                            }
                        }
                    });
                    ui.add_space(2.0);
                    if ui
                        .add(
                            egui::Button::new(egui::RichText::new("🎥 Webcam").color(TEXT))
                                .fill(Color32::from_rgb(0x3a, 0x3a, 0x3a)),
                        )
                        .clicked()
                    {
                        self.input_source = InputSource::Webcam(0);
                    }
                });

                // Swap button
                ui.vertical(|ui| {
                    ui.add_space(40.0);
                    if ui
                        .add(
                            egui::Button::new(egui::RichText::new("⇄").color(TEXT))
                                .fill(Color32::from_rgb(0x3a, 0x3a, 0x3a))
                                .min_size(egui::vec2(36.0, 36.0)),
                        )
                        .clicked()
                    {
                        self.swap_paths();
                    }
                });

                // Target column
                ui.vertical(|ui| {
                    ui.label(egui::RichText::new("Target face").color(TITLE_BLUE));
                    ui.add_space(2.0);
                    let drop = egui::Frame::NONE
                        .fill(Color32::from_rgb(0x2a, 0x2a, 0x2a))
                        .stroke(egui::Stroke::new(2.0, Color32::from_rgb(0x44, 0x44, 0x44)))
                        .corner_radius(8.0)
                        .inner_margin(egui::Margin::same(12));
                    drop.show(ui, |ui| {
                        ui.set_min_size(egui::vec2(100.0, 80.0));
                        if let Some(texture) = &self.target_face_texture {
                            ui.add(
                                egui::Image::new(egui::load::SizedTexture::from_handle(texture))
                                    .max_size(egui::Vec2::splat(76.0)),
                            );
                        } else if let Some(preview) = &self.target_face_preview {
                            let texture = ui.ctx().load_texture(
                                "target_face",
                                preview.clone(),
                                egui::TextureOptions::LINEAR,
                            );
                            self.target_face_texture = Some(texture.clone());
                            ui.add(
                                egui::Image::new(egui::load::SizedTexture::from_handle(&texture))
                                    .max_size(egui::Vec2::splat(76.0)),
                            );
                        } else {
                            ui.label(
                                egui::RichText::new("Drop or pick\nface")
                                    .color(TEXT_DIM)
                                    .size(11.0),
                            );
                        }
                    });
                    ui.add_space(2.0);
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new("Select face").color(Color32::WHITE),
                            )
                            .fill(ACCENT),
                        )
                        .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Image", &["jpg", "png", "bmp", "webp"])
                            .pick_file()
                        {
                            self.target_face_path = Some(path.clone());
                            match load_image(&path) {
                                Ok((data, w, h)) => {
                                    self.target_face_preview = Some(egui::ColorImage::from_rgb(
                                        [w as usize, h as usize],
                                        &data,
                                    ));
                                    self.target_face_texture = None; // force reload
                                }
                                Err(e) => self.status = format!("Error: {e}"),
                            }
                        }
                    }
                });
            });

            // ── Output preview ──
            if self.output_preview.is_some()
                || self.output_texture.is_some()
                || self.output_gpu_texture.is_some()
            {
                card(ui, "Preview", |ui| {
                    if let Some(texture) = self.output_gpu_texture {
                        ui.add(egui::Image::new(texture).max_size(egui::vec2(420.0, 236.0)));
                    } else if let Some(texture) = &self.output_texture {
                        ui.add(
                            egui::Image::new(egui::load::SizedTexture::from_handle(texture))
                                .max_size(egui::vec2(420.0, 236.0)),
                        );
                    } else if let Some(preview) = &self.output_preview {
                        let texture = ui.ctx().load_texture(
                            "output_preview",
                            preview.clone(),
                            egui::TextureOptions::LINEAR,
                        );
                        self.output_texture = Some(texture.clone());
                        ui.add(
                            egui::Image::new(egui::load::SizedTexture::from_handle(&texture))
                                .max_size(egui::vec2(420.0, 236.0)),
                        );
                    }
                });
                ui.add_space(4.0);
            }

            ui.add_space(4.0);

            // ── Options card ──
            card(ui, "Options", |ui| {
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        toggle(ui, "Enabled", &mut self.params.enabled);
                        toggle(ui, "Color correction", &mut self.params.color_correction);
                        toggle(
                            ui,
                            "Histogram matching",
                            &mut self.params.histogram_matching,
                        );
                        toggle(ui, "Occluder", &mut self.params.occluder_enabled);
                    });
                    ui.vertical(|ui| {
                        toggle(ui, "XSeg mask", &mut self.params.xseg_enabled);
                        toggle(ui, "Face parser", &mut self.params.faceparser_enabled);
                        toggle(ui, "Restore mouth", &mut self.params.restore_mouth);
                        toggle(ui, "Restore eyes", &mut self.params.restore_eyes);
                    });
                });
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Swap resolution:").color(TEXT_DIM));
                    ui.radio_value(&mut self.params.dim, SwapDim::Dim1, "128");
                    ui.radio_value(&mut self.params.dim, SwapDim::Dim2, "256");
                    ui.radio_value(&mut self.params.dim, SwapDim::Dim3, "384");
                    ui.radio_value(&mut self.params.dim, SwapDim::Dim4, "512");
                });
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Restorer:").color(TEXT_DIM));
                    toggle(ui, "enabled", &mut self.params.restorer_enabled);
                });
                if self.params.restorer_enabled {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Hot path:").color(TEXT_DIM));
                        ui.radio_value(
                            &mut self.params.restorer_mode,
                            RestorerMode::Realtime,
                            "Realtime",
                        );
                        ui.radio_value(
                            &mut self.params.restorer_mode,
                            RestorerMode::Quality,
                            "Quality",
                        );
                    });
                }
            });

            ui.add_space(4.0);

            // ── Refinement card ──
            card(ui, "Refinement", |ui| {
                labeled_slider(ui, "Strength", &mut self.params.strength, 0.0, 1.0);
                labeled_slider(
                    ui,
                    "Similarity threshold",
                    &mut self.params.similarity_threshold,
                    0.0,
                    1.0,
                );
                labeled_slider(
                    ui,
                    "Restorer alpha",
                    &mut self.params.restorer_alpha,
                    0.0,
                    1.0,
                );
                ui.separator();
                labeled_u32_slider(ui, "Border top", &mut self.params.border_top, 0, 100);
                labeled_u32_slider(ui, "Border bottom", &mut self.params.border_bottom, 0, 100);
                labeled_u32_slider(ui, "Border left", &mut self.params.border_left, 0, 100);
                labeled_u32_slider(ui, "Border right", &mut self.params.border_right, 0, 100);
                labeled_u32_slider(ui, "Border blur", &mut self.params.border_blur, 0, 100);
            });

            ui.add_space(4.0);

            // ── Output card ──
            card(ui, "Output", |ui| {
                ui.radio_value(
                    &mut self.output_dest,
                    OutputDest::VirtualCamera(10),
                    "Virtual Camera (/dev/video10)",
                );
                ui.radio_value(
                    &mut self.output_dest,
                    OutputDest::File(PathBuf::new()),
                    "Save to file",
                );
            });

            // ── Action row ──
            ui.horizontal(|ui| {
                let (btn_text, btn_color) = if self.running {
                    ("Stop", DANGER)
                } else {
                    ("Start", ACCENT)
                };
                let btn = egui::Button::new(egui::RichText::new(btn_text).color(Color32::WHITE))
                    .fill(btn_color)
                    .min_size(egui::vec2(120.0, 32.0));
                if ui.add(btn).clicked() {
                    if self.running {
                        self.stop(ui.ctx());
                    } else {
                        self.start(ui.ctx());
                    }
                }
                if self.running {
                    ui.spinner();
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new("noperson")
                            .color(Color32::from_rgb(0x6e, 0xa8, 0xff))
                            .small(),
                    );
                });
            });
        }
    }
}

fn send_worker_done(
    tx: &mpsc::SyncSender<WorkerMsg>,
    ctx: &egui::Context,
    result: anyhow::Result<()>,
) {
    let _ = tx.send(WorkerMsg::Done(result));
    ctx.request_repaint();
}

fn load_image(path: &std::path::Path) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    let img = image::open(path).map_err(|e| anyhow::anyhow!("Failed to open image: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    Ok((rgb.into_raw(), w, h))
}

fn load_preview(path: &std::path::Path) -> anyhow::Result<egui::ColorImage> {
    let (data, width, height) = load_image(path)?;
    color_preview_from_rgb(&data, width, height, 1024)
}

fn color_preview_from_rgb(
    data: &[u8],
    width: u32,
    height: u32,
    max_side: u32,
) -> anyhow::Result<egui::ColorImage> {
    anyhow::ensure!(
        data.len() == width as usize * height as usize * 3,
        "invalid RGB preview buffer"
    );
    let largest = width.max(height);
    if largest <= max_side {
        return Ok(egui::ColorImage::from_rgb(
            [width as usize, height as usize],
            data,
        ));
    }
    let scale = max_side as f32 / largest as f32;
    let preview_width = ((width as f32 * scale).round() as u32).max(1);
    let preview_height = ((height as f32 * scale).round() as u32).max(1);
    let source = image::RgbImage::from_raw(width, height, data.to_vec())
        .ok_or_else(|| anyhow::anyhow!("invalid RGB preview buffer"))?;
    let preview = image::imageops::resize(
        &source,
        preview_width,
        preview_height,
        image::imageops::FilterType::Triangle,
    );
    Ok(egui::ColorImage::from_rgb(
        [preview.width() as usize, preview.height() as usize],
        preview.as_raw(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::{Harness, kittest::Queryable as _};

    #[test]
    fn image_picker_history_reopens_the_latest_selected_directory() {
        let mut history = ImagePickerHistory::default();
        assert_eq!(history.directory(), None);

        history.remember_selection(std::path::Path::new("/photos/first/face.jpg"));
        assert_eq!(
            history.directory(),
            Some(std::path::Path::new("/photos/first"))
        );

        history.remember_selection(std::path::Path::new("/photos/second/target.png"));
        assert_eq!(
            history.directory(),
            Some(std::path::Path::new("/photos/second"))
        );
    }

    #[test]
    fn webcam_without_a_face_starts_camera_preview_instead_of_failing() {
        assert_eq!(
            start_intent(&InputSource::Webcam(0), None),
            Some(StartIntent::CameraPreview)
        );
        assert_eq!(
            start_intent(&InputSource::Webcam(0), Some(Path::new("face.jpg"))),
            Some(StartIntent::FaceSwap)
        );
        assert_eq!(
            start_intent(&InputSource::Photo("photo.jpg".into()), None),
            None
        );
    }

    #[test]
    fn output_viewport_toggles_between_fit_and_natural_size() {
        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(500.0, 400.0));
        let image_size = egui::vec2(1000.0, 500.0);
        let mut state = OutputViewport::default();

        assert_eq!(state.scale(viewport.size(), image_size), 0.5);
        state.toggle_zoom(viewport, image_size, viewport.center());
        assert_eq!(state.scale(viewport.size(), image_size), 1.0);

        state.toggle_zoom(viewport, image_size, viewport.center());
        assert_eq!(state.scale(viewport.size(), image_size), 0.5);
        assert_eq!(state.pan, egui::Vec2::ZERO);
    }

    #[test]
    fn output_viewport_wheel_zoom_preserves_the_pixel_under_the_pointer() {
        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(500.0, 400.0));
        let image_size = egui::vec2(1000.0, 500.0);
        let pointer = egui::pos2(400.0, 100.0);
        let mut state = OutputViewport::default();
        let before = state.image_point_at(viewport, image_size, pointer);

        state.zoom_by(viewport, image_size, pointer, 2.0);

        assert_eq!(state.image_point_at(viewport, image_size, pointer), before);
        assert_eq!(state.scale(viewport.size(), image_size), 1.0);
    }

    #[test]
    fn output_viewport_clamps_zoom_and_resets_pan() {
        let viewport = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(500.0, 400.0));
        let image_size = egui::vec2(1000.0, 500.0);
        let mut state = OutputViewport::default();

        state.zoom_by(viewport, image_size, viewport.center(), 100.0);
        assert_eq!(state.scale(viewport.size(), image_size), 8.0);
        state.pan_by(egui::vec2(20.0, -15.0));
        assert_eq!(state.pan, egui::vec2(20.0, -15.0));

        state.reset();
        assert_eq!(state.scale(viewport.size(), image_size), 0.5);
        assert_eq!(state.pan, egui::Vec2::ZERO);
    }

    #[test]
    fn live_layout_exposes_the_complete_primary_workflow() {
        let harness = Harness::builder()
            .with_size(egui::vec2(900.0, 900.0))
            .build_ui_state(
                |ui, app| app.render_ui(ui),
                App::new(PathBuf::from("models")),
            );

        for label in [
            "Source face",
            "Target",
            "Photo",
            "Camera",
            "Output preview",
            "Advanced settings",
            "Start live",
        ] {
            assert!(
                harness.query_by_label(label).is_some(),
                "missing primary live control: {label}"
            );
        }
        assert!(harness.query_by_label("Video").is_none());
        assert!(harness.query_by_label("Preview").is_none());
        assert!(
            harness
                .query_by_label(&elegance::glyphs::POWER.to_string())
                .is_none(),
            "the redundant private-use power glyph collides with Inter and renders as G"
        );
        for unsupported in [
            "Color correction",
            "Occlusion mask",
            "Restore mouth",
            "Restore eyes",
            "Histogram matching",
        ] {
            assert!(
                harness.query_by_label(unsupported).is_none(),
                "inert control must not be exposed: {unsupported}"
            );
        }
    }

    #[test]
    fn live_layout_installs_the_product_theme_before_rendering() {
        let observed = Arc::new(std::sync::Mutex::new(Color32::TRANSPARENT));
        let captured = Arc::clone(&observed);
        let _harness = Harness::builder()
            .with_size(egui::vec2(666.0, 839.0))
            .build_ui_state(
                move |ui, app| {
                    app.render_ui(ui);
                    *captured.lock().expect("theme observation lock") = ui.visuals().panel_fill;
                },
                App::new(PathBuf::from("models")),
            );

        assert_eq!(
            *observed.lock().expect("theme observation lock"),
            Color32::from_rgb(0x0d, 0x0d, 0x11)
        );
    }

    #[test]
    #[ignore = "local visual baseline is stored in ignored tests/snapshots"]
    fn live_layout_visual_snapshot() {
        let mut harness = Harness::builder()
            .with_size(egui::vec2(666.0, 839.0))
            .wgpu()
            .build_ui_state(
                |ui, app| app.render_ui(ui),
                App::new(PathBuf::from("models")),
            );
        harness.snapshot("live_layout_default");
    }

    #[test]
    #[ignore = "local visual baseline is stored in ignored tests/snapshots"]
    fn live_layout_with_output_visual_snapshot() -> anyhow::Result<()> {
        let (data, width, height) = load_image(std::path::Path::new("face.jpg"))?;
        let preview = color_preview_from_rgb(&data, width, height, 1024)?;
        let mut app = App::new(PathBuf::from("models"));
        app.target_face_path = Some(PathBuf::from("face.jpg"));
        app.target_face_preview = Some(preview.clone());
        app.input_source = InputSource::Photo(PathBuf::from("face.jpg"));
        app.input_preview = Some(preview.clone());
        app.output_preview = Some(preview);
        app.output_frame = Some((data, width, height));

        let mut harness = Harness::builder()
            .with_size(egui::vec2(666.0, 839.0))
            .wgpu()
            .build_ui_state(|ui, app| app.render_ui(ui), app);
        harness.snapshot("live_layout_with_output");
        Ok(())
    }

    #[test]
    fn advanced_controls_are_hidden_until_requested() {
        let mut harness = Harness::builder()
            .with_size(egui::vec2(900.0, 900.0))
            .build_ui_state(
                |ui, app| app.render_ui(ui),
                App::new(PathBuf::from("models")),
            );

        assert!(harness.query_by_label("Border top").is_none());
        harness.get_by_label("Advanced settings").click();
        harness.run();
        assert!(harness.query_by_label("Border top").is_some());
        assert!(harness.query_by_label("Similarity threshold").is_some());
    }

    #[test]
    fn live_layout_exposes_provider_and_output_actions() {
        let harness = Harness::builder()
            .with_size(egui::vec2(666.0, 839.0))
            .build_ui_state(
                |ui, app| app.render_ui(ui),
                App::new(PathBuf::from("models")),
            );

        for label in ["Native CUDA", "TensorRT", "Save output"] {
            assert!(
                harness.query_by_label(label).is_some(),
                "missing live control: {label}"
            );
        }
    }

    #[test]
    fn clicking_output_image_opens_enlarged_preview() {
        let mut app = App::new(PathBuf::from("models"));
        app.output_preview = Some(egui::ColorImage::from_rgb(
            [2, 2],
            &[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255],
        ));
        app.output_frame = Some((vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255], 2, 2));
        let mut harness = Harness::builder()
            .with_size(egui::vec2(666.0, 839.0))
            .build_ui_state(
                |ui, app| {
                    app.render_ui(ui);
                    app.output_zoom(ui.ctx());
                },
                app,
            );

        harness.get_by_label("Output image").click();
        harness.run();
        assert!(harness.query_by_label("Close preview").is_some());
    }

    #[test]
    fn provider_selection_updates_the_live_backend() {
        let mut harness = Harness::builder()
            .with_size(egui::vec2(666.0, 839.0))
            .build_ui_state(
                |ui, app| app.render_ui(ui),
                App::new(PathBuf::from("models")),
            );

        harness.get_by_label("TensorRT").click();
        harness.run();
        assert!(harness.query_by_label("TENSORRT · LOCAL").is_some());
    }

    #[test]
    fn webcam_start_state_opens_realtime_preview_and_clears_metrics() {
        let mut app = App::new(PathBuf::from("models"));
        app.fps = 42.0;
        app.face_count = 3;

        app.begin_realtime_preview(&InputSource::Webcam(0));

        assert!(app.realtime_preview.is_open());
        assert_eq!(app.fps, 0.0);
        assert_eq!(app.face_count, 0);
    }

    #[test]
    fn photo_start_state_does_not_open_realtime_preview() {
        let mut app = App::new(PathBuf::from("models"));

        app.begin_realtime_preview(&InputSource::Photo(PathBuf::from("face.jpg")));

        assert!(!app.realtime_preview.is_open());
    }

    #[test]
    fn manual_preview_close_does_not_stop_live_worker() {
        let mut app = App::new(PathBuf::from("models"));
        app.running = true;
        app.realtime_preview.open();
        app.realtime_preview.mark_close_requested();

        app.apply_realtime_preview_close_request();

        assert!(!app.realtime_preview.is_open());
        assert!(app.running);
    }

    #[test]
    fn worker_done_closes_realtime_preview() {
        let mut app = App::new(PathBuf::from("models"));
        let (tx, rx) = std::sync::mpsc::channel();
        app.msg_rx = Some(rx);
        app.running = true;
        app.realtime_preview.open();
        tx.send(WorkerMsg::Done(Ok(()))).unwrap();

        app.poll_worker(&egui::Context::default());

        assert!(!app.realtime_preview.is_open());
        assert!(!app.running);
    }

    #[test]
    fn preview_mailbox_renders_only_the_latest_queued_frame() {
        let mut app = App::new(PathBuf::from("models"));
        let (tx, rx) = std::sync::mpsc::sync_channel(3);
        app.msg_rx = Some(rx);
        tx.send(WorkerMsg::Frame(vec![1, 1, 1], 1, 1)).unwrap();
        tx.send(WorkerMsg::Frame(vec![2, 2, 2], 1, 1)).unwrap();

        app.poll_worker(&egui::Context::default());

        assert_eq!(app.output_frame, Some((vec![2, 2, 2], 1, 1)));
    }

    #[test]
    fn terminal_worker_message_wakes_the_root_viewport() {
        use std::sync::atomic::AtomicUsize;

        let ctx = egui::Context::default();
        let repaint_count = Arc::new(AtomicUsize::new(0));
        let observed_repaints = repaint_count.clone();
        ctx.set_request_repaint_callback(move |_| {
            observed_repaints.fetch_add(1, Ordering::Relaxed);
        });
        let (tx, rx) = std::sync::mpsc::sync_channel(1);

        send_worker_done(&tx, &ctx, Ok(()));

        assert!(matches!(rx.recv().unwrap(), WorkerMsg::Done(Ok(()))));
        assert!(
            repaint_count.load(Ordering::Relaxed) > 0,
            "terminal worker message must wake the root viewport"
        );
    }
}
