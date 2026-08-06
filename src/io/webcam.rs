//! Webcam capture via nokhwa (v4l2 on Linux).
//!
//! Produces RGB frames at the requested resolution and FPS.
//!
//! Note: `nokhwa::Camera` is NOT `Send` on Linux (v4l2 backend holds a
//! raw fd in a `RefCell`). The camera must live on the thread that created
//! it. The pipeline thread owns the camera and sends frames back via channel.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use nokhwa::Camera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
    Resolution,
};

use super::video::{Frame, FrameSource};

/// Webcam frame source — NOT Send. Must be driven from a single thread.
pub struct WebcamSource {
    camera: Camera,
    width: u32,
    height: u32,
    fps: f32,
}

pub enum WebcamCapture {
    Mjpeg {
        buffer: nokhwa::Buffer,
        width: u32,
        height: u32,
    },
    Rgb(Frame),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WebcamCaptureInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f32,
}

struct LatestCaptureState<T> {
    latest: Option<T>,
    closed: bool,
    error: Option<String>,
}

struct LatestCaptureMailbox<T> {
    state: Mutex<LatestCaptureState<T>>,
    ready: Condvar,
}

impl<T> Default for LatestCaptureMailbox<T> {
    fn default() -> Self {
        Self {
            state: Mutex::new(LatestCaptureState {
                latest: None,
                closed: false,
                error: None,
            }),
            ready: Condvar::new(),
        }
    }
}

impl<T> LatestCaptureMailbox<T> {
    fn publish(&self, capture: T) {
        let mut state = self.state.lock().expect("webcam mailbox poisoned");
        state.latest = Some(capture);
        self.ready.notify_one();
    }

    fn close(&self, error: Option<String>) {
        let mut state = self.state.lock().expect("webcam mailbox poisoned");
        state.closed = true;
        state.error = error;
        self.ready.notify_all();
    }

    fn recv_latest(&self, timeout: Duration) -> anyhow::Result<Option<T>> {
        let state = self.state.lock().expect("webcam mailbox poisoned");
        let (mut state, _) = self
            .ready
            .wait_timeout_while(state, timeout, |state| {
                state.latest.is_none() && !state.closed
            })
            .expect("webcam mailbox poisoned while waiting");
        if let Some(error) = &state.error {
            anyhow::bail!(error.clone());
        }
        Ok(state.latest.take())
    }
}

/// Dedicated camera owner. Capture never waits for inference: a newly read
/// frame atomically replaces the stale mailbox entry, keeping latency bounded
/// to one camera frame without moving nokhwa's non-Send camera object.
pub struct WebcamCaptureWorker {
    mailbox: Arc<LatestCaptureMailbox<WebcamCapture>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl WebcamCaptureWorker {
    pub fn spawn(
        index: usize,
        width: u32,
        height: u32,
        fps: f32,
        stop: Arc<AtomicBool>,
    ) -> anyhow::Result<(Self, WebcamCaptureInfo)> {
        let mailbox = Arc::new(LatestCaptureMailbox::default());
        let thread_mailbox = Arc::clone(&mailbox);
        let thread_stop = Arc::clone(&stop);
        let (init_tx, init_rx) = std::sync::mpsc::sync_channel(1);
        let thread = std::thread::Builder::new()
            .name("webcam-capture".to_owned())
            .spawn(move || {
                let mut camera = match WebcamSource::new(index, width, height, fps) {
                    Ok(camera) => camera,
                    Err(error) => {
                        let _ = init_tx.send(Err(error.to_string()));
                        return;
                    }
                };
                let info = WebcamCaptureInfo {
                    width: camera.width,
                    height: camera.height,
                    fps: camera.fps,
                };
                if init_tx.send(Ok(info)).is_err() {
                    return;
                }

                let mut last_frame_at = Instant::now();
                while !thread_stop.load(Ordering::Acquire) {
                    match camera.next_capture_result() {
                        Ok(capture) => {
                            last_frame_at = Instant::now();
                            thread_mailbox.publish(capture);
                        }
                        Err(error) if last_frame_at.elapsed() >= Duration::from_secs(5) => {
                            thread_mailbox.close(Some(format!(
                                "Webcam stopped delivering frames for 5 seconds: {error}"
                            )));
                            return;
                        }
                        Err(_) => std::thread::sleep(Duration::from_millis(1)),
                    }
                }
                thread_mailbox.close(None);
            })?;

        let info = match init_rx.recv() {
            Ok(Ok(info)) => info,
            Ok(Err(error)) => {
                let _ = thread.join();
                anyhow::bail!(error);
            }
            Err(error) => {
                let _ = thread.join();
                anyhow::bail!("webcam capture thread exited during startup: {error}");
            }
        };
        Ok((
            Self {
                mailbox,
                stop,
                thread: Some(thread),
            },
            info,
        ))
    }

    pub fn recv_latest(&self, timeout: Duration) -> anyhow::Result<Option<WebcamCapture>> {
        self.mailbox.recv_latest(timeout)
    }
}

impl Drop for WebcamCaptureWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

const PRESET_RESOLUTIONS: &[(u32, u32)] = &[
    (640, 480),
    (960, 540),
    (1280, 720),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
];
static PROBED_PRESETS: OnceLock<Vec<WebcamPreset>> = OnceLock::new();

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebcamPreset {
    pub camera_index: usize,
    pub camera_name: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

impl WebcamPreset {
    #[must_use]
    pub fn label(&self) -> String {
        format!(
            "{} ({}): {}p{}",
            self.camera_name, self.camera_index, self.height, self.fps
        )
    }
}

fn open_camera_with_fallback(
    camera_index: CameraIndex,
    width: u32,
    height: u32,
    fps: u32,
) -> anyhow::Result<Camera> {
    let resolution = Resolution::new(width, height);
    let attempts = [FrameFormat::MJPEG, FrameFormat::NV12];
    for format in attempts {
        let request = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(
            CameraFormat::new(resolution, format, fps),
        ));
        if let Ok(camera) = Camera::new(camera_index.clone(), request) {
            return Ok(camera);
        }
    }

    Camera::new(
        camera_index,
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
    )
    .map_err(|error| anyhow::anyhow!("failed to open camera with any color format: {error}"))
}

fn presets_from_formats(
    camera_index: usize,
    camera_name: &str,
    formats: &[CameraFormat],
) -> Vec<WebcamPreset> {
    let mut best_per_resolution = HashMap::<(u32, u32), u32>::new();
    for minimum_fps in [25, 15] {
        for format in formats {
            let resolution = format.resolution();
            let geometry = (resolution.width(), resolution.height());
            if PRESET_RESOLUTIONS.contains(&geometry) && format.frame_rate() >= minimum_fps {
                best_per_resolution
                    .entry(geometry)
                    .and_modify(|fps| *fps = (*fps).max(format.frame_rate()))
                    .or_insert(format.frame_rate());
            }
        }
        if !best_per_resolution.is_empty() {
            break;
        }
    }
    if best_per_resolution.is_empty()
        && let Some(format) = formats.first()
    {
        best_per_resolution.insert(
            (format.resolution().width(), format.resolution().height()),
            format.frame_rate(),
        );
    }

    let mut presets = best_per_resolution
        .into_iter()
        .map(|((width, height), fps)| WebcamPreset {
            camera_index,
            camera_name: camera_name.to_owned(),
            width,
            height,
            fps,
        })
        .collect::<Vec<_>>();
    presets.sort_by(|left, right| right.fps.cmp(&left.fps).then(right.width.cmp(&left.width)));
    presets
}

impl WebcamSource {
    /// Open a webcam by index (0 = default camera).
    pub fn new(index: usize, width: u32, height: u32, fps: f32) -> anyhow::Result<Self> {
        let mut camera = open_camera_with_fallback(
            CameraIndex::Index(index as u32),
            width,
            height,
            fps.round().max(1.0) as u32,
        )?;

        camera
            .open_stream()
            .map_err(|e| anyhow::anyhow!("Failed to open camera stream: {e}"))?;

        let actual_format = camera.camera_format();
        let actual_w = actual_format.width();
        let actual_h = actual_format.height();
        let actual_fps = actual_format.frame_rate() as f32;

        Ok(Self {
            camera,
            width: actual_w,
            height: actual_h,
            fps: actual_fps,
        })
    }

    /// List available cameras.
    pub fn list_cameras() -> Vec<String> {
        match nokhwa::query(ApiBackend::Auto) {
            Ok(cameras) => cameras
                .iter()
                .enumerate()
                .map(|(i, c)| format!("{}: {}", i, c.human_name()))
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Probe every camera and expose the same unified presets as CrossSwap.
    pub fn probe_all() -> Vec<WebcamPreset> {
        PROBED_PRESETS.get_or_init(probe_all_cameras).clone()
    }

    /// Read and decode one frame while preserving the backend error.
    pub fn next_frame_result(&mut self) -> anyhow::Result<Frame> {
        let buffer = self
            .camera
            .frame()
            .map_err(|error| anyhow::anyhow!("camera capture failed: {error}"))?;
        let frame = match capture_from_buffer(buffer)? {
            WebcamCapture::Mjpeg {
                buffer,
                width: _,
                height: _,
            } => {
                let rgb = image::load_from_memory(buffer.buffer())
                    .map_err(|error| anyhow::anyhow!("MJPEG frame decode failed: {error}"))?
                    .into_rgb8();
                let (width, height) = rgb.dimensions();
                Frame::from_data(rgb.into_raw(), width, height)
            }
            WebcamCapture::Rgb(frame) => frame,
        };
        self.width = frame.width;
        self.height = frame.height;
        Ok(frame)
    }

    /// Capture one frame without expanding MJPEG on the CPU.
    pub fn next_capture_result(&mut self) -> anyhow::Result<WebcamCapture> {
        let buffer = self
            .camera
            .frame()
            .map_err(|error| anyhow::anyhow!("camera capture failed: {error}"))?;
        capture_from_buffer(buffer)
    }
}

fn capture_from_buffer(buffer: nokhwa::Buffer) -> anyhow::Result<WebcamCapture> {
    let resolution = buffer.resolution();
    if buffer.source_frame_format() == FrameFormat::MJPEG {
        return Ok(WebcamCapture::Mjpeg {
            width: resolution.width(),
            height: resolution.height(),
            buffer,
        });
    }
    let rgb = buffer
        .decode_image::<RgbFormat>()
        .map_err(|error| anyhow::anyhow!("camera frame decode failed: {error}"))?;
    Ok(WebcamCapture::Rgb(Frame::from_data(
        rgb.into_raw(),
        resolution.width(),
        resolution.height(),
    )))
}

fn probe_all_cameras() -> Vec<WebcamPreset> {
    let mut presets = Vec::new();
    for info in nokhwa::query(ApiBackend::Auto).unwrap_or_default() {
        let index = match info.index() {
            CameraIndex::Index(index) => *index as usize,
            CameraIndex::String(index) => index.parse().unwrap_or(0),
        };
        let mut camera = match Camera::new(
            info.index().clone(),
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
        ) {
            Ok(camera) => camera,
            Err(_) => continue,
        };
        let formats = camera.compatible_camera_formats().unwrap_or_default();
        presets.extend(presets_from_formats(index, &info.human_name(), &formats));
    }
    if presets.is_empty() {
        presets.push(WebcamPreset {
            camera_index: 0,
            camera_name: "Camera".to_owned(),
            width: 1280,
            height: 720,
            fps: 30,
        });
    }
    presets
}

impl FrameSource for WebcamSource {
    fn next_frame(&mut self) -> Option<Frame> {
        self.next_frame_result().ok()
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn fps(&self) -> f32 {
        self.fps
    }

    fn frame_count(&self) -> Option<u64> {
        None
    }
}

impl Drop for WebcamSource {
    fn drop(&mut self) {
        let _ = self.camera.stop_stream();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_prefer_frame_rate_then_resolution_like_crossswap() {
        let formats = [
            CameraFormat::new(Resolution::new(2560, 1440), FrameFormat::MJPEG, 30),
            CameraFormat::new(Resolution::new(1920, 1080), FrameFormat::MJPEG, 60),
            CameraFormat::new(Resolution::new(1280, 720), FrameFormat::YUYV, 10),
        ];

        let presets = presets_from_formats(0, "Oracle Pro", &formats);

        assert_eq!(presets.len(), 2);
        assert_eq!((presets[0].width, presets[0].fps), (1920, 60));
        assert_eq!((presets[1].width, presets[1].fps), (2560, 30));
    }

    #[test]
    fn capture_preserves_mjpeg_until_the_gpu_decode_stage() {
        let jpeg = [0xff, 0xd8, 1, 2, 3, 0xff, 0xd9];
        let buffer = nokhwa::Buffer::new(Resolution::new(2560, 1440), &jpeg, FrameFormat::MJPEG);

        let capture = capture_from_buffer(buffer).expect("MJPEG routing must not decode on CPU");

        match capture {
            WebcamCapture::Mjpeg {
                buffer,
                width,
                height,
            } => {
                assert_eq!(buffer.buffer(), jpeg);
                assert_eq!((width, height), (2560, 1440));
            }
            WebcamCapture::Rgb(_) => panic!("MJPEG was decoded before the GPU stage"),
        }
    }

    #[test]
    fn latest_capture_mailbox_replaces_stale_frames_without_a_queue() {
        let mailbox = LatestCaptureMailbox::default();
        mailbox.publish(1_u64);
        mailbox.publish(2_u64);

        assert_eq!(mailbox.recv_latest(Duration::ZERO).unwrap(), Some(2));
        assert_eq!(mailbox.recv_latest(Duration::ZERO).unwrap(), None);
    }

    #[test]
    #[ignore = "requires a live camera selected by NOPERSON_CAMERA_INDEX"]
    fn configured_camera_produces_an_rgb_frame() {
        let index = std::env::var("NOPERSON_CAMERA_INDEX")
            .expect("set NOPERSON_CAMERA_INDEX")
            .parse::<usize>()
            .expect("camera index must be an integer");
        let preset = WebcamSource::probe_all()
            .into_iter()
            .filter(|preset| preset.camera_index == index)
            .max_by_key(|preset| (preset.width, preset.fps))
            .expect("camera must expose at least one preset");
        eprintln!("testing {}", preset.label());
        let mut camera = WebcamSource::new(index, preset.width, preset.height, preset.fps as f32)
            .expect("open camera");

        let mut last_error = None;
        let frame = (0..30)
            .find_map(|_| match camera.next_frame_result() {
                Ok(frame) => {
                    eprintln!(
                        "captured {}x{} RGB frame ({} bytes)",
                        frame.width,
                        frame.height,
                        frame.data.len()
                    );
                    Some(frame)
                }
                Err(error) => {
                    eprintln!("capture attempt failed: {error}");
                    last_error = Some(error.to_string());
                    None
                }
            })
            .unwrap_or_else(|| {
                panic!(
                    "camera must produce a decodable frame; last error: {}",
                    last_error.as_deref().unwrap_or("unknown")
                )
            });

        assert_eq!(frame.data.len(), (frame.width * frame.height * 3) as usize);
        assert_eq!(camera.dimensions(), (frame.width, frame.height));
    }
}
