//! Video file input (decode) and output (encode).
//!
//! Uses ffmpeg-next for video decoding/encoding when available.
//! The pipeline operates on raw RGB frames via the `FrameSource` trait.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Deserialize;

pub fn ffmpeg_decode_args(path: &Path) -> Vec<String> {
    ffmpeg_decode_args_impl(path, None)
}

pub fn ffmpeg_decode_args_with_threads(path: &Path, worker_threads: usize) -> Vec<String> {
    ffmpeg_decode_args_impl(path, Some(worker_threads.clamp(1, 32)))
}

fn ffmpeg_decode_args_impl(path: &Path, worker_threads: Option<usize>) -> Vec<String> {
    let mut args = vec!["-v".to_owned(), "error".to_owned()];
    if let Some(worker_threads) = worker_threads {
        args.extend(["-threads".to_owned(), worker_threads.to_string()]);
    }
    args.extend([
        "-i".to_owned(),
        path.to_string_lossy().into_owned(),
        "-f".to_owned(),
        "rawvideo".to_owned(),
        "-pix_fmt".to_owned(),
        "rgb24".to_owned(),
        "-".to_owned(),
    ]);
    args
}

pub fn ffmpeg_encode_args(path: &Path, width: u32, height: u32, fps: f32) -> Vec<String> {
    ffmpeg_encode_args_impl(path, width, height, fps, None)
}

pub fn ffmpeg_encode_args_with_threads(
    path: &Path,
    width: u32,
    height: u32,
    fps: f32,
    worker_threads: usize,
) -> Vec<String> {
    ffmpeg_encode_args_impl(path, width, height, fps, Some(worker_threads.clamp(1, 32)))
}

fn ffmpeg_encode_args_impl(
    path: &Path,
    width: u32,
    height: u32,
    fps: f32,
    worker_threads: Option<usize>,
) -> Vec<String> {
    let mut args = vec![
        "-v".to_owned(),
        "error".to_owned(),
        "-f".to_owned(),
        "rawvideo".to_owned(),
        "-pix_fmt".to_owned(),
        "rgb24".to_owned(),
        "-s:v".to_owned(),
        format!("{width}x{height}"),
        "-r".to_owned(),
        format!("{fps}"),
        "-i".to_owned(),
        "-".to_owned(),
        "-an".to_owned(),
        "-c:v".to_owned(),
        "libx264".to_owned(),
    ];
    if let Some(worker_threads) = worker_threads {
        args.extend(["-threads".to_owned(), worker_threads.to_string()]);
    }
    args.extend([
        "-pix_fmt".to_owned(),
        "yuv420p".to_owned(),
        "-y".to_owned(),
        path.to_string_lossy().into_owned(),
    ]);
    args
}

pub fn ffmpeg_remux_args(video_only: &Path, original: &Path, output: &Path) -> Vec<String> {
    vec![
        "-v".to_owned(),
        "error".to_owned(),
        "-i".to_owned(),
        video_only.to_string_lossy().into_owned(),
        "-i".to_owned(),
        original.to_string_lossy().into_owned(),
        "-map".to_owned(),
        "0:v:0".to_owned(),
        "-map".to_owned(),
        "1:a?".to_owned(),
        "-c:v".to_owned(),
        "copy".to_owned(),
        "-c:a".to_owned(),
        "copy".to_owned(),
        "-shortest".to_owned(),
        "-y".to_owned(),
        output.to_string_lossy().into_owned(),
    ]
}

pub fn remux_original_audio(
    video_only: &Path,
    original: &Path,
    output: &Path,
) -> anyhow::Result<()> {
    let result = Command::new("ffmpeg")
        .args(ffmpeg_remux_args(video_only, original, output))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()?;
    anyhow::ensure!(
        result.status.success(),
        "ffmpeg audio remux failed: {}",
        String::from_utf8_lossy(&result.stderr).trim()
    );
    Ok(())
}

#[derive(Debug, Clone)]
pub struct Timeline<T> {
    initial: T,
    markers: BTreeMap<u64, T>,
}

impl<T> Timeline<T> {
    pub fn new(initial: T) -> Self {
        Self {
            initial,
            markers: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, frame: u64, value: T) -> Option<T> {
        self.markers.insert(frame, value)
    }

    pub fn at(&self, frame: u64) -> &T {
        self.markers
            .range(..=frame)
            .next_back()
            .map_or(&self.initial, |(_, value)| value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum VideoState {
    #[default]
    Idle,
    Running {
        processed: u64,
        total: Option<u64>,
    },
    Cancelling {
        processed: u64,
        total: Option<u64>,
    },
    Cancelled {
        processed: u64,
    },
    Completed {
        processed: u64,
    },
    Failed {
        processed: u64,
        message: String,
    },
}

#[derive(Debug, Clone, Default)]
pub struct VideoLifecycle {
    state: VideoState,
}

impl VideoLifecycle {
    pub fn state(&self) -> &VideoState {
        &self.state
    }

    pub fn start(&mut self, total: Option<u64>) -> anyhow::Result<()> {
        anyhow::ensure!(
            matches!(self.state, VideoState::Idle),
            "video job already started"
        );
        self.state = VideoState::Running {
            processed: 0,
            total,
        };
        Ok(())
    }

    pub fn frame_completed(&mut self) -> anyhow::Result<()> {
        match &mut self.state {
            VideoState::Running { processed, .. } => {
                *processed += 1;
                Ok(())
            }
            _ => anyhow::bail!("frame completed outside a running video job"),
        }
    }

    pub fn request_cancel(&mut self) -> anyhow::Result<()> {
        let state = std::mem::take(&mut self.state);
        self.state = match state {
            VideoState::Running { processed, total } => VideoState::Cancelling { processed, total },
            state => {
                self.state = state;
                anyhow::bail!("video job is not running")
            }
        };
        Ok(())
    }

    pub fn finish_cancelled(&mut self) -> anyhow::Result<()> {
        let state = std::mem::take(&mut self.state);
        self.state = match state {
            VideoState::Cancelling { processed, .. } => VideoState::Cancelled { processed },
            state => {
                self.state = state;
                anyhow::bail!("video cancellation was not requested")
            }
        };
        Ok(())
    }

    pub fn complete(&mut self) -> anyhow::Result<()> {
        let state = std::mem::take(&mut self.state);
        self.state = match state {
            VideoState::Running { processed, .. } => VideoState::Completed { processed },
            state => {
                self.state = state;
                anyhow::bail!("video job is not running")
            }
        };
        Ok(())
    }

    pub fn fail(&mut self, message: impl Into<String>) -> anyhow::Result<()> {
        let state = std::mem::take(&mut self.state);
        self.state = match state {
            VideoState::Running { processed, .. } | VideoState::Cancelling { processed, .. } => {
                VideoState::Failed {
                    processed,
                    message: message.into(),
                }
            }
            state => {
                self.state = state;
                anyhow::bail!("video job is not active")
            }
        };
        Ok(())
    }
}

/// Drive decode → transform → encode while exposing deterministic frame
/// markers and cancellation only at complete frame boundaries.
pub fn process_video_frames<S, K, T, F>(
    source: &mut S,
    sink: &mut K,
    lifecycle: &mut VideoLifecycle,
    cancel: &AtomicBool,
    timeline: &Timeline<T>,
    mut transform: F,
) -> anyhow::Result<()>
where
    S: FrameSource,
    K: FrameSink,
    F: FnMut(Frame, &T, u64) -> anyhow::Result<Frame>,
{
    lifecycle.start(source.frame_count())?;
    let mut frame_index = 0_u64;
    let (width, height) = source.dimensions();
    let mut frame = Frame::new(width, height);
    loop {
        if cancel.load(Ordering::Acquire) {
            lifecycle.request_cancel()?;
            lifecycle.finish_cancelled()?;
            return Ok(());
        }
        let has_frame = match source.next_frame_into(&mut frame) {
            Ok(has_frame) => has_frame,
            Err(error) => {
                lifecycle.fail(error.to_string())?;
                return Err(error);
            }
        };
        if !has_frame {
            lifecycle.complete()?;
            return Ok(());
        }
        let processed = match transform(frame, timeline.at(frame_index), frame_index) {
            Ok(frame) => frame,
            Err(error) => {
                lifecycle.fail(error.to_string())?;
                return Err(error);
            }
        };
        if let Err(error) = sink.write_frame(&processed.data, processed.width, processed.height) {
            lifecycle.fail(error.to_string())?;
            return Err(error);
        }
        lifecycle.frame_completed()?;
        frame_index += 1;
        frame = processed;
    }
}

/// Frame source — produces RGB frames one at a time.
///
/// NOT `Send` on Linux — webcam backends hold raw fds in thread-local state.
/// The source must be driven from a dedicated thread.
pub trait FrameSource {
    /// Get the next frame as HWC u8 RGB. Returns None at EOF.
    fn next_frame(&mut self) -> Option<Frame>;

    /// Fallible decode path used by batch processing.
    fn next_frame_result(&mut self) -> anyhow::Result<Option<Frame>> {
        Ok(self.next_frame())
    }

    /// Fill a caller-owned frame buffer. Streaming decoders override this to
    /// avoid allocating one RGB vector per frame.
    fn next_frame_into(&mut self, frame: &mut Frame) -> anyhow::Result<bool> {
        match self.next_frame_result()? {
            Some(next) => {
                *frame = next;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Frame dimensions (width, height).
    fn dimensions(&self) -> (u32, u32);

    /// Frames per second.
    fn fps(&self) -> f32;

    /// Total frame count (None for live sources like webcam).
    fn frame_count(&self) -> Option<u64>;
}

/// Frame sink — consumes RGB frames for output (virtual cam, file writer).
pub trait FrameSink {
    /// Write a single RGB frame.
    fn write_frame(&mut self, frame: &[u8], width: u32, height: u32) -> anyhow::Result<()>;
}

/// A single RGB frame in HWC layout.
#[derive(Clone)]
pub struct Frame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

impl Frame {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            data: vec![0u8; (width * height * 3) as usize],
            width,
            height,
        }
    }

    pub fn from_data(data: Vec<u8>, width: u32, height: u32) -> Self {
        assert_eq!(data.len(), (width * height * 3) as usize);
        Self {
            data,
            width,
            height,
        }
    }

    pub fn pixel(&self, x: u32, y: u32) -> [u8; 3] {
        let idx = ((y * self.width + x) * 3) as usize;
        [self.data[idx], self.data[idx + 1], self.data[idx + 2]]
    }

    pub fn set_pixel(&mut self, x: u32, y: u32, rgb: [u8; 3]) {
        let idx = ((y * self.width + x) * 3) as usize;
        self.data[idx] = rgb[0];
        self.data[idx + 1] = rgb[1];
        self.data[idx + 2] = rgb[2];
    }
}

/// Video/image file reader.
///
/// Uses the `image` crate to decode frames. Supports static images (one frame)
/// and animated GIF (multiple frames). For real video files (mp4/mkv/webm)
/// you need ffmpeg-next — not wired here yet. Photo files work end-to-end.
pub struct VideoFileSource {
    path: PathBuf,
    width: u32,
    height: u32,
    fps: f32,
    frame_count: Option<u64>,
    /// Pre-decoded frames in memory (for static images and short animations).
    /// For long videos this would stream from a decoder instead.
    frames: Vec<Frame>,
    cursor: usize,
}

impl VideoFileSource {
    pub fn new(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let img = image::open(&path)
            .map_err(|e| anyhow::anyhow!("Failed to decode {}: {e}", path.display()))?;
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let frame = Frame::from_data(rgb.into_raw(), w, h);
        Ok(Self {
            path,
            width: w,
            height: h,
            fps: 30.0,
            frame_count: Some(1),
            frames: vec![frame],
            cursor: 0,
        })
    }

    /// Path of the source file.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl FrameSource for VideoFileSource {
    fn next_frame(&mut self) -> Option<Frame> {
        if self.cursor >= self.frames.len() {
            return None;
        }
        let f = self.frames[self.cursor].clone();
        self.cursor += 1;
        Some(f)
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn fps(&self) -> f32 {
        self.fps
    }

    fn frame_count(&self) -> Option<u64> {
        self.frame_count
    }
}

#[derive(Deserialize)]
struct ProbeDocument {
    streams: Vec<ProbeStream>,
}

#[derive(Deserialize)]
struct ProbeStream {
    width: u32,
    height: u32,
    avg_frame_rate: String,
    nb_frames: Option<String>,
}

pub struct FfmpegVideoSource {
    child: Child,
    stdout: ChildStdout,
    width: u32,
    height: u32,
    fps: f32,
    frame_count: Option<u64>,
    finished: bool,
}

impl FfmpegVideoSource {
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        Self::open_impl(path.as_ref(), None)
    }

    pub fn open_with_threads(
        path: impl AsRef<Path>,
        worker_threads: usize,
    ) -> anyhow::Result<Self> {
        Self::open_impl(path.as_ref(), Some(worker_threads))
    }

    fn open_impl(path: &Path, worker_threads: Option<usize>) -> anyhow::Result<Self> {
        let probe = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames",
                "-of",
                "json",
            ])
            .arg(path)
            .output()?;
        anyhow::ensure!(
            probe.status.success(),
            "ffprobe failed: {}",
            String::from_utf8_lossy(&probe.stderr).trim()
        );
        let document: ProbeDocument = serde_json::from_slice(&probe.stdout)?;
        let stream = document
            .streams
            .first()
            .ok_or_else(|| anyhow::anyhow!("video stream is missing"))?;
        let fps = parse_frame_rate(&stream.avg_frame_rate)?;
        let frame_count = stream
            .nb_frames
            .as_deref()
            .and_then(|value| value.parse().ok());
        let decode_args = worker_threads.map_or_else(
            || ffmpeg_decode_args(path),
            |threads| ffmpeg_decode_args_with_threads(path, threads),
        );
        let mut child = Command::new("ffmpeg")
            .args(decode_args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("ffmpeg stdout pipe is missing"))?;
        Ok(Self {
            child,
            stdout,
            width: stream.width,
            height: stream.height,
            fps,
            frame_count,
            finished: false,
        })
    }

    fn read_frame_into(&mut self, frame: &mut Frame) -> anyhow::Result<bool> {
        if self.finished {
            return Ok(false);
        }
        frame.width = self.width;
        frame.height = self.height;
        frame
            .data
            .resize(self.width as usize * self.height as usize * 3, 0);
        let mut offset = 0;
        while offset < frame.data.len() {
            match self.stdout.read(&mut frame.data[offset..]) {
                Ok(0) if offset == 0 => {
                    let status = self.child.wait()?;
                    self.finished = true;
                    let mut stderr = String::new();
                    if let Some(mut pipe) = self.child.stderr.take() {
                        pipe.read_to_string(&mut stderr)?;
                    }
                    anyhow::ensure!(
                        status.success(),
                        "ffmpeg decoder exited with {status}: {}",
                        stderr.trim()
                    );
                    return Ok(false);
                }
                Ok(0) => anyhow::bail!(
                    "ffmpeg produced a truncated RGB frame: {offset}/{} bytes",
                    frame.data.len()
                ),
                Ok(read) => offset += read,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(true)
    }

    fn read_frame_result(&mut self) -> anyhow::Result<Option<Frame>> {
        let mut frame = Frame::new(self.width, self.height);
        Ok(self.read_frame_into(&mut frame)?.then_some(frame))
    }
}

impl FrameSource for FfmpegVideoSource {
    fn next_frame(&mut self) -> Option<Frame> {
        match self.read_frame_result() {
            Ok(frame) => frame,
            Err(error) => {
                tracing::error!("ffmpeg decode failed: {error}");
                None
            }
        }
    }

    fn next_frame_result(&mut self) -> anyhow::Result<Option<Frame>> {
        self.read_frame_result()
    }

    fn next_frame_into(&mut self, frame: &mut Frame) -> anyhow::Result<bool> {
        self.read_frame_into(frame)
    }

    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn fps(&self) -> f32 {
        self.fps
    }

    fn frame_count(&self) -> Option<u64> {
        self.frame_count
    }
}

impl Drop for FfmpegVideoSource {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

pub struct FfmpegVideoSink {
    child: Child,
    stdin: Option<ChildStdin>,
    width: u32,
    height: u32,
}

impl FfmpegVideoSink {
    pub fn create(
        path: impl AsRef<Path>,
        width: u32,
        height: u32,
        fps: f32,
    ) -> anyhow::Result<Self> {
        Self::create_impl(path.as_ref(), width, height, fps, None)
    }

    pub fn create_with_threads(
        path: impl AsRef<Path>,
        width: u32,
        height: u32,
        fps: f32,
        worker_threads: usize,
    ) -> anyhow::Result<Self> {
        Self::create_impl(path.as_ref(), width, height, fps, Some(worker_threads))
    }

    fn create_impl(
        path: &Path,
        width: u32,
        height: u32,
        fps: f32,
        worker_threads: Option<usize>,
    ) -> anyhow::Result<Self> {
        let encode_args = worker_threads.map_or_else(
            || ffmpeg_encode_args(path, width, height, fps),
            |threads| ffmpeg_encode_args_with_threads(path, width, height, fps, threads),
        );
        let mut child = Command::new("ffmpeg")
            .args(encode_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("ffmpeg stdin pipe is missing"))?;
        Ok(Self {
            child,
            stdin: Some(stdin),
            width,
            height,
        })
    }

    pub fn finish(mut self) -> anyhow::Result<()> {
        self.stdin.take();
        let status = self.child.wait()?;
        anyhow::ensure!(status.success(), "ffmpeg encoder exited with {status}");
        Ok(())
    }
}

impl FrameSink for FfmpegVideoSink {
    fn write_frame(&mut self, frame: &[u8], width: u32, height: u32) -> anyhow::Result<()> {
        anyhow::ensure!(
            width == self.width && height == self.height,
            "encoded frame dimensions changed"
        );
        anyhow::ensure!(
            frame.len() == width as usize * height as usize * 3,
            "encoded RGB frame length is invalid"
        );
        self.stdin
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("ffmpeg encoder is already closed"))?
            .write_all(frame)?;
        Ok(())
    }
}

impl Drop for FfmpegVideoSink {
    fn drop(&mut self) {
        self.stdin.take();
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn parse_frame_rate(value: &str) -> anyhow::Result<f32> {
    let (numerator, denominator) = value
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("invalid frame rate {value}"))?;
    let numerator: f32 = numerator.parse()?;
    let denominator: f32 = denominator.parse()?;
    anyhow::ensure!(denominator > 0.0, "invalid frame rate denominator");
    Ok(numerator / denominator)
}
