//! Video file input (decode) and output (encode).
//!
//! Uses ffmpeg-next for video decoding/encoding when available.
//! The pipeline operates on raw RGB frames via the `FrameSource` trait.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde::Deserialize;

pub fn ffmpeg_decode_args(path: &Path) -> Vec<String> {
    ["-v", "error", "-i"]
        .into_iter()
        .map(str::to_owned)
        .chain(std::iter::once(path.to_string_lossy().into_owned()))
        .chain(
            ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
                .into_iter()
                .map(str::to_owned),
        )
        .collect()
}

pub fn ffmpeg_encode_args(path: &Path, width: u32, height: u32, fps: f32) -> Vec<String> {
    vec![
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
        "-pix_fmt".to_owned(),
        "yuv420p".to_owned(),
        "-y".to_owned(),
        path.to_string_lossy().into_owned(),
    ]
}

/// Frame source — produces RGB frames one at a time.
///
/// NOT `Send` on Linux — webcam backends hold raw fds in thread-local state.
/// The source must be driven from a dedicated thread.
pub trait FrameSource {
    /// Get the next frame as HWC u8 RGB. Returns None at EOF.
    fn next_frame(&mut self) -> Option<Frame>;

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
}

impl FfmpegVideoSource {
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
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
        let mut child = Command::new("ffmpeg")
            .args(ffmpeg_decode_args(path))
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
        })
    }
}

impl FrameSource for FfmpegVideoSource {
    fn next_frame(&mut self) -> Option<Frame> {
        let mut data = vec![0_u8; self.width as usize * self.height as usize * 3];
        match self.stdout.read_exact(&mut data) {
            Ok(()) => Some(Frame::from_data(data, self.width, self.height)),
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => None,
            Err(error) => {
                tracing::error!("ffmpeg decode failed: {error}");
                None
            }
        }
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
        let _ = self.child.kill();
        let _ = self.child.wait();
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
        let mut child = Command::new("ffmpeg")
            .args(ffmpeg_encode_args(path.as_ref(), width, height, fps))
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
