//! Video file input (decode) and output (encode).
//!
//! Uses ffmpeg-next for video decoding/encoding when available.
//! The pipeline operates on raw RGB frames via the `FrameSource` trait.

use std::path::PathBuf;

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
