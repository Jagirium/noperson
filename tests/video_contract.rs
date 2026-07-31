use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use noperson::io::video::{
    Frame, FrameSink, FrameSource, Timeline, VideoLifecycle, VideoState, ffmpeg_decode_args,
    ffmpeg_encode_args, ffmpeg_remux_args, process_video_frames,
};

#[test]
fn ffmpeg_decode_contract_streams_exact_rgb_frames() {
    let args = ffmpeg_decode_args(Path::new("input.mp4"));
    assert_eq!(
        args,
        [
            "-v",
            "error",
            "-i",
            "input.mp4",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-"
        ]
    );
}

struct MemorySource {
    frames: Vec<Frame>,
}

struct ErrorSource;

impl FrameSource for ErrorSource {
    fn next_frame(&mut self) -> Option<Frame> {
        None
    }
    fn next_frame_result(&mut self) -> anyhow::Result<Option<Frame>> {
        anyhow::bail!("decode exploded")
    }
    fn dimensions(&self) -> (u32, u32) {
        (1, 1)
    }
    fn fps(&self) -> f32 {
        30.0
    }
    fn frame_count(&self) -> Option<u64> {
        None
    }
}

struct ReuseSource {
    remaining: u8,
    buffer_ptr: Option<usize>,
}

impl FrameSource for ReuseSource {
    fn next_frame(&mut self) -> Option<Frame> {
        panic!("driver allocated instead of reusing")
    }
    fn next_frame_into(&mut self, frame: &mut Frame) -> anyhow::Result<bool> {
        if self.remaining == 0 {
            return Ok(false);
        }
        let ptr = frame.data.as_ptr() as usize;
        if let Some(expected) = self.buffer_ptr {
            assert_eq!(ptr, expected);
        } else {
            self.buffer_ptr = Some(ptr);
        }
        frame.data.fill(self.remaining);
        self.remaining -= 1;
        Ok(true)
    }
    fn dimensions(&self) -> (u32, u32) {
        (1, 1)
    }
    fn fps(&self) -> f32 {
        30.0
    }
    fn frame_count(&self) -> Option<u64> {
        Some(u64::from(self.remaining))
    }
}

impl FrameSource for MemorySource {
    fn next_frame(&mut self) -> Option<Frame> {
        self.frames.pop()
    }
    fn dimensions(&self) -> (u32, u32) {
        (1, 1)
    }
    fn fps(&self) -> f32 {
        30.0
    }
    fn frame_count(&self) -> Option<u64> {
        Some(self.frames.len() as u64)
    }
}

#[derive(Default)]
struct MemorySink(Vec<Vec<u8>>);

impl FrameSink for MemorySink {
    fn write_frame(&mut self, frame: &[u8], _: u32, _: u32) -> anyhow::Result<()> {
        self.0.push(frame.to_vec());
        Ok(())
    }
}

#[test]
fn video_driver_applies_timeline_and_stops_at_cancel_boundary() {
    let mut source = MemorySource {
        frames: vec![Frame::from_data(vec![1, 1, 1], 1, 1); 3],
    };
    let mut sink = MemorySink::default();
    let mut lifecycle = VideoLifecycle::default();
    let cancel = AtomicBool::new(false);
    let mut timeline = Timeline::new(1_u8);
    timeline.insert(1, 2);
    process_video_frames(
        &mut source,
        &mut sink,
        &mut lifecycle,
        &cancel,
        &timeline,
        |mut frame, marker, index| {
            frame.data[0] = *marker;
            if index == 0 {
                cancel.store(true, Ordering::Release);
            }
            Ok(frame)
        },
    )
    .unwrap();
    assert_eq!(sink.0, [vec![1, 1, 1]]);
    assert_eq!(lifecycle.state(), &VideoState::Cancelled { processed: 1 });
}

#[test]
fn video_driver_preserves_decode_errors_in_terminal_state() {
    let mut source = ErrorSource;
    let mut sink = MemorySink::default();
    let mut lifecycle = VideoLifecycle::default();
    let error = process_video_frames(
        &mut source,
        &mut sink,
        &mut lifecycle,
        &AtomicBool::new(false),
        &Timeline::new(()),
        |frame, _, _| Ok(frame),
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "decode exploded");
    assert_eq!(
        lifecycle.state(),
        &VideoState::Failed {
            processed: 0,
            message: "decode exploded".to_owned(),
        }
    );
}

#[test]
fn video_driver_reuses_one_decode_buffer_for_the_whole_stream() {
    let mut source = ReuseSource {
        remaining: 3,
        buffer_ptr: None,
    };
    let mut sink = MemorySink::default();
    process_video_frames(
        &mut source,
        &mut sink,
        &mut VideoLifecycle::default(),
        &AtomicBool::new(false),
        &Timeline::new(()),
        |frame, _, _| Ok(frame),
    )
    .unwrap();
    assert_eq!(sink.0.len(), 3);
}

#[test]
fn remux_contract_keeps_processed_video_and_optional_original_audio() {
    assert_eq!(
        ffmpeg_remux_args(
            Path::new("silent.mp4"),
            Path::new("original.mkv"),
            Path::new("release.mp4"),
        ),
        [
            "-v",
            "error",
            "-i",
            "silent.mp4",
            "-i",
            "original.mkv",
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-shortest",
            "-y",
            "release.mp4",
        ]
    );
}

#[test]
fn timeline_uses_latest_marker_at_or_before_frame() {
    let mut timeline = Timeline::new("base");
    timeline.insert(10, "ten");
    timeline.insert(25, "twenty-five");
    assert_eq!(timeline.at(0), &"base");
    assert_eq!(timeline.at(24), &"ten");
    assert_eq!(timeline.at(25), &"twenty-five");
}

#[test]
fn video_lifecycle_reports_progress_cancel_and_terminal_state() {
    let mut lifecycle = VideoLifecycle::default();
    lifecycle.start(Some(4)).unwrap();
    lifecycle.frame_completed().unwrap();
    assert_eq!(
        lifecycle.state(),
        &VideoState::Running {
            processed: 1,
            total: Some(4)
        }
    );
    lifecycle.request_cancel().unwrap();
    assert_eq!(
        lifecycle.state(),
        &VideoState::Cancelling {
            processed: 1,
            total: Some(4)
        }
    );
    lifecycle.finish_cancelled().unwrap();
    assert_eq!(lifecycle.state(), &VideoState::Cancelled { processed: 1 });
}

#[test]
fn ffmpeg_encode_contract_preserves_geometry_and_frame_rate() {
    let args = ffmpeg_encode_args(Path::new("output.mp4"), 1920, 1080, 60.0);
    assert_eq!(
        args,
        [
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            "1920x1080",
            "-r",
            "60",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            "output.mp4",
        ]
    );
}
