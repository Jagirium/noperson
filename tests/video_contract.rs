use std::path::Path;

use noperson::io::video::{ffmpeg_decode_args, ffmpeg_encode_args};

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
