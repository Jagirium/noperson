//! Webcam capture via nokhwa (v4l2 on Linux).
//!
//! Produces RGB frames at the requested resolution and FPS.
//!
//! Note: `nokhwa::Camera` is NOT `Send` on Linux (v4l2 backend holds a
//! raw fd in a `RefCell`). The camera must live on the thread that created
//! it. The pipeline thread owns the camera and sends frames back via channel.

use nokhwa::Camera;
use nokhwa::pixel_format::{FormatDecoder, RgbAFormat};
use nokhwa::utils::{ApiBackend, CameraFormat, CameraIndex, RequestedFormat, RequestedFormatType};

use super::video::{Frame, FrameSource};

/// Webcam frame source — NOT Send. Must be driven from a single thread.
pub struct WebcamSource {
    camera: Camera,
    width: u32,
    height: u32,
    fps: f32,
}

fn discovery_request() -> RequestedFormat<'static> {
    RequestedFormat::new::<RgbAFormat>(RequestedFormatType::None)
}

fn select_capture_format(
    available: &[CameraFormat],
    width: u32,
    height: u32,
    fps: u32,
) -> Option<CameraFormat> {
    available
        .iter()
        .copied()
        .filter(|format| RgbAFormat::FORMATS.contains(&format.format()))
        .min_by_key(|format| {
            let resolution = format.resolution();
            let dx = i64::from(resolution.width()) - i64::from(width);
            let dy = i64::from(resolution.height()) - i64::from(height);
            let resolution_distance = (dx * dx + dy * dy) as u64;
            let fps_distance = format.frame_rate().abs_diff(fps);
            let decoder_rank = RgbAFormat::FORMATS
                .iter()
                .position(|candidate| *candidate == format.format())
                .unwrap_or(usize::MAX);
            (resolution_distance, fps_distance, decoder_rank)
        })
}

impl WebcamSource {
    /// Open a webcam by index (0 = default camera).
    pub fn new(index: usize, width: u32, height: u32, fps: f32) -> anyhow::Result<Self> {
        let mut camera = Camera::new(CameraIndex::Index(index as u32), discovery_request())
            .map_err(|e| anyhow::anyhow!("Failed to open camera {index}: {e}"))?;
        let compatible = camera
            .compatible_camera_formats()
            .map_err(|e| anyhow::anyhow!("Failed to query camera {index} formats: {e}"))?;
        let selected = select_capture_format(&compatible, width, height, fps as u32)
            .ok_or_else(|| anyhow::anyhow!("Camera {index} has no decodable color format"))?;
        camera
            .set_camera_requset(RequestedFormat::new::<RgbAFormat>(
                RequestedFormatType::Exact(selected),
            ))
            .map_err(|e| anyhow::anyhow!("Failed to configure camera {index}: {e}"))?;

        camera
            .open_stream()
            .map_err(|e| anyhow::anyhow!("Failed to open camera stream: {e}"))?;

        let actual_format = camera.camera_format();
        let actual_w = actual_format.width();
        let actual_h = actual_format.height();

        Ok(Self {
            camera,
            width: actual_w,
            height: actual_h,
            fps,
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
}

impl FrameSource for WebcamSource {
    fn next_frame(&mut self) -> Option<Frame> {
        match self.camera.frame() {
            Ok(buffer) => match buffer.decode_image::<RgbAFormat>() {
                Ok(rgba) => {
                    let mut rgb = Vec::with_capacity((self.width * self.height * 3) as usize);
                    for chunk in rgba.chunks(4) {
                        rgb.push(chunk[0]);
                        rgb.push(chunk[1]);
                        rgb.push(chunk[2]);
                    }
                    Some(Frame::from_data(rgb, self.width, self.height))
                }
                Err(_) => None,
            },
            Err(_) => None,
        }
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
    use nokhwa::utils::{FrameFormat, Resolution};

    #[test]
    fn discovery_request_accepts_yuyv_when_raw_rgb_is_unavailable() {
        let available = [CameraFormat::new(
            Resolution::new(640, 480),
            FrameFormat::YUYV,
            30,
        )];

        let selected = discovery_request().fulfill(&available);

        assert_eq!(selected, Some(available[0]));
    }

    #[test]
    fn capture_selection_prefers_requested_geometry_across_decodable_formats() {
        let yuyv_640 = CameraFormat::new(Resolution::new(640, 480), FrameFormat::YUYV, 30);
        let mjpeg_1080 = CameraFormat::new(Resolution::new(1920, 1080), FrameFormat::MJPEG, 30);

        let selected = select_capture_format(&[mjpeg_1080, yuyv_640], 640, 480, 30);

        assert_eq!(selected, Some(yuyv_640));
    }

    #[test]
    #[ignore = "requires a live camera selected by NOPERSON_CAMERA_INDEX"]
    fn configured_camera_produces_an_rgb_frame() {
        let index = std::env::var("NOPERSON_CAMERA_INDEX")
            .expect("set NOPERSON_CAMERA_INDEX")
            .parse::<usize>()
            .expect("camera index must be an integer");
        let mut camera = WebcamSource::new(index, 640, 480, 30.0).expect("open camera");

        let frame = (0..30)
            .find_map(|_| camera.next_frame())
            .expect("camera must produce a decodable frame");

        assert_eq!(frame.data.len(), (frame.width * frame.height * 3) as usize);
        assert_eq!(camera.dimensions(), (frame.width, frame.height));
    }
}
