//! noperson compatibility adapter over the standalone `virtualcam` crate.

#![cfg(target_os = "linux")]

use anyhow::Context;
use virtualcam::{BackendKind, Camera, PixelFormat};

pub struct VirtualCamera {
    camera: Camera,
}

impl VirtualCamera {
    pub fn open(device_num: u32, width: u32, height: u32, fps: u32) -> anyhow::Result<Self> {
        let camera = Camera::builder(width, height, f64::from(fps))
            .format(PixelFormat::RGB)
            .backend(BackendKind::V4l2)
            .device(format!("/dev/video{device_num}"))
            .build()
            .context("failed to open v4l2loopback virtual camera")?;
        Ok(Self { camera })
    }

    pub fn send_frame(&mut self, frame: &[u8]) -> anyhow::Result<()> {
        self.camera
            .send(frame)
            .context("failed to publish a V4L2 virtual-camera frame")
    }

    pub fn device_path(&self) -> String {
        self.camera.device().to_owned()
    }
}
