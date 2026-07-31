//! noperson GPU adapter over the standalone `virtualcam` crate.

#![cfg(target_os = "windows")]

use anyhow::{Context, ensure};
use cudarc::driver::CudaSlice;
use virtualcam::{Camera, PixelFormat};

use crate::gpu::ops::GpuOps;

pub struct VirtualCamera {
    camera: Camera,
    device_nv12: Option<CudaSlice<u8>>,
}

impl VirtualCamera {
    pub fn open(_device_num: u32, width: u32, height: u32, fps: u32) -> anyhow::Result<Self> {
        let camera = Camera::builder(width, height, f64::from(fps))
            .format(PixelFormat::RGB)
            .build()
            .context("failed to open a Windows virtual-camera backend")?;
        Ok(Self {
            camera,
            device_nv12: None,
        })
    }

    pub fn send_frame(&mut self, frame: &[u8]) -> anyhow::Result<()> {
        self.camera
            .send(frame)
            .context("failed to publish a virtual-camera frame")
    }

    pub fn send_nv12(&mut self, frame: &[u8]) -> anyhow::Result<()> {
        self.camera
            .send_native(frame)
            .context("failed to publish native NV12")
    }

    pub fn send_gpu_frame(
        &mut self,
        gpu: &GpuOps,
        frame: &CudaSlice<f32>,
        width: u32,
        height: u32,
    ) -> anyhow::Result<()> {
        ensure!(
            self.camera.native_format() == PixelFormat::NV12,
            "selected backend does not accept native NV12"
        );
        let (output_width, output_height) = self.camera.native_dimensions();
        let nv12_len = PixelFormat::NV12.frame_size(output_width, output_height);
        if self
            .device_nv12
            .as_ref()
            .is_none_or(|buffer| buffer.len() != nv12_len)
        {
            self.device_nv12 = Some(gpu.alloc_zeros_u8(nv12_len)?);
        }

        let device = self
            .device_nv12
            .as_mut()
            .expect("NV12 device buffer initialized");
        gpu.chw_f32_to_nv12_scaled(frame, device, height, width, output_height, output_width)?;
        self.camera
            .send_native_gpu(&gpu.stream, device)
            .context("failed to publish a GPU-produced virtual-camera frame")
    }

    pub fn device_path(&self) -> String {
        self.camera.device().to_owned()
    }

    pub fn close(self) {}
}
