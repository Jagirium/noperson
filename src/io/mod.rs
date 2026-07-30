//! I/O — video file, webcam, virtual camera output.

pub mod video;
pub mod webcam;

/// The backend implementation lives in the standalone `virtualcam` crate.
#[cfg(target_os = "linux")]
pub mod vcam_linux;
#[cfg(target_os = "linux")]
pub use vcam_linux::VirtualCamera;

#[cfg(target_os = "windows")]
pub mod vcam_windows;
#[cfg(target_os = "windows")]
pub use vcam_windows::VirtualCamera;

/// Publish the final GPU frame through the native platform backend.
///
/// Windows converts CHW directly to backend-native NV12 on CUDA. Linux reuses
/// the RGB preview frame and lets `virtualcam` own the V4L2 stream lifecycle.
#[cfg(target_os = "windows")]
pub fn send_virtual_camera_frame(
    camera: &mut VirtualCamera,
    gpu: &crate::gpu::ops::GpuOps,
    chw: &cudarc::driver::CudaSlice<f32>,
    _rgb: &[u8],
    width: u32,
    height: u32,
) -> anyhow::Result<()> {
    camera.send_gpu_frame(gpu, chw, width, height)
}

#[cfg(target_os = "linux")]
pub fn send_virtual_camera_frame(
    camera: &mut VirtualCamera,
    _gpu: &crate::gpu::ops::GpuOps,
    _chw: &cudarc::driver::CudaSlice<f32>,
    rgb: &[u8],
    _width: u32,
    _height: u32,
) -> anyhow::Result<()> {
    camera.send_frame(rgb)?;
    Ok(())
}
