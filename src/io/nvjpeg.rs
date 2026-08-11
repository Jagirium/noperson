//! Runtime-loaded nvJPEG decode directly into caller-owned CUDA memory.

use std::ffi::c_void;

use libloading::Library;
use nokhwa::utils::FrameFormat;

use crate::backend::{Buffer, ComputeStream, DevicePtrMut};

const NVJPEG_STATUS_SUCCESS: i32 = 0;
const NVJPEG_OUTPUT_RGBI: i32 = 5;
const NVJPEG_COMPONENTS: usize = 4;

type NvJpegHandle = *mut c_void;
type NvJpegState = *mut c_void;
type CreateSimple = unsafe extern "C" fn(*mut NvJpegHandle) -> i32;
type Destroy = unsafe extern "C" fn(NvJpegHandle) -> i32;
type StateCreate = unsafe extern "C" fn(NvJpegHandle, *mut NvJpegState) -> i32;
type StateDestroy = unsafe extern "C" fn(NvJpegState) -> i32;
type GetImageInfo = unsafe extern "C" fn(
    NvJpegHandle,
    *const u8,
    usize,
    *mut i32,
    *mut i32,
    *mut i32,
    *mut i32,
) -> i32;
type Decode = unsafe extern "C" fn(
    NvJpegHandle,
    NvJpegState,
    *const u8,
    usize,
    i32,
    *mut NvJpegImage,
    *mut c_void,
) -> i32;

#[repr(C)]
struct NvJpegImage {
    channel: [*mut u8; NVJPEG_COMPONENTS],
    pitch: [usize; NVJPEG_COMPONENTS],
}

struct NvJpegApi {
    _library: Library,
    destroy: Destroy,
    state_destroy: StateDestroy,
    get_image_info: GetImageInfo,
    decode: Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeRoute {
    NvJpeg,
    Cpu,
}

#[must_use]
pub const fn decode_route(format: FrameFormat, nvjpeg_available: bool) -> DecodeRoute {
    if matches!(format, FrameFormat::MJPEG) && nvjpeg_available {
        DecodeRoute::NvJpeg
    } else {
        DecodeRoute::Cpu
    }
}

#[cfg(target_os = "linux")]
#[must_use]
pub const fn runtime_library_names() -> &'static [&'static str] {
    &["libnvjpeg.so.12", "libnvjpeg.so"]
}

#[cfg(target_os = "windows")]
#[must_use]
pub const fn runtime_library_names() -> &'static [&'static str] {
    &["nvjpeg64_12.dll"]
}

pub fn validate_destination(
    width: u32,
    height: u32,
    destination_len: usize,
) -> anyhow::Result<usize> {
    let required = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| anyhow::anyhow!("nvJPEG RGB destination size overflow"))?;
    anyhow::ensure!(
        destination_len >= required,
        "nvJPEG RGB destination requires {required} bytes, got {destination_len}"
    );
    Ok(required)
}

/// One persistent nvJPEG handle and decoder state owned by the webcam worker.
pub struct NvJpegDecoder {
    api: NvJpegApi,
    handle: NvJpegHandle,
    state: NvJpegState,
}

impl NvJpegDecoder {
    pub fn load() -> anyhow::Result<Self> {
        let mut failures = Vec::new();
        for name in runtime_library_names() {
            // SAFETY: nvJPEG is loaded by its stable CUDA 12 soname. Every
            // resolved symbol is copied as a function pointer while the
            // owning Library remains alive in NvJpegApi.
            let library = match unsafe { Library::new(*name) } {
                Ok(library) => library,
                Err(error) => {
                    failures.push(format!("{name}: {error}"));
                    continue;
                }
            };
            return unsafe { Self::from_library(library) };
        }
        anyhow::bail!("nvJPEG runtime unavailable: {}", failures.join("; "))
    }

    unsafe fn from_library(library: Library) -> anyhow::Result<Self> {
        unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> anyhow::Result<T> {
            Ok(*unsafe { library.get::<T>(name) }?)
        }

        let create_simple: CreateSimple = unsafe { symbol(&library, b"nvjpegCreateSimple\0")? };
        let destroy: Destroy = unsafe { symbol(&library, b"nvjpegDestroy\0")? };
        let state_create: StateCreate = unsafe { symbol(&library, b"nvjpegJpegStateCreate\0")? };
        let state_destroy: StateDestroy = unsafe { symbol(&library, b"nvjpegJpegStateDestroy\0")? };
        let get_image_info: GetImageInfo = unsafe { symbol(&library, b"nvjpegGetImageInfo\0")? };
        let decode: Decode = unsafe { symbol(&library, b"nvjpegDecode\0")? };

        let mut handle = std::ptr::null_mut();
        check_status(unsafe { create_simple(&mut handle) }, "nvjpegCreateSimple")?;
        let mut state = std::ptr::null_mut();
        if let Err(error) = check_status(
            unsafe { state_create(handle, &mut state) },
            "nvjpegJpegStateCreate",
        ) {
            let _ = unsafe { destroy(handle) };
            return Err(error);
        }

        Ok(Self {
            api: NvJpegApi {
                _library: library,
                destroy,
                state_destroy,
                get_image_info,
                decode,
            },
            handle,
            state,
        })
    }

    /// Decode one JPEG to interleaved RGB in `destination` on `stream`.
    pub fn decode_into(
        &mut self,
        jpeg: &[u8],
        destination: &mut Buffer<u8>,
        stream: &ComputeStream,
    ) -> anyhow::Result<(u32, u32)> {
        anyhow::ensure!(!jpeg.is_empty(), "nvJPEG input is empty");
        stream.context().bind_to_thread()?;

        let mut components = 0;
        let mut subsampling = 0;
        let mut widths = [0_i32; NVJPEG_COMPONENTS];
        let mut heights = [0_i32; NVJPEG_COMPONENTS];
        check_status(
            unsafe {
                (self.api.get_image_info)(
                    self.handle,
                    jpeg.as_ptr(),
                    jpeg.len(),
                    &mut components,
                    &mut subsampling,
                    widths.as_mut_ptr(),
                    heights.as_mut_ptr(),
                )
            },
            "nvjpegGetImageInfo",
        )?;
        anyhow::ensure!(
            components == 1 || components == 3,
            "nvJPEG reported unsupported component count {components}"
        );
        let width = u32::try_from(widths[0]).map_err(|_| anyhow::anyhow!("invalid JPEG width"))?;
        let height =
            u32::try_from(heights[0]).map_err(|_| anyhow::anyhow!("invalid JPEG height"))?;
        anyhow::ensure!(width > 0 && height > 0, "nvJPEG reported empty geometry");
        validate_destination(width, height, destination.len())?;

        let (device_ptr, _write_guard) = destination.device_ptr_mut(stream);
        let mut image = NvJpegImage {
            channel: [
                device_ptr as usize as *mut u8,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ],
            pitch: [width as usize * 3, 0, 0, 0],
        };
        check_status(
            unsafe {
                (self.api.decode)(
                    self.handle,
                    self.state,
                    jpeg.as_ptr(),
                    jpeg.len(),
                    NVJPEG_OUTPUT_RGBI,
                    &mut image,
                    stream.cu_stream().cast(),
                )
            },
            "nvjpegDecode",
        )?;
        Ok((width, height))
    }
}

impl Drop for NvJpegDecoder {
    fn drop(&mut self) {
        if !self.state.is_null() {
            let _ = unsafe { (self.api.state_destroy)(self.state) };
        }
        if !self.handle.is_null() {
            let _ = unsafe { (self.api.destroy)(self.handle) };
        }
    }
}

fn check_status(status: i32, operation: &'static str) -> anyhow::Result<()> {
    anyhow::ensure!(
        status == NVJPEG_STATUS_SUCCESS,
        "{operation} failed with nvJPEG status {status}"
    );
    Ok(())
}
