//! Safe ownership and metadata contract for the in-process video backend.
//!
//! Native FFmpeg and NVCodec handles never cross this module's public API.

use std::collections::BTreeSet;

#[cfg(target_os = "linux")]
use std::ffi::{CStr, CString, c_char, c_int, c_void};
#[cfg(target_os = "linux")]
use std::os::unix::ffi::OsStrExt;
#[cfg(target_os = "linux")]
use std::path::Path;
#[cfg(target_os = "linux")]
use std::ptr::NonNull;
#[cfg(target_os = "linux")]
use std::sync::Arc;

#[cfg(target_os = "linux")]
use crate::backend::ComputeStream;
#[cfg(target_os = "linux")]
use crate::backend::cuda::{driver_result as result, driver_sys as sys};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Nv12,
    P010,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorRange {
    Full,
    Limited,
    #[default]
    Unspecified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMatrix {
    Bt709,
    Bt601,
    Bt2020NonConstantLuminance,
    #[default]
    Unspecified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorPrimaries {
    Bt709,
    Bt2020,
    #[default]
    Unspecified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransferCharacteristic {
    Bt709,
    Smpte2084,
    Hlg,
    #[default]
    Unspecified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChromaLocation {
    Left,
    Center,
    TopLeft,
    #[default]
    Unspecified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VideoColorInfo {
    pub range: ColorRange,
    pub matrix: ColorMatrix,
    pub primaries: ColorPrimaries,
    pub transfer: TransferCharacteristic,
    pub chroma_location: ChromaLocation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoDescriptor {
    pub width: u32,
    pub height: u32,
    pub pixel_format: PixelFormat,
    pub color: VideoColorInfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    H264,
    Hevc,
    Av1,
    Vp9,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoStreamInfo {
    pub index: i32,
    pub codec: VideoCodec,
    pub width: u32,
    pub height: u32,
    pub bit_depth: u32,
    pub time_base_num: i32,
    pub time_base_den: i32,
    pub frame_rate_num: u32,
    pub frame_rate_den: u32,
    pub frame_count: Option<u64>,
    pub duration_ts: Option<i64>,
    pub color: VideoColorInfo,
    pub extradata: Vec<u8>,
}

impl VideoStreamInfo {
    pub fn fps(&self) -> Option<f64> {
        (self.frame_rate_num > 0 && self.frame_rate_den > 0)
            .then(|| f64::from(self.frame_rate_num) / f64::from(self.frame_rate_den))
    }

    pub fn duration_seconds(&self) -> Option<f64> {
        let duration = self.duration_ts?;
        (duration > 0 && self.time_base_num > 0 && self.time_base_den > 0).then(|| {
            duration as f64 * f64::from(self.time_base_num) / f64::from(self.time_base_den)
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedVideoPacket {
    pub data: Vec<u8>,
    pub pts: i64,
    pub dts: i64,
    pub duration: i64,
    pub stream_index: i32,
    pub is_keyframe: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvCodecCapabilities {
    pub cuda_driver: bool,
    pub nvdec: bool,
    pub nvenc: bool,
    pub nvenc_api_major: u32,
    pub nvenc_api_minor: u32,
}

impl VideoDescriptor {
    pub fn new(
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
        color: VideoColorInfo,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(width > 0 && height > 0, "video dimensions must be non-zero");
        anyhow::ensure!(
            width.is_multiple_of(2) && height.is_multiple_of(2),
            "4:2:0 video dimensions must be even: {width}x{height}"
        );
        Ok(Self {
            width,
            height,
            pixel_format,
            color,
        })
    }

    pub const fn bytes_per_sample(self) -> usize {
        match self.pixel_format {
            PixelFormat::Nv12 => 1,
            PixelFormat::P010 => 2,
        }
    }

    pub const fn luma_samples(self) -> usize {
        self.width as usize * self.height as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SurfaceToken(u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Lifecycle {
    #[default]
    Idle,
    Open,
    Draining,
    Finished,
}

#[derive(Debug, Default)]
pub struct NativeVideoState {
    lifecycle: Lifecycle,
    next_surface: u64,
    mapped_surfaces: BTreeSet<SurfaceToken>,
}

impl NativeVideoState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn open(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.lifecycle == Lifecycle::Idle,
            "video pipeline is not idle"
        );
        self.lifecycle = Lifecycle::Open;
        Ok(())
    }

    pub fn begin_draining(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.lifecycle == Lifecycle::Open,
            "video pipeline is not open"
        );
        self.lifecycle = Lifecycle::Draining;
        Ok(())
    }

    pub fn map_surface(&mut self) -> anyhow::Result<SurfaceToken> {
        anyhow::ensure!(
            matches!(self.lifecycle, Lifecycle::Open | Lifecycle::Draining),
            "video pipeline does not accept decoded surfaces"
        );
        let token = SurfaceToken(self.next_surface);
        self.next_surface = self
            .next_surface
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("video surface token overflow"))?;
        self.mapped_surfaces.insert(token);
        Ok(token)
    }

    pub fn unmap_surface(&mut self, token: SurfaceToken) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.mapped_surfaces.remove(&token),
            "video surface token is not mapped"
        );
        Ok(())
    }

    pub fn finish(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            matches!(self.lifecycle, Lifecycle::Open | Lifecycle::Draining),
            "video pipeline is not active"
        );
        anyhow::ensure!(
            self.mapped_surfaces.is_empty(),
            "cannot finish video pipeline while surfaces are mapped"
        );
        self.lifecycle = Lifecycle::Finished;
        Ok(())
    }

    pub fn is_finished(&self) -> bool {
        self.lifecycle == Lifecycle::Finished
    }
}

#[cfg(target_os = "linux")]
const NATIVE_VIDEO_ABI_VERSION: u32 = 1;
#[cfg(target_os = "linux")]
const NATIVE_VIDEO_ERROR_CAPACITY: usize = 1024;
#[cfg(target_os = "linux")]
const AV_PKT_FLAG_KEY: u32 = 1;

#[cfg(target_os = "linux")]
#[repr(C)]
struct NativeError {
    code: i32,
    message: [c_char; NATIVE_VIDEO_ERROR_CAPACITY],
}

#[cfg(target_os = "linux")]
impl NativeError {
    fn empty() -> Self {
        Self {
            code: 0,
            message: [0; NATIVE_VIDEO_ERROR_CAPACITY],
        }
    }

    fn into_anyhow(self, operation: &str) -> anyhow::Error {
        let message = unsafe { CStr::from_ptr(self.message.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        if message.is_empty() {
            anyhow::anyhow!("{operation} failed with native video error {}", self.code)
        } else {
            anyhow::anyhow!("{operation} failed: {message}")
        }
    }
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct NativeStreamInfo {
    abi_version: u32,
    index: i32,
    codec: i32,
    width: u32,
    height: u32,
    bit_depth: u32,
    time_base_num: i32,
    time_base_den: i32,
    frame_rate_num: u32,
    frame_rate_den: u32,
    frame_count: i64,
    duration_ts: i64,
    color_range: i32,
    color_matrix: i32,
    color_primaries: i32,
    color_transfer: i32,
    chroma_location: i32,
    extradata: *const u8,
    extradata_len: usize,
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct NativePacket {
    data: *const u8,
    data_len: usize,
    pts: i64,
    dts: i64,
    duration: i64,
    stream_index: i32,
    flags: u32,
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct NativeNvCodecCapabilities {
    abi_version: u32,
    cuda_driver: u8,
    nvdec: u8,
    nvenc: u8,
    reserved: u8,
    nvenc_api_major: u32,
    nvenc_api_minor: u32,
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct NativeCudaVideoSurface {
    abi_version: u32,
    pixel_format: u32,
    device_ptr: u64,
    pitch: u32,
    width: u32,
    height: u32,
    timestamp_100ns: i64,
    picture_index: i32,
    progressive: u32,
}

#[cfg(target_os = "linux")]
#[repr(C)]
struct NativeNvEncoderConfig {
    abi_version: u32,
    codec: i32,
    pixel_format: u32,
    width: u32,
    height: u32,
    frame_rate_num: u32,
    frame_rate_den: u32,
    time_base_num: i32,
    time_base_den: i32,
    constant_qp: u32,
    gop_length: u32,
    color_range: i32,
    color_matrix: i32,
    color_primaries: i32,
    color_transfer: i32,
    chroma_location: i32,
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn np_video_nvcodec_probe(
        out_capabilities: *mut NativeNvCodecCapabilities,
        error: *mut NativeError,
    ) -> c_int;
    fn np_video_nvdecoder_open(
        cuda_context: *mut c_void,
        cuda_stream: *mut c_void,
        codec: i32,
        out_decoder: *mut *mut c_void,
        error: *mut NativeError,
    ) -> c_int;
    fn np_video_nvdecoder_send(
        decoder: *mut c_void,
        data: *const u8,
        data_len: usize,
        timestamp_100ns: i64,
        error: *mut NativeError,
    ) -> c_int;
    fn np_video_nvdecoder_flush(decoder: *mut c_void, error: *mut NativeError) -> c_int;
    fn np_video_nvdecoder_peek(
        decoder: *mut c_void,
        out_picture_index: *mut i32,
        error: *mut NativeError,
    ) -> c_int;
    fn np_video_nvdecoder_map(
        decoder: *mut c_void,
        out_surface: *mut NativeCudaVideoSurface,
        error: *mut NativeError,
    ) -> c_int;
    fn np_video_nvdecoder_unmap(
        decoder: *mut c_void,
        device_ptr: u64,
        error: *mut NativeError,
    ) -> c_int;
    fn np_video_nvdecoder_close(decoder: *mut c_void);
    fn np_video_nvencoder_open(
        cuda_context: *mut c_void,
        cuda_stream: *mut c_void,
        config: *const NativeNvEncoderConfig,
        out_encoder: *mut *mut c_void,
        out_video: *mut NativeStreamInfo,
        error: *mut NativeError,
    ) -> c_int;
    fn np_video_nvencoder_encode(
        encoder: *mut c_void,
        device_ptr: u64,
        pitch: u32,
        pts: i64,
        duration: i64,
        out_packet: *mut NativePacket,
        error: *mut NativeError,
    ) -> c_int;
    fn np_video_nvencoder_flush(encoder: *mut c_void, error: *mut NativeError) -> c_int;
    fn np_video_nvencoder_receive(
        encoder: *mut c_void,
        out_packet: *mut NativePacket,
        error: *mut NativeError,
    ) -> c_int;
    fn np_video_nvencoder_close(encoder: *mut c_void);
    fn np_video_demux_open(
        path: *const c_char,
        out_demuxer: *mut *mut c_void,
        out_video: *mut NativeStreamInfo,
        error: *mut NativeError,
    ) -> c_int;
    fn np_video_demux_read(
        demuxer: *mut c_void,
        out_packet: *mut NativePacket,
        error: *mut NativeError,
    ) -> c_int;
    fn np_video_demux_read_decode(
        demuxer: *mut c_void,
        out_packet: *mut NativePacket,
        error: *mut NativeError,
    ) -> c_int;
    fn np_video_demux_close(demuxer: *mut c_void);
    fn np_video_mux_open(
        path: *const c_char,
        video: *const NativeStreamInfo,
        out_muxer: *mut *mut c_void,
        error: *mut NativeError,
    ) -> c_int;
    fn np_video_mux_write(
        muxer: *mut c_void,
        packet: *const NativePacket,
        error: *mut NativeError,
    ) -> c_int;
    fn np_video_mux_finish(muxer: *mut c_void, error: *mut NativeError) -> c_int;
    fn np_video_mux_close(muxer: *mut c_void);
    fn np_video_remux_audio(
        video_path: *const c_char,
        source_path: *const c_char,
        output_path: *const c_char,
        error: *mut NativeError,
    ) -> c_int;
}

#[cfg(target_os = "linux")]
pub fn remux_audio(
    video_path: impl AsRef<Path>,
    source_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let path = |value: &Path| {
        CString::new(value.as_os_str().as_bytes())
            .map_err(|_| anyhow::anyhow!("media path contains a NUL byte"))
    };
    let video = path(video_path.as_ref())?;
    let source = path(source_path.as_ref())?;
    let output = path(output_path.as_ref())?;
    let mut error = NativeError::empty();
    let status = unsafe {
        np_video_remux_audio(video.as_ptr(), source.as_ptr(), output.as_ptr(), &mut error)
    };
    if status < 0 {
        return Err(error.into_anyhow("remux source audio"));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
impl NvCodecCapabilities {
    pub fn probe() -> anyhow::Result<Self> {
        let mut native = std::mem::MaybeUninit::<NativeNvCodecCapabilities>::zeroed();
        let mut error = NativeError::empty();
        let status = unsafe { np_video_nvcodec_probe(native.as_mut_ptr(), &mut error) };
        if status < 0 {
            return Err(error.into_anyhow("probe NVCodec driver APIs"));
        }
        let native = unsafe { native.assume_init() };
        anyhow::ensure!(
            native.abi_version == NATIVE_VIDEO_ABI_VERSION,
            "native video ABI mismatch: expected {}, got {}",
            NATIVE_VIDEO_ABI_VERSION,
            native.abi_version
        );
        Ok(Self {
            cuda_driver: native.cuda_driver != 0,
            nvdec: native.nvdec != 0,
            nvenc: native.nvenc != 0,
            nvenc_api_major: native.nvenc_api_major,
            nvenc_api_minor: native.nvenc_api_minor,
        })
    }
}

#[cfg(target_os = "linux")]
pub struct NvDecoder {
    inner: std::sync::Arc<NvDecoderInner>,
}

#[cfg(target_os = "linux")]
struct NvDecoderInner {
    handle: NonNull<c_void>,
}

#[cfg(target_os = "linux")]
unsafe impl Send for NvDecoderInner {}
#[cfg(target_os = "linux")]
unsafe impl Sync for NvDecoderInner {}

#[cfg(target_os = "linux")]
impl NvDecoder {
    /// Opens NVDEC against an existing CUDA context and processing stream.
    ///
    /// # Safety
    /// Both handles must remain alive until this decoder and every mapped
    /// surface borrowed from it have been dropped.
    pub unsafe fn open(
        cuda_context: *mut c_void,
        cuda_stream: *mut c_void,
        codec: VideoCodec,
    ) -> anyhow::Result<Self> {
        let mut handle = std::ptr::null_mut();
        let mut error = NativeError::empty();
        let status = unsafe {
            np_video_nvdecoder_open(
                cuda_context,
                cuda_stream,
                native_codec(codec),
                &mut handle,
                &mut error,
            )
        };
        if status < 0 {
            return Err(error.into_anyhow("open NVDEC session"));
        }
        Ok(Self {
            inner: std::sync::Arc::new(NvDecoderInner {
                handle: NonNull::new(handle)
                    .ok_or_else(|| anyhow::anyhow!("NVDEC returned a null decoder"))?,
            }),
        })
    }

    pub fn send_packet(
        &mut self,
        packet: &EncodedVideoPacket,
        time_base_num: i32,
        time_base_den: i32,
    ) -> anyhow::Result<()> {
        let timestamp = rescale_timestamp(packet.pts, time_base_num, time_base_den, 10_000_000)?;
        let mut error = NativeError::empty();
        let status = unsafe {
            np_video_nvdecoder_send(
                self.inner.handle.as_ptr(),
                packet.data.as_ptr(),
                packet.data.len(),
                timestamp,
                &mut error,
            )
        };
        if status < 0 {
            return Err(error.into_anyhow("submit NVDEC packet"));
        }
        Ok(())
    }

    pub fn flush(&mut self) -> anyhow::Result<()> {
        let mut error = NativeError::empty();
        let status = unsafe { np_video_nvdecoder_flush(self.inner.handle.as_ptr(), &mut error) };
        if status < 0 {
            return Err(error.into_anyhow("flush NVDEC session"));
        }
        Ok(())
    }

    pub fn next_frame(&mut self) -> anyhow::Result<Option<MappedVideoSurface>> {
        let mut native = std::mem::MaybeUninit::<NativeCudaVideoSurface>::zeroed();
        let mut error = NativeError::empty();
        let status = unsafe {
            np_video_nvdecoder_map(self.inner.handle.as_ptr(), native.as_mut_ptr(), &mut error)
        };
        if status < 0 {
            return Err(error.into_anyhow("map NVDEC surface"));
        }
        if status == 0 {
            return Ok(None);
        }
        let native = unsafe { native.assume_init() };
        anyhow::ensure!(
            native.abi_version == NATIVE_VIDEO_ABI_VERSION,
            "native video ABI mismatch: expected {}, got {}",
            NATIVE_VIDEO_ABI_VERSION,
            native.abi_version
        );
        let pixel_format = match native.pixel_format {
            1 => PixelFormat::Nv12,
            2 => PixelFormat::P010,
            value => anyhow::bail!("NVDEC returned unknown pixel format {value}"),
        };
        Ok(Some(MappedVideoSurface {
            decoder: std::sync::Arc::clone(&self.inner),
            device_ptr: native.device_ptr,
            pitch: native.pitch,
            width: native.width,
            height: native.height,
            timestamp_100ns: native.timestamp_100ns,
            picture_index: native.picture_index,
            pixel_format,
        }))
    }

    pub fn next_picture_index(&self) -> anyhow::Result<Option<i32>> {
        let mut picture_index = 0;
        let mut error = NativeError::empty();
        let status = unsafe {
            np_video_nvdecoder_peek(self.inner.handle.as_ptr(), &mut picture_index, &mut error)
        };
        if status < 0 {
            return Err(error.into_anyhow("peek NVDEC surface"));
        }
        Ok((status > 0).then_some(picture_index))
    }
}

#[cfg(target_os = "linux")]
impl Drop for NvDecoderInner {
    fn drop(&mut self) {
        unsafe { np_video_nvdecoder_close(self.handle.as_ptr()) };
    }
}

#[cfg(target_os = "linux")]
pub struct MappedVideoSurface {
    decoder: std::sync::Arc<NvDecoderInner>,
    device_ptr: u64,
    pitch: u32,
    width: u32,
    height: u32,
    timestamp_100ns: i64,
    picture_index: i32,
    pixel_format: PixelFormat,
}

#[cfg(target_os = "linux")]
impl MappedVideoSurface {
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    pub fn pitch(&self) -> u32 {
        self.pitch
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn timestamp_100ns(&self) -> i64 {
        self.timestamp_100ns
    }

    pub fn picture_index(&self) -> i32 {
        self.picture_index
    }

    pub fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }
}

#[cfg(target_os = "linux")]
impl Drop for MappedVideoSurface {
    fn drop(&mut self) {
        let mut error = NativeError::empty();
        let status = unsafe {
            np_video_nvdecoder_unmap(self.decoder.handle.as_ptr(), self.device_ptr, &mut error)
        };
        if status < 0 {
            tracing::warn!(
                "failed to release NVDEC surface: {}",
                error.into_anyhow("unmap")
            );
        }
    }
}

/// A pitch-linear CUDA allocation compatible with NVENC resource registration.
///
/// NVENC does not accept allocations from CUDA's stream-ordered memory pool, so
/// encoder inputs deliberately use the synchronous driver allocator. Declare
/// these surfaces before `NvEncoder`: Rust's reverse drop order then guarantees
/// NVENC unregisters every resource before the underlying CUDA memory is freed.
#[cfg(target_os = "linux")]
pub struct NvEncoderInputSurface {
    stream: Arc<ComputeStream>,
    device_ptr: u64,
    pitch: u32,
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
}

#[cfg(target_os = "linux")]
impl NvEncoderInputSurface {
    pub fn new(
        stream: Arc<ComputeStream>,
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
    ) -> anyhow::Result<Self> {
        VideoDescriptor::new(width, height, pixel_format, VideoColorInfo::default())?;
        let row_bytes = width as usize
            * match pixel_format {
                PixelFormat::Nv12 => 1,
                PixelFormat::P010 => 2,
            };
        let allocation_rows = height as usize + height as usize / 2;
        stream.context().bind_to_thread()?;
        let mut device_ptr = 0;
        let mut pitch = 0;
        unsafe {
            sys::cuMemAllocPitch_v2(&mut device_ptr, &mut pitch, row_bytes, allocation_rows, 16)
                .result()?;
        }
        let pitch = match u32::try_from(pitch) {
            Ok(pitch) => pitch,
            Err(error) => {
                unsafe { result::free_sync(device_ptr)? };
                return Err(error.into());
            }
        };
        Ok(Self {
            stream,
            device_ptr,
            pitch,
            width,
            height,
            pixel_format,
        })
    }

    pub const fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    pub const fn pitch(&self) -> u32 {
        self.pitch
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    pub const fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }
}

#[cfg(target_os = "linux")]
impl Drop for NvEncoderInputSurface {
    fn drop(&mut self) {
        if let Err(error) = self.stream.synchronize() {
            tracing::warn!("failed to synchronize NVENC input surface: {error}");
        }
        if let Err(error) = self.stream.context().bind_to_thread() {
            tracing::warn!("failed to bind CUDA context before freeing NVENC surface: {error}");
            return;
        }
        if let Err(error) = unsafe { result::free_sync(self.device_ptr) } {
            tracing::warn!("failed to free NVENC input surface: {error}");
        }
    }
}

#[cfg(target_os = "linux")]
fn native_codec(codec: VideoCodec) -> i32 {
    match codec {
        VideoCodec::H264 => 1,
        VideoCodec::Hevc => 2,
        VideoCodec::Av1 => 3,
        VideoCodec::Vp9 => 4,
    }
}

#[cfg(target_os = "linux")]
fn rescale_timestamp(
    value: i64,
    numerator: i32,
    denominator: i32,
    scale: i64,
) -> anyhow::Result<i64> {
    anyhow::ensure!(numerator > 0 && denominator > 0, "invalid video time base");
    if value == i64::MIN {
        return Ok(0);
    }
    let scaled = i128::from(value)
        .checked_mul(i128::from(numerator))
        .and_then(|value| value.checked_mul(i128::from(scale)))
        .ok_or_else(|| anyhow::anyhow!("video timestamp overflow"))?
        / i128::from(denominator);
    i64::try_from(scaled).map_err(|_| anyhow::anyhow!("video timestamp overflow"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvEncoderConfig {
    pub codec: VideoCodec,
    pub pixel_format: PixelFormat,
    pub width: u32,
    pub height: u32,
    pub frame_rate_num: u32,
    pub frame_rate_den: u32,
    pub time_base_num: i32,
    pub time_base_den: i32,
    pub constant_qp: u32,
    pub gop_length: u32,
    pub color: VideoColorInfo,
}

impl NvEncoderConfig {
    pub fn h264_quality(
        width: u32,
        height: u32,
        frame_rate_num: u32,
        frame_rate_den: u32,
        time_base_num: i32,
        time_base_den: i32,
    ) -> Self {
        let frames_per_second = frame_rate_num.div_ceil(frame_rate_den.max(1));
        Self {
            codec: VideoCodec::H264,
            pixel_format: PixelFormat::Nv12,
            width,
            height,
            frame_rate_num,
            frame_rate_den,
            time_base_num,
            time_base_den,
            constant_qp: 18,
            gop_length: frames_per_second.saturating_mul(10).max(1),
            color: VideoColorInfo::default(),
        }
    }

    pub fn with_color(mut self, color: VideoColorInfo) -> Self {
        self.color = color;
        self
    }
}

#[cfg(target_os = "linux")]
pub struct NvEncoder {
    handle: NonNull<c_void>,
    video: VideoStreamInfo,
    finished: bool,
}

#[cfg(target_os = "linux")]
impl NvEncoder {
    /// Opens NVENC against an existing CUDA context and processing stream.
    ///
    /// # Safety
    /// Both CUDA handles and every device allocation submitted to the encoder
    /// must outlive this encoder or be retained until the corresponding call returns.
    pub unsafe fn open(
        cuda_context: *mut c_void,
        cuda_stream: *mut c_void,
        config: NvEncoderConfig,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            config.width > 0 && config.height > 0,
            "NVENC dimensions must be non-zero"
        );
        anyhow::ensure!(
            config.width.is_multiple_of(2) && config.height.is_multiple_of(2),
            "NVENC 4:2:0 dimensions must be even"
        );
        let native_config = NativeNvEncoderConfig {
            abi_version: NATIVE_VIDEO_ABI_VERSION,
            codec: native_codec(config.codec),
            pixel_format: match config.pixel_format {
                PixelFormat::Nv12 => 1,
                PixelFormat::P010 => 2,
            },
            width: config.width,
            height: config.height,
            frame_rate_num: config.frame_rate_num,
            frame_rate_den: config.frame_rate_den,
            time_base_num: config.time_base_num,
            time_base_den: config.time_base_den,
            constant_qp: config.constant_qp,
            gop_length: config.gop_length,
            color_range: native_color_range(config.color.range),
            color_matrix: native_color_matrix(config.color.matrix),
            color_primaries: native_color_primaries(config.color.primaries),
            color_transfer: native_color_transfer(config.color.transfer),
            chroma_location: native_chroma_location(config.color.chroma_location),
        };
        let mut handle = std::ptr::null_mut();
        let mut native_video = std::mem::MaybeUninit::<NativeStreamInfo>::zeroed();
        let mut error = NativeError::empty();
        let status = unsafe {
            np_video_nvencoder_open(
                cuda_context,
                cuda_stream,
                &native_config,
                &mut handle,
                native_video.as_mut_ptr(),
                &mut error,
            )
        };
        if status < 0 {
            return Err(error.into_anyhow("open NVENC session"));
        }
        let handle =
            NonNull::new(handle).ok_or_else(|| anyhow::anyhow!("NVENC returned a null encoder"))?;
        let native_video = unsafe { native_video.assume_init() };
        let video = match video_stream_from_native(&native_video) {
            Ok(video) => video,
            Err(error) => {
                unsafe { np_video_nvencoder_close(handle.as_ptr()) };
                return Err(error);
            }
        };
        Ok(Self {
            handle,
            video,
            finished: false,
        })
    }

    pub fn video_stream(&self) -> &VideoStreamInfo {
        &self.video
    }

    /// Encodes a pitch-linear CUDA allocation containing NV12 or P010.
    ///
    /// # Safety
    /// `device_ptr` must address the configured image format and dimensions.
    /// Keep a ring of at least five allocations: NVENC may retain four inputs
    /// until a later call returns the corresponding encoded packet.
    pub unsafe fn encode_device_frame(
        &mut self,
        device_ptr: u64,
        pitch: u32,
        pts: i64,
        duration: i64,
    ) -> anyhow::Result<Option<EncodedVideoPacket>> {
        anyhow::ensure!(!self.finished, "NVENC session is already finished");
        let mut native = std::mem::MaybeUninit::<NativePacket>::zeroed();
        let mut error = NativeError::empty();
        let status = unsafe {
            np_video_nvencoder_encode(
                self.handle.as_ptr(),
                device_ptr,
                pitch,
                pts,
                duration,
                native.as_mut_ptr(),
                &mut error,
            )
        };
        if status < 0 {
            return Err(error.into_anyhow("encode CUDA frame with NVENC"));
        }
        if status == 0 {
            return Ok(None);
        }
        let native = unsafe { native.assume_init() };
        Ok(Some(copy_native_packet(&native)?))
    }

    pub fn finish(&mut self) -> anyhow::Result<Vec<EncodedVideoPacket>> {
        if self.finished {
            return Ok(Vec::new());
        }
        let mut error = NativeError::empty();
        let status = unsafe { np_video_nvencoder_flush(self.handle.as_ptr(), &mut error) };
        if status < 0 {
            return Err(error.into_anyhow("flush NVENC session"));
        }
        let mut packets = Vec::new();
        loop {
            let mut native = std::mem::MaybeUninit::<NativePacket>::zeroed();
            let mut error = NativeError::empty();
            let status = unsafe {
                np_video_nvencoder_receive(self.handle.as_ptr(), native.as_mut_ptr(), &mut error)
            };
            if status < 0 {
                return Err(error.into_anyhow("receive NVENC packet"));
            }
            if status == 0 {
                break;
            }
            let native = unsafe { native.assume_init() };
            packets.push(copy_native_packet(&native)?);
        }
        self.finished = true;
        Ok(packets)
    }
}

#[cfg(target_os = "linux")]
impl Drop for NvEncoder {
    fn drop(&mut self) {
        unsafe { np_video_nvencoder_close(self.handle.as_ptr()) };
    }
}

#[cfg(target_os = "linux")]
fn video_stream_from_native(native: &NativeStreamInfo) -> anyhow::Result<VideoStreamInfo> {
    anyhow::ensure!(
        native.abi_version == NATIVE_VIDEO_ABI_VERSION,
        "native video ABI mismatch: expected {}, got {}",
        NATIVE_VIDEO_ABI_VERSION,
        native.abi_version
    );
    let codec = match native.codec {
        1 => VideoCodec::H264,
        2 => VideoCodec::Hevc,
        3 => VideoCodec::Av1,
        4 => VideoCodec::Vp9,
        value => anyhow::bail!("native video returned unknown codec {value}"),
    };
    let extradata = if native.extradata_len == 0 {
        Vec::new()
    } else {
        anyhow::ensure!(
            !native.extradata.is_null(),
            "native video returned null extradata"
        );
        unsafe { std::slice::from_raw_parts(native.extradata, native.extradata_len) }.to_vec()
    };
    Ok(VideoStreamInfo {
        index: native.index,
        codec,
        width: native.width,
        height: native.height,
        bit_depth: native.bit_depth,
        time_base_num: native.time_base_num,
        time_base_den: native.time_base_den,
        frame_rate_num: native.frame_rate_num,
        frame_rate_den: native.frame_rate_den,
        frame_count: u64::try_from(native.frame_count)
            .ok()
            .filter(|count| *count > 0),
        duration_ts: (native.duration_ts > 0).then_some(native.duration_ts),
        color: map_color_info(native),
        extradata,
    })
}

#[cfg(target_os = "linux")]
fn copy_native_packet(native: &NativePacket) -> anyhow::Result<EncodedVideoPacket> {
    anyhow::ensure!(
        native.data_len == 0 || !native.data.is_null(),
        "native video returned a null packet buffer"
    );
    let data = if native.data_len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(native.data, native.data_len) }.to_vec()
    };
    Ok(EncodedVideoPacket {
        data,
        pts: native.pts,
        dts: native.dts,
        duration: native.duration,
        stream_index: native.stream_index,
        is_keyframe: native.flags & AV_PKT_FLAG_KEY != 0,
    })
}

#[cfg(target_os = "linux")]
pub struct NativeDemuxer {
    handle: NonNull<c_void>,
    video: VideoStreamInfo,
}

#[cfg(target_os = "linux")]
impl NativeDemuxer {
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = CString::new(path.as_ref().as_os_str().as_bytes())
            .map_err(|_| anyhow::anyhow!("media path contains a NUL byte"))?;
        let mut handle = std::ptr::null_mut();
        let mut native = std::mem::MaybeUninit::<NativeStreamInfo>::zeroed();
        let mut error = NativeError::empty();
        let status = unsafe {
            np_video_demux_open(path.as_ptr(), &mut handle, native.as_mut_ptr(), &mut error)
        };
        if status < 0 {
            return Err(error.into_anyhow("open native demuxer"));
        }
        let handle = NonNull::new(handle)
            .ok_or_else(|| anyhow::anyhow!("native demuxer returned a null handle"))?;
        let native = unsafe { native.assume_init() };
        if native.abi_version != NATIVE_VIDEO_ABI_VERSION {
            unsafe { np_video_demux_close(handle.as_ptr()) };
            anyhow::bail!(
                "native video ABI mismatch: expected {}, got {}",
                NATIVE_VIDEO_ABI_VERSION,
                native.abi_version
            );
        }
        let codec = match native.codec {
            1 => VideoCodec::H264,
            2 => VideoCodec::Hevc,
            3 => VideoCodec::Av1,
            4 => VideoCodec::Vp9,
            value => {
                unsafe { np_video_demux_close(handle.as_ptr()) };
                anyhow::bail!("native demuxer returned unknown codec {value}");
            }
        };
        let extradata = if native.extradata_len == 0 {
            Vec::new()
        } else {
            if native.extradata.is_null() {
                unsafe { np_video_demux_close(handle.as_ptr()) };
                anyhow::bail!("native demuxer returned null codec extradata");
            }
            unsafe { std::slice::from_raw_parts(native.extradata, native.extradata_len) }.to_vec()
        };
        let video = VideoStreamInfo {
            index: native.index,
            codec,
            width: native.width,
            height: native.height,
            bit_depth: native.bit_depth,
            time_base_num: native.time_base_num,
            time_base_den: native.time_base_den,
            frame_rate_num: native.frame_rate_num,
            frame_rate_den: native.frame_rate_den,
            frame_count: u64::try_from(native.frame_count)
                .ok()
                .filter(|count| *count > 0),
            duration_ts: (native.duration_ts > 0).then_some(native.duration_ts),
            color: map_color_info(&native),
            extradata,
        };
        Ok(Self { handle, video })
    }

    pub fn video_stream(&self) -> &VideoStreamInfo {
        &self.video
    }

    pub fn next_video_packet(&mut self) -> anyhow::Result<Option<EncodedVideoPacket>> {
        self.read_packet(false)
    }

    pub fn next_decode_packet(&mut self) -> anyhow::Result<Option<EncodedVideoPacket>> {
        self.read_packet(true)
    }

    fn read_packet(&mut self, decode_format: bool) -> anyhow::Result<Option<EncodedVideoPacket>> {
        let mut native = std::mem::MaybeUninit::<NativePacket>::zeroed();
        let mut error = NativeError::empty();
        let status = unsafe {
            if decode_format {
                np_video_demux_read_decode(self.handle.as_ptr(), native.as_mut_ptr(), &mut error)
            } else {
                np_video_demux_read(self.handle.as_ptr(), native.as_mut_ptr(), &mut error)
            }
        };
        if status < 0 {
            return Err(error.into_anyhow("read native video packet"));
        }
        if status == 0 {
            return Ok(None);
        }
        let native = unsafe { native.assume_init() };
        anyhow::ensure!(
            native.data_len == 0 || !native.data.is_null(),
            "native demuxer returned a null packet buffer"
        );
        let data = if native.data_len == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(native.data, native.data_len) }.to_vec()
        };
        Ok(Some(EncodedVideoPacket {
            data,
            pts: native.pts,
            dts: native.dts,
            duration: native.duration,
            stream_index: native.stream_index,
            is_keyframe: native.flags & AV_PKT_FLAG_KEY != 0,
        }))
    }
}

#[cfg(target_os = "linux")]
impl Drop for NativeDemuxer {
    fn drop(&mut self) {
        unsafe { np_video_demux_close(self.handle.as_ptr()) };
    }
}

#[cfg(target_os = "linux")]
pub struct NativeMuxer {
    handle: Option<NonNull<c_void>>,
}

#[cfg(target_os = "linux")]
impl NativeMuxer {
    pub fn create(path: impl AsRef<Path>, video: &VideoStreamInfo) -> anyhow::Result<Self> {
        let path = CString::new(path.as_ref().as_os_str().as_bytes())
            .map_err(|_| anyhow::anyhow!("media path contains a NUL byte"))?;
        let native = native_stream_info(video);
        let mut handle = std::ptr::null_mut();
        let mut error = NativeError::empty();
        let status = unsafe { np_video_mux_open(path.as_ptr(), &native, &mut handle, &mut error) };
        if status < 0 {
            return Err(error.into_anyhow("open native muxer"));
        }
        let handle = NonNull::new(handle)
            .ok_or_else(|| anyhow::anyhow!("native muxer returned a null handle"))?;
        Ok(Self {
            handle: Some(handle),
        })
    }

    pub fn write_video_packet(&mut self, packet: &EncodedVideoPacket) -> anyhow::Result<()> {
        let handle = self
            .handle
            .ok_or_else(|| anyhow::anyhow!("native muxer is already finished"))?;
        let native = NativePacket {
            data: packet.data.as_ptr(),
            data_len: packet.data.len(),
            pts: packet.pts,
            dts: packet.dts,
            duration: packet.duration,
            stream_index: packet.stream_index,
            flags: u32::from(packet.is_keyframe) * AV_PKT_FLAG_KEY,
        };
        let mut error = NativeError::empty();
        let status = unsafe { np_video_mux_write(handle.as_ptr(), &native, &mut error) };
        if status < 0 {
            return Err(error.into_anyhow("write native mux packet"));
        }
        Ok(())
    }

    pub fn finish(mut self) -> anyhow::Result<()> {
        let handle = self
            .handle
            .take()
            .ok_or_else(|| anyhow::anyhow!("native muxer is already finished"))?;
        let mut error = NativeError::empty();
        let status = unsafe { np_video_mux_finish(handle.as_ptr(), &mut error) };
        if status < 0 {
            return Err(error.into_anyhow("finish native muxer"));
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
impl Drop for NativeMuxer {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            unsafe { np_video_mux_close(handle.as_ptr()) };
        }
    }
}

#[cfg(target_os = "linux")]
fn native_stream_info(video: &VideoStreamInfo) -> NativeStreamInfo {
    NativeStreamInfo {
        abi_version: NATIVE_VIDEO_ABI_VERSION,
        index: video.index,
        codec: match video.codec {
            VideoCodec::H264 => 1,
            VideoCodec::Hevc => 2,
            VideoCodec::Av1 => 3,
            VideoCodec::Vp9 => 4,
        },
        width: video.width,
        height: video.height,
        bit_depth: video.bit_depth,
        time_base_num: video.time_base_num,
        time_base_den: video.time_base_den,
        frame_rate_num: video.frame_rate_num,
        frame_rate_den: video.frame_rate_den,
        frame_count: video
            .frame_count
            .and_then(|count| i64::try_from(count).ok())
            .unwrap_or(0),
        duration_ts: video.duration_ts.unwrap_or(0),
        color_range: native_color_range(video.color.range),
        color_matrix: native_color_matrix(video.color.matrix),
        color_primaries: native_color_primaries(video.color.primaries),
        color_transfer: native_color_transfer(video.color.transfer),
        chroma_location: native_chroma_location(video.color.chroma_location),
        extradata: video.extradata.as_ptr(),
        extradata_len: video.extradata.len(),
    }
}

#[cfg(target_os = "linux")]
const fn native_color_range(value: ColorRange) -> i32 {
    match value {
        ColorRange::Limited => 1,
        ColorRange::Full => 2,
        ColorRange::Unspecified => 0,
    }
}

#[cfg(target_os = "linux")]
const fn native_color_matrix(value: ColorMatrix) -> i32 {
    match value {
        ColorMatrix::Bt709 => 1,
        ColorMatrix::Bt601 => 6,
        ColorMatrix::Bt2020NonConstantLuminance => 9,
        ColorMatrix::Unspecified => 2,
    }
}

#[cfg(target_os = "linux")]
const fn native_color_primaries(value: ColorPrimaries) -> i32 {
    match value {
        ColorPrimaries::Bt709 => 1,
        ColorPrimaries::Bt2020 => 9,
        ColorPrimaries::Unspecified => 2,
    }
}

#[cfg(target_os = "linux")]
const fn native_color_transfer(value: TransferCharacteristic) -> i32 {
    match value {
        TransferCharacteristic::Bt709 => 1,
        TransferCharacteristic::Smpte2084 => 16,
        TransferCharacteristic::Hlg => 18,
        TransferCharacteristic::Unspecified => 2,
    }
}

#[cfg(target_os = "linux")]
const fn native_chroma_location(value: ChromaLocation) -> i32 {
    match value {
        ChromaLocation::Left => 1,
        ChromaLocation::Center => 2,
        ChromaLocation::TopLeft => 3,
        ChromaLocation::Unspecified => 0,
    }
}

#[cfg(target_os = "linux")]
fn map_color_info(native: &NativeStreamInfo) -> VideoColorInfo {
    VideoColorInfo {
        range: match native.color_range {
            1 => ColorRange::Limited,
            2 => ColorRange::Full,
            _ => ColorRange::Unspecified,
        },
        matrix: match native.color_matrix {
            1 => ColorMatrix::Bt709,
            5 | 6 => ColorMatrix::Bt601,
            9 => ColorMatrix::Bt2020NonConstantLuminance,
            _ => ColorMatrix::Unspecified,
        },
        primaries: match native.color_primaries {
            1 => ColorPrimaries::Bt709,
            9 => ColorPrimaries::Bt2020,
            _ => ColorPrimaries::Unspecified,
        },
        transfer: match native.color_transfer {
            1 => TransferCharacteristic::Bt709,
            16 => TransferCharacteristic::Smpte2084,
            18 => TransferCharacteristic::Hlg,
            _ => TransferCharacteristic::Unspecified,
        },
        chroma_location: match native.chroma_location {
            1 => ChromaLocation::Left,
            2 => ChromaLocation::Center,
            3 => ChromaLocation::TopLeft,
            _ => ChromaLocation::Unspecified,
        },
    }
}
