use noperson::io::native_video::{
    ChromaLocation, ColorMatrix, ColorPrimaries, ColorRange, NativeVideoState, NvCodecCapabilities,
    PixelFormat, TransferCharacteristic, VideoCodec, VideoColorInfo, VideoDescriptor,
};

#[cfg(target_os = "linux")]
use noperson::io::native_video::{
    MappedVideoSurface, NativeDemuxer, NativeMuxer, NvDecoder, NvEncoder, NvEncoderConfig,
    NvEncoderInputSurface, remux_audio,
};

#[test]
fn descriptor_accepts_nv12_and_p010_with_explicit_color_metadata() {
    let nv12 = VideoDescriptor::new(
        1920,
        1080,
        PixelFormat::Nv12,
        VideoColorInfo {
            range: ColorRange::Limited,
            matrix: ColorMatrix::Bt709,
            primaries: ColorPrimaries::Bt709,
            transfer: TransferCharacteristic::Bt709,
            chroma_location: ChromaLocation::Left,
        },
    )
    .unwrap();
    assert_eq!(nv12.bytes_per_sample(), 1);
    assert_eq!(nv12.luma_samples(), 1920 * 1080);

    let p010 = VideoDescriptor::new(
        3840,
        2160,
        PixelFormat::P010,
        VideoColorInfo {
            range: ColorRange::Limited,
            matrix: ColorMatrix::Bt2020NonConstantLuminance,
            primaries: ColorPrimaries::Bt2020,
            transfer: TransferCharacteristic::Smpte2084,
            chroma_location: ChromaLocation::Left,
        },
    )
    .unwrap();
    assert_eq!(p010.bytes_per_sample(), 2);
    assert_eq!(p010.luma_samples(), 3840 * 2160);
}

#[test]
fn descriptor_rejects_geometry_that_cannot_represent_420_chroma() {
    let error =
        VideoDescriptor::new(1919, 1080, PixelFormat::Nv12, VideoColorInfo::default()).unwrap_err();
    assert_eq!(
        error.to_string(),
        "4:2:0 video dimensions must be even: 1919x1080"
    );
}

#[test]
fn native_video_state_machine_releases_each_mapped_surface_before_finish() {
    let mut state = NativeVideoState::new();
    state.open().unwrap();
    state.begin_draining().unwrap();

    let first = state.map_surface().unwrap();
    let second = state.map_surface().unwrap();
    assert_ne!(first, second);
    assert!(state.finish().is_err());

    state.unmap_surface(first).unwrap();
    state.unmap_surface(second).unwrap();
    state.finish().unwrap();
    assert!(state.is_finished());
}

#[test]
fn native_video_state_machine_rejects_duplicate_surface_release() {
    let mut state = NativeVideoState::new();
    state.open().unwrap();
    let token = state.map_surface().unwrap();
    state.unmap_surface(token).unwrap();

    let error = state.unmap_surface(token).unwrap_err();
    assert_eq!(error.to_string(), "video surface token is not mapped");
}

#[test]
fn native_video_pipeline_uses_deferred_surfaces_and_a_bounded_encoder_ring() {
    let runtime = std::fs::read_to_string("src/extra_gui/runtime.rs").unwrap();
    assert!(runtime.contains("DeferredVideoSurface"));
    assert!(runtime.contains("record_event(None)"));
    assert!(runtime.contains("let encode_surfaces = (0..5)"));
    assert!(
        !runtime.contains("gpu_ops.stream.synchronize()?"),
        "production recording must not synchronize the whole CUDA stream per frame"
    );

    let encoder = std::fs::read_to_string("native/video/nvcodec_encoder.cpp").unwrap();
    assert!(encoder.contains("constexpr size_t bitstream_ring_size = 4"));
    assert!(encoder.contains("free_bitstreams"));
    assert!(encoder.contains("np_video_nvencoder_receive"));
}

#[test]
fn nvenc_input_surfaces_are_pitched_and_outlive_resource_registration() {
    let video = std::fs::read_to_string("src/io/native_video.rs").unwrap();
    assert!(video.contains("cuMemAllocPitch_v2"));
    assert!(video.contains("pub struct NvEncoderInputSurface"));

    let runtime = std::fs::read_to_string("src/extra_gui/runtime.rs").unwrap();
    assert!(runtime.contains("NvEncoderInputSurface::new"));
    assert!(runtime.contains("process_chw_to_pitched_nv12"));
    assert!(
        runtime.find("let encode_surfaces").unwrap() < runtime.find("let mut encoder").unwrap(),
        "encoder must drop and unregister resources before its CUDA surfaces are freed"
    );
    assert!(!runtime.contains("alloc_zeros_u8(output_bytes)"));
}

#[test]
fn nvenc_errors_use_stable_status_names() {
    let encoder = std::fs::read_to_string("native/video/nvcodec_encoder.cpp").unwrap();
    assert!(encoder.contains("NV_ENC_ERR_RESOURCE_REGISTER_FAILED"));
    assert!(encoder.contains("nvenc_status_name(status)"));
    assert!(!encoder.contains("nvEncGetLastErrorString"));
}

#[test]
fn nvenc_receives_a_pointer_to_the_cuda_stream_handle() {
    let encoder = std::fs::read_to_string("native/video/nvcodec_encoder.cpp").unwrap();
    assert!(encoder.contains("&encoder->stream, &encoder->stream"));
    assert!(!encoder.contains("nvEncSetIOCudaStreams(encoder->session, cuda_stream, cuda_stream)"));
}

#[test]
fn nv12_cuda_color_coefficients_use_the_kernel_float_abi() {
    let ops = std::fs::read_to_string("src/gpu/ops.rs").unwrap();
    assert!(ops.contains("let (rv, gu, gv, bu): (f32, f32, f32, f32)"));
    assert!(ops.contains("let coefficients: [f32; 11]"));
}

#[test]
fn nvdec_releases_a_reused_picture_slot_before_mapping_it_again() {
    let shim = std::fs::read_to_string("native/video/nvcodec_shim.cpp").unwrap();
    assert!(shim.contains("np_video_nvdecoder_peek"));

    let runtime = std::fs::read_to_string("src/extra_gui/runtime.rs").unwrap();
    assert!(runtime.contains("decoder.next_picture_index()?"));
    assert!(runtime.contains("surface.picture_index() == next_picture_index"));
    assert!(runtime.contains(".remove(position)"));
}

#[cfg(target_os = "linux")]
#[test]
fn native_demuxer_reads_h264_packets_and_timestamps_in_process() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("two-frames.mp4");
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=2:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            "-y",
        ])
        .arg(&input)
        .status()
        .unwrap();
    assert!(status.success());

    let mut demuxer = NativeDemuxer::open(&input).unwrap();
    let stream = demuxer.video_stream().clone();
    assert_eq!(stream.codec, VideoCodec::H264);
    assert_eq!((stream.width, stream.height), (64, 48));
    assert_eq!((stream.time_base_num, stream.time_base_den), (1, 16384));
    assert!(!stream.extradata.is_empty());

    let first = demuxer.next_video_packet().unwrap().unwrap();
    assert!(first.is_keyframe);
    assert!(!first.data.is_empty());
    assert_eq!(first.stream_index, stream.index);
    assert!(first.duration > 0);

    let second = demuxer.next_video_packet().unwrap().unwrap();
    assert!(!second.data.is_empty());
    assert!(demuxer.next_video_packet().unwrap().is_none());
}

#[cfg(target_os = "linux")]
#[test]
fn nvcodec_probe_loads_driver_apis_without_cuda_toolkit() {
    let capabilities = NvCodecCapabilities::probe().unwrap();
    if !capabilities.cuda_driver && !capabilities.nvdec && !capabilities.nvenc {
        return;
    }
    assert!(capabilities.cuda_driver);
    assert!(capabilities.nvdec);
    assert!(capabilities.nvenc);
    assert!(capabilities.nvenc_api_major >= 13);
}

#[cfg(target_os = "linux")]
#[test]
fn native_muxer_preserves_encoded_packets_and_timestamps_in_process() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input.mp4");
    let output = directory.path().join("output.mp4");
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=2:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            "-y",
        ])
        .arg(&input)
        .status()
        .unwrap();
    assert!(status.success());
    let mut source = NativeDemuxer::open(&input).unwrap();
    let stream = source.video_stream().clone();
    let mut sink = NativeMuxer::create(&output, &stream).unwrap();
    let mut written = 0;
    while let Some(packet) = source.next_video_packet().unwrap() {
        sink.write_video_packet(&packet).unwrap();
        written += 1;
    }
    sink.finish().unwrap();
    assert_eq!(written, 2);

    let mut result = NativeDemuxer::open(&output).unwrap();
    assert_eq!(result.video_stream().codec, VideoCodec::H264);
    assert_eq!(
        (result.video_stream().width, result.video_stream().height),
        (64, 48)
    );
    let first = result.next_video_packet().unwrap().unwrap();
    let second = result.next_video_packet().unwrap().unwrap();
    assert!(first.is_keyframe);
    assert!(first.duration > 0);
    assert!(second.duration > 0);
    assert!(result.next_video_packet().unwrap().is_none());
}

#[cfg(target_os = "linux")]
#[test]
fn native_muxer_builds_mp4_extradata_from_annex_b_key_packet() {
    let directory = tempfile::tempdir().unwrap();
    let elementary = directory.path().join("input.h264");
    let output = directory.path().join("output.mp4");
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=2:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "h264",
            "-y",
        ])
        .arg(&elementary)
        .status()
        .unwrap();
    assert!(status.success());

    let mut source = NativeDemuxer::open(&elementary).unwrap();
    let stream = source.video_stream().clone();
    assert!(!stream.extradata.is_empty());
    let mut sink = NativeMuxer::create(&output, &stream).unwrap();
    let mut index = 0_i64;
    while let Some(mut packet) = source.next_video_packet().unwrap() {
        packet.pts = index * 8192;
        packet.dts = packet.pts;
        packet.duration = 8192;
        sink.write_video_packet(&packet).unwrap();
        index += 1;
    }
    sink.finish().unwrap();

    let mut result = NativeDemuxer::open(&output).unwrap();
    assert_eq!(result.video_stream().codec, VideoCodec::H264);
    assert!(!result.video_stream().extradata.is_empty());
    assert!(result.next_video_packet().unwrap().is_some());
}

#[cfg(target_os = "linux")]
#[test]
fn native_remux_replaces_video_and_preserves_source_audio() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.mp4");
    let video = directory.path().join("video.mp4");
    let output = directory.path().join("output.mp4");
    let source_status = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=2:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=1",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            "-y",
        ])
        .arg(&source)
        .status()
        .unwrap();
    assert!(source_status.success());
    let video_status = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=red:size=80x60:rate=2:duration=1",
            "-c:v",
            "libx264",
            "-an",
            "-y",
        ])
        .arg(&video)
        .status()
        .unwrap();
    assert!(video_status.success());

    remux_audio(&video, &source, &output).unwrap();
    let probe = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height",
            "-of",
            "csv=p=0",
        ])
        .arg(&output)
        .output()
        .unwrap();
    assert!(probe.status.success());
    let streams = String::from_utf8(probe.stdout).unwrap();
    assert!(streams.lines().any(|line| line == "video,80,60"));
    assert!(streams.lines().any(|line| line.starts_with("audio")));
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "single final GPU verification"]
fn nvcodec_decodes_and_encodes_device_resident_frames() -> anyhow::Result<()> {
    use std::sync::Arc;

    use anyhow::Context;
    use cudarc::driver::CudaContext;
    use noperson::gpu::npp;
    use noperson::gpu::ops::GpuOps;

    let directory = tempfile::tempdir()?;
    let input = directory.path().join("input.mp4");
    let output = directory.path().join("output.mp4");
    let reference_png = directory.path().join("reference.png");
    let reference_raw = directory.path().join("reference.rgb");
    let decoded_png = directory.path().join("decoded.png");
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=256x144:rate=12:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_range",
            "tv",
            "-y",
        ])
        .arg(&input)
        .status()?;
    assert!(status.success());
    let status = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&input)
        .args([
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-y",
        ])
        .arg(&reference_raw)
        .status()?;
    assert!(status.success());
    let status = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&input)
        .args(["-frames:v", "1", "-y"])
        .arg(&reference_png)
        .status()?;
    assert!(status.success());

    npp::initialize_runtime(std::path::Path::new("libs/base"))
        .context("initialize the local NPP runtime")?;
    let context = Arc::new(CudaContext::new(0).context("create CUDA context")?);
    let stream = context.new_stream()?;
    let gpu = GpuOps::new(&context, stream.clone()).context("initialize GPU kernels")?;
    let mut source = NativeDemuxer::open(&input)?;
    let source_info = source.video_stream().clone();
    let mut decoder = unsafe {
        NvDecoder::open(
            context.cu_ctx() as *mut std::ffi::c_void,
            stream.cu_stream() as *mut std::ffi::c_void,
            source_info.codec,
        )?
    };
    let config = NvEncoderConfig::h264_quality(
        256,
        144,
        source_info.frame_rate_num,
        source_info.frame_rate_den,
        source_info.time_base_num,
        source_info.time_base_den,
    )
    .with_color(source_info.color);
    let mut encoder = unsafe {
        NvEncoder::open(
            context.cu_ctx() as *mut std::ffi::c_void,
            stream.cu_stream() as *mut std::ffi::c_void,
            config,
        )?
    };
    let mut muxer = NativeMuxer::create(&output, encoder.video_stream())?;
    let mut chw = gpu.alloc_zeros(3 * 256 * 144)?;
    let mut decoded_rgb = gpu.alloc_zeros_u8(3 * 256 * 144)?;
    let mut decoded_frames = Vec::with_capacity(24 * 3 * 256 * 144);
    let nv12 = (0..5)
        .map(|_| NvEncoderInputSurface::new(stream.clone(), 256, 144, PixelFormat::Nv12))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let frame_duration = i64::from(source_info.time_base_den)
        * i64::from(source_info.frame_rate_den)
        / (i64::from(source_info.time_base_num) * i64::from(source_info.frame_rate_num));
    let mut encoded = 0_i64;
    let mut encode_decoded = |decoded: MappedVideoSurface| -> anyhow::Result<()> {
        assert_ne!(decoded.device_ptr(), 0);
        assert_eq!((decoded.width(), decoded.height()), (256, 144));
        assert_eq!(decoded.pixel_format(), PixelFormat::Nv12);
        unsafe {
            gpu.nv12_device_to_chw_f32(
                decoded.device_ptr(),
                decoded.pitch(),
                &mut chw,
                decoded.height(),
                decoded.width(),
                decoded.pixel_format(),
                source_info.color.matrix,
                source_info.color.range,
            )?;
        }
        stream
            .synchronize()
            .context("finish NVDEC surface conversion")?;
        gpu.chw_f32_to_hwc_u8(&chw, &mut decoded_rgb, 144, 256)?;
        let host = stream.clone_dtoh(&decoded_rgb)?;
        if encoded == 0 {
            image::save_buffer(&decoded_png, &host, 256, 144, image::ColorType::Rgb8)?;
        }
        decoded_frames.extend_from_slice(&host);
        drop(decoded);
        let encode_surface = &nv12[encoded as usize % nv12.len()];
        unsafe {
            gpu.chw_f32_to_pitched_nv12_scaled_color(
                &chw,
                encode_surface.device_ptr(),
                encode_surface.pitch(),
                144,
                256,
                144,
                256,
                source_info.color.matrix,
                source_info.color.range,
                PixelFormat::Nv12,
            )?;
        }
        let pts = encoded * frame_duration;
        if let Some(packet) = unsafe {
            encoder.encode_device_frame(
                encode_surface.device_ptr(),
                encode_surface.pitch(),
                pts,
                frame_duration,
            )?
        } {
            muxer.write_video_packet(&packet)?;
        }
        encoded += 1;
        Ok(())
    };
    while let Some(packet) = source.next_decode_packet()? {
        decoder.send_packet(
            &packet,
            source_info.time_base_num,
            source_info.time_base_den,
        )?;
        while let Some(decoded) = decoder.next_frame()? {
            encode_decoded(decoded)?;
        }
    }
    decoder.flush().context("flush NVDEC parser")?;
    while let Some(decoded) = decoder.next_frame()? {
        encode_decoded(decoded)?;
    }
    assert_eq!(encoded, 24);
    let pending = encoder.finish()?;
    assert_eq!(pending.len(), 4);
    for packet in pending {
        muxer.write_video_packet(&packet)?;
    }
    muxer.finish()?;

    let decoded_raw = directory.path().join("decoded.rgb");
    std::fs::write(&decoded_raw, &decoded_frames)?;
    let reference_frames = std::fs::read(&reference_raw)?;
    assert_eq!(decoded_frames.len(), reference_frames.len());
    let frame_bytes = 3 * 256 * 144;
    let max_frame_mse = decoded_frames
        .chunks_exact(frame_bytes)
        .zip(reference_frames.chunks_exact(frame_bytes))
        .map(|(decoded, reference)| {
            decoded
                .iter()
                .zip(reference)
                .map(|(&actual, &expected)| {
                    let difference = i64::from(actual) - i64::from(expected);
                    (difference * difference) as f64
                })
                .sum::<f64>()
                / frame_bytes as f64
        })
        .fold(0.0_f64, f64::max);
    if max_frame_mse >= 4.0 {
        std::fs::create_dir_all("test")?;
        std::fs::copy(&input, "test/nvdec-order-source.mp4")?;
        std::fs::copy(&decoded_raw, "test/nvdec-order-output.rgb")?;
    }
    assert!(
        max_frame_mse < 4.0,
        "NVDEC sequence maximum frame MSE is {max_frame_mse:.2}"
    );

    let decode_comparison = std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-i"])
        .arg(&reference_png)
        .args(["-i"])
        .arg(&decoded_png)
        .args(["-lavfi", "[0:v][1:v]psnr", "-f", "null", "-"])
        .output()?;
    assert!(decode_comparison.status.success());
    let decode_report = String::from_utf8_lossy(&decode_comparison.stderr);
    let decode_average = decode_report
        .split("average:")
        .nth(1)
        .and_then(|tail| tail.split_whitespace().next())
        .and_then(|value| value.parse::<f32>().ok())
        .context("ffmpeg did not report NVDEC-to-RGB PSNR")?;
    if decode_average < 30.0 {
        std::fs::create_dir_all("test")?;
        std::fs::copy(&reference_png, "test/nvdec-reference.png")?;
        std::fs::copy(&decoded_png, "test/nvdec-decoded.png")?;
    }
    assert!(
        decode_average >= 30.0,
        "NVDEC-to-RGB PSNR is {decode_average:.2} dB"
    );

    let mut result = NativeDemuxer::open(&output)?;
    assert_eq!(result.video_stream().codec, VideoCodec::H264);
    assert_eq!(result.video_stream().color.matrix, ColorMatrix::Bt709);
    assert_eq!(result.video_stream().color.range, ColorRange::Limited);
    assert!(result.next_video_packet()?.is_some());
    assert!(result.next_video_packet()?.is_some());

    let comparison = std::process::Command::new("ffmpeg")
        .args(["-hide_banner", "-i"])
        .arg(&input)
        .args(["-i"])
        .arg(&output)
        .args(["-lavfi", "[0:v][1:v]psnr", "-f", "null", "-"])
        .output()?;
    assert!(comparison.status.success());
    let report = String::from_utf8_lossy(&comparison.stderr);
    let average = report
        .split("average:")
        .nth(1)
        .and_then(|tail| tail.split_whitespace().next())
        .and_then(|value| value.parse::<f32>().ok())
        .context("ffmpeg did not report round-trip PSNR")?;
    if average < 30.0 {
        std::fs::create_dir_all("test")?;
        std::fs::copy(&input, "test/nvcodec-order-source.mp4")?;
        std::fs::copy(&output, "test/nvcodec-order-output.mp4")?;
    }
    assert!(average >= 30.0, "NV12 round-trip PSNR is {average:.2} dB");
    Ok(())
}
