use nokhwa::utils::FrameFormat;
use noperson::io::nvjpeg::{
    DecodeRoute, decode_route, runtime_library_names, validate_destination,
};
use noperson::pipeline::workspace::FrameRingLayout;

#[test]
fn mjpeg_uses_nvjpeg_only_when_the_runtime_is_available() {
    assert_eq!(decode_route(FrameFormat::MJPEG, true), DecodeRoute::NvJpeg);
    assert_eq!(decode_route(FrameFormat::MJPEG, false), DecodeRoute::Cpu);
    assert_eq!(decode_route(FrameFormat::YUYV, true), DecodeRoute::Cpu);
}

#[test]
fn loader_targets_the_cuda_12_nvjpeg_soname() {
    #[cfg(target_os = "linux")]
    assert_eq!(
        runtime_library_names(),
        &["libnvjpeg.so.12", "libnvjpeg.so"]
    );

    #[cfg(target_os = "windows")]
    assert_eq!(runtime_library_names(), &["nvjpeg64_12.dll"]);
}

#[test]
fn interleaved_rgb_destination_must_hold_the_complete_frame() {
    assert!(validate_destination(2560, 1440, 2560 * 1440 * 3).is_ok());
    let error = validate_destination(2560, 1440, 2560 * 1440 * 3 - 1)
        .expect_err("undersized CUDA output must be rejected before FFI");
    assert!(error.to_string().contains("requires 11059200 bytes"));
}

#[test]
fn frame_ring_is_sized_for_the_selected_2k_capture() {
    let layout = FrameRingLayout::new(2560, 1440).expect("2K camera geometry is valid");
    assert_eq!(layout.max_pixels(), 2560 * 1440);
    assert_eq!(layout.rgb_elements(), 2560 * 1440 * 3);
}

#[test]
fn live_camera_capture_and_ui_telemetry_never_serialize_inference() {
    let app = std::fs::read_to_string("src/app.rs").expect("realtime app source exists");
    assert!(app.contains("capture_worker.recv_latest"));
    assert!(app.contains("tx.try_send(WorkerMsg::FaceCount"));
    assert!(app.contains("tx.try_send(WorkerMsg::Fps"));
}
