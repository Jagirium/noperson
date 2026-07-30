use noperson::live::{LiveFrameError, validate_rgb_frame};

#[test]
fn live_frame_contract_rejects_truncated_camera_data() {
    let error = validate_rgb_frame(&[0; 11], 2, 2).unwrap_err();
    assert_eq!(
        error,
        LiveFrameError::LengthMismatch {
            width: 2,
            height: 2,
            expected: 12,
            actual: 11,
        }
    );
}

#[test]
fn live_frame_contract_accepts_photo_and_camera_rgb_frames() {
    validate_rgb_frame(&[0; 12], 2, 2).unwrap();
}

#[test]
#[ignore = "requires CUDA + ONNX models"]
fn elon_self_swap_meets_photo_and_camera_quality_gates() -> anyhow::Result<()> {
    use std::sync::Arc;

    use cudarc::driver::CudaContext;
    use image::GenericImageView;
    use noperson::config::parameters::FaceSwapParams;
    use noperson::gpu::ops::GpuOps;
    use noperson::live::LiveEngine;
    use noperson::quality::compare_rgb;

    let context = Arc::new(CudaContext::new(0)?);
    let stream = context.default_stream().clone();
    let gpu = Arc::new(GpuOps::new(&context, stream.clone())?);
    let mut engine = LiveEngine::new(
        gpu,
        std::path::Path::new("models"),
        std::path::Path::new("face.jpg"),
        FaceSwapParams::default(),
        &stream,
    )?;
    let original = image::open("face.jpg")?;
    let (width, height) = original.dimensions();
    let original = original.to_rgb8();
    let output = engine.process_rgb(original.as_raw(), width, height)?;
    let next_live_frame = engine.process_rgb(original.as_raw(), width, height)?;
    let metrics = compare_rgb(original.as_raw(), &output.data, width, height)?;

    assert_eq!(output.faces_detected, 1);
    assert_eq!(output.faces_swapped, 1);
    assert_eq!(
        output.data, next_live_frame.data,
        "identical consecutive camera frames must not flicker"
    );
    assert!(metrics.mae < 0.75, "self-swap MAE regressed: {metrics:?}");
    assert!(metrics.psnr > 36.0, "self-swap PSNR regressed: {metrics:?}");
    assert!(
        metrics.seam_p99 <= 7.0,
        "hard paste boundary detected: {metrics:?}"
    );
    if std::env::var_os("NOPERSON_DUMP_TEST_IMAGES").is_some() {
        std::fs::create_dir_all("artifacts")?;
        image::save_buffer(
            "artifacts/elon-self-swap.png",
            &output.data,
            width,
            height,
            image::ColorType::Rgb8,
        )?;
    }
    Ok(())
}
