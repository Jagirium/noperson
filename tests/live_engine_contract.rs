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

    use image::GenericImageView;
    use noperson::backend::ComputeContext;
    use noperson::backend::{ComputeOps, cuda::npp};
    use noperson::config::{
        parameters::{ColorAdjustParams, FaceSwapParams},
        settings::ExecutionProvider,
    };
    use noperson::live::LiveEngine;
    use noperson::models::manager::ModelManager;
    use noperson::pipeline::{
        color::adjust_color_reference, face_mask::gaussian_blur, face_recognizer::FaceRecognizer,
    };
    use noperson::quality::compare_rgb;

    npp::initialize_runtime(std::path::Path::new("libs/base"))?;
    let context = Arc::new(ComputeContext::new(0)?);
    let stream = context.new_stream()?;
    let gpu = Arc::new(ComputeOps::new(&context, stream.clone())?);

    // Shared-memory Gaussian tiling stays numerically equivalent to the CPU oracle.
    let mut gaussian_expected: Vec<f32> = (0..49)
        .map(|index| ((index * 17 + 3) % 29) as f32 / 28.0)
        .collect();
    let gaussian_input = gaussian_expected.clone();
    gaussian_blur(&mut gaussian_expected, 7, 7, 5, 1.1);
    let mut weights: Vec<f32> = (-2..=2)
        .map(|offset| (-(offset * offset) as f32 / (2.0 * 1.1 * 1.1)).exp())
        .collect();
    let weight_sum: f32 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= weight_sum;
    }
    let mut gaussian_gpu = gpu.upload(&gaussian_input)?;
    let mut gaussian_tmp = gpu.alloc_zeros(49)?;
    let gaussian_kernel = gpu.upload(&weights)?;
    gpu.gaussian_blur_mask(
        &mut gaussian_gpu,
        &mut gaussian_tmp,
        7,
        7,
        &gaussian_kernel,
        5,
    )?;
    let gaussian_actual = gpu.download(&gaussian_gpu)?;
    for (actual, expected) in gaussian_actual.iter().zip(&gaussian_expected) {
        assert!(
            (actual - expected).abs() < 2e-5,
            "Gaussian mismatch: {actual} != {expected}"
        );
    }

    // Fused prep+contrast accumulation stays below one 8-bit level.
    let color_input: Vec<[f32; 3]> = (0..64)
        .map(|pixel| {
            [
                ((pixel * 37 + 11) % 256) as f32,
                ((pixel * 19 + 73) % 256) as f32,
                ((pixel * 53 + 29) % 256) as f32,
            ]
        })
        .collect();
    let color_params = ColorAdjustParams {
        enabled: true,
        red: 4.0,
        green: -3.0,
        blue: 2.0,
        brightness: 0.93,
        contrast: 1.17,
        saturation: 0.82,
        sharpness: 1.25,
        hue: 0.08,
        gamma: 1.03,
        noise: 0.0,
    };
    let color_expected = adjust_color_reference(&color_input, 8, 8, &color_params);
    let color_chw: Vec<f32> = (0..3)
        .flat_map(|channel| color_input.iter().map(move |rgb| rgb[channel]))
        .collect();
    let mut color_gpu = gpu.upload(&color_chw)?;
    let mut color_scratch = gpu.alloc_zeros(color_chw.len())?;
    let mut gray_sum = stream.alloc_zeros::<u32>(1)?;
    let mut gray_partials = stream.alloc_zeros::<u32>(1024)?;
    gpu.adjust_color(
        &mut color_gpu,
        &mut color_scratch,
        &mut gray_sum,
        &mut gray_partials,
        8,
        8,
        color_params.gamma,
        [color_params.red, color_params.green, color_params.blue],
        color_params.brightness,
        color_params.contrast,
        color_params.saturation,
        color_params.sharpness,
        color_params.hue,
        color_params.noise,
        0,
    )?;
    let color_actual = gpu.download(&color_gpu)?;
    for (pixel, expected) in color_expected.iter().enumerate() {
        for channel in 0..3 {
            let actual = color_actual[channel * 64 + pixel];
            assert!((actual - expected[channel]).abs() < 1.0, "color mismatch");
        }
    }

    // The no-blur differencing fast path has the same bimodal per-pixel mask.
    let original = [0.0, 9.0, 20.0, 30.0, 40.0, 49.0];
    let swapped = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let mut swapped_gpu = gpu.upload(&swapped)?;
    let original_gpu = gpu.upload(&original)?;
    gpu.fake_diff_composite_direct(&mut swapped_gpu, &original_gpu, 2, 4)?;
    assert_eq!(
        gpu.download(&swapped_gpu)?,
        [0.0, 20.0, 20.0, 40.0, 40.0, 60.0]
    );

    // Ordered warp-ballot compaction keeps anchors around chunk boundaries sorted.
    let anchors = 8_400usize;
    let selected = [1usize, 255, 256, 8_399];
    let mut detector_output = vec![0.0f32; 20 * anchors];
    for (ordinal, anchor) in selected.into_iter().enumerate() {
        detector_output[anchor] = 100.0 + ordinal as f32;
        detector_output[anchors + anchor] = 200.0 + ordinal as f32;
        detector_output[2 * anchors + anchor] = 20.0;
        detector_output[3 * anchors + anchor] = 30.0;
        detector_output[4 * anchors + anchor] = 0.9 - ordinal as f32 * 0.05;
    }
    let detector_gpu = gpu.upload(&detector_output)?;
    let mut candidates = gpu.alloc_zeros(anchors * 15)?;
    let mut candidate_count = stream.alloc_zeros::<u32>(1)?;
    gpu.compact_yolo_faces(
        &detector_gpu,
        &mut candidates,
        &mut candidate_count,
        0.5,
        1.0,
    )?;
    let mut count = [0u32; 1];
    stream.memcpy_dtoh(&candidate_count, &mut count)?;
    assert_eq!(count[0], selected.len() as u32);
    let candidates = gpu.download(&candidates)?;
    for (ordinal, expected_score) in [0.9, 0.85, 0.8, 0.75].into_iter().enumerate() {
        assert!((candidates[ordinal * 15 + 14] - expected_score).abs() < 1e-6);
    }

    // Fused latent projection keeps the CPU reduction contract.
    let embedding: Vec<f32> = (0..512).map(|index| (index as f32 * 0.017).sin()).collect();
    let mut emap = vec![0.0f32; 512 * 512];
    for index in 0..512 {
        emap[index * 512 + index] = 1.0;
    }
    let latent_expected = FaceRecognizer::calc_latent(&embedding, &emap);
    let mut embedding_gpu = gpu.upload(&embedding)?;
    let emap_gpu = gpu.upload(&emap)?;
    let mut latent_gpu = gpu.alloc_zeros(512)?;
    FaceRecognizer::calc_latent_gpu(&gpu, &mut embedding_gpu, &emap_gpu, &mut latent_gpu)?;
    for (actual, expected) in gpu.download(&latent_gpu)?.iter().zip(&latent_expected) {
        assert!((actual - expected).abs() < 1e-5, "latent mismatch");
    }

    let mut engine = LiveEngine::new(
        gpu.clone(),
        std::path::Path::new("models"),
        std::path::Path::new("assets/photos/face.jpg"),
        FaceSwapParams::default(),
        &stream,
    )?;
    let original = image::open("assets/photos/face.jpg")?;
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

    // TensorRT must at least build a real production detector session from the
    // same runtime generation before the one-shot GPU gate succeeds.
    drop(engine);
    let mut tensorrt = ModelManager::with_provider("models", ExecutionProvider::TensorRt);
    tensorrt.set_compute_stream(stream.cu_stream() as *mut ())?;
    tensorrt.load("RuntimeCheck", "yoloface_8n.onnx")?;
    Ok(())
}
