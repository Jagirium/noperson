use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use image::GenericImageView;

use noperson::config::parameters::FaceSwapParams;
use noperson::config::settings::ExecutionProvider;
use noperson::gpu::ops::GpuOps;
use noperson::live::LiveEngine;
use noperson::math::{affine, constants::ARCFACE_DST};
use noperson::models::live_catalog::CANONICAL_SWAPPER_FILENAME;
use noperson::models::manager::ModelManager;
use noperson::pipeline::face_detector::YoloFaceDetector;
use noperson::pipeline::face_recognizer::FaceRecognizer;
use noperson::pipeline::face_swapper::FaceSwapper;
use noperson::pipeline::workspace::GpuWorkspace;

fn init_gpu() -> anyhow::Result<(Arc<CudaStream>, Arc<GpuOps>)> {
    let context = Arc::new(CudaContext::new(0)?);
    let stream = context.default_stream().clone();
    let gpu = Arc::new(GpuOps::new(&context, stream.clone())?);
    Ok((stream, gpu))
}

fn load_cuda_reference_models(stream: &Arc<CudaStream>) -> anyhow::Result<ModelManager> {
    let mut manager = ModelManager::with_provider("models", ExecutionProvider::Cuda);
    manager.set_compute_stream(stream.cu_stream() as *mut ());
    manager.load("YoloFace8n", "yoloface_8n.onnx")?;
    manager.load("Inswapper128ArcFace", "w600k_r50.onnx")?;
    manager.load("Inswapper128", CANONICAL_SWAPPER_FILENAME)?;
    manager.load_emap(CANONICAL_SWAPPER_FILENAME)?;
    Ok(manager)
}

fn prepare_identical_swap_inputs(
    gpu: &GpuOps,
    stream: &Arc<CudaStream>,
    manager: &mut ModelManager,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let image = image::open("assets/photos/face.jpg")?;
    let (width, height) = image.dimensions();
    let rgb = image.to_rgb8();
    let input = gpu.upload_u8(rgb.as_raw())?;
    let mut frame = gpu.alloc_zeros(3 * width as usize * height as usize)?;
    gpu.hwc_u8_to_chw_f32(&input, &mut frame, height, width)?;

    let detector = YoloFaceDetector::new(0.5);
    let mut workspace = GpuWorkspace::new(stream)?;
    let (faces, _) = detector.detect_gpu(manager, gpu, &frame, &mut workspace, height, width)?;
    let face = faces
        .first()
        .ok_or_else(|| anyhow::anyhow!("No face detected in face.jpg"))?;
    let embedding = FaceRecognizer::recognize_gpu(
        manager,
        gpu,
        &frame,
        height,
        width,
        &face.kps_5,
        &mut workspace,
    )?;
    let latent = FaceRecognizer::calc_latent(
        &embedding,
        manager
            .emap
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("emap not loaded"))?,
    );

    let mut template = ARCFACE_DST;
    for point in &mut template {
        point[0] += 8.0;
    }
    let transform = affine::estimate_face_affine(&face.kps_5, &template);
    gpu.warp_affine_npp(
        &frame,
        &mut workspace.face_256,
        height,
        width,
        128,
        128,
        &transform,
    )?;
    let crop = gpu.download(&workspace.face_256)?[..3 * 128 * 128].to_vec();
    Ok((crop, latent))
}

fn identity_embedding(
    gpu: &GpuOps,
    stream: &Arc<CudaStream>,
    manager: &mut ModelManager,
    rgb: &[u8],
    width: u32,
    height: u32,
) -> anyhow::Result<Vec<f32>> {
    let input = gpu.upload_u8(rgb)?;
    let mut frame = gpu.alloc_zeros(3 * width as usize * height as usize)?;
    gpu.hwc_u8_to_chw_f32(&input, &mut frame, height, width)?;
    let detector = YoloFaceDetector::new(0.5);
    let mut workspace = GpuWorkspace::new(stream)?;
    let (faces, _) = detector.detect_gpu(manager, gpu, &frame, &mut workspace, height, width)?;
    let face = faces
        .first()
        .ok_or_else(|| anyhow::anyhow!("No face detected for identity cosine"))?;
    FaceRecognizer::recognize_gpu(
        manager,
        gpu,
        &frame,
        height,
        width,
        &face.kps_5,
        &mut workspace,
    )
    .map(Vec::from)
}

fn tensor_similarity(reference: &[f32], candidate: &[f32]) -> (f32, f32, f32) {
    assert_eq!(reference.len(), candidate.len());
    let mut absolute_error = 0.0f32;
    let mut max_error = 0.0f32;
    let mut dot = 0.0f32;
    let mut reference_norm = 0.0f32;
    let mut candidate_norm = 0.0f32;
    for (&left, &right) in reference.iter().zip(candidate) {
        let error = (left - right).abs();
        absolute_error += error;
        max_error = max_error.max(error);
        dot += left * right;
        reference_norm += left * left;
        candidate_norm += right * right;
    }
    let mae = absolute_error / reference.len() as f32;
    let cosine = dot / (reference_norm.sqrt() * candidate_norm.sqrt());
    (mae, max_error, cosine)
}

fn save_rgb(path: &str, data: &[u8], width: u32, height: u32) -> anyhow::Result<()> {
    image::save_buffer(path, data, width, height, image::ColorType::Rgb8)?;
    Ok(())
}

fn chw_f32_to_rgb_u8(chw: &[f32], width: usize, height: usize) -> Vec<u8> {
    let plane = width * height;
    let mut rgb = vec![0u8; plane * 3];
    for y in 0..height {
        for x in 0..width {
            let pixel = y * width + x;
            for channel in 0..3 {
                rgb[pixel * 3 + channel] =
                    chw[channel * plane + pixel].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    rgb
}

#[test]
#[ignore = "requires CUDA, ONNX models, and face.jpg"]
fn dump_rust_alignment_for_python_stage_comparison() -> anyhow::Result<()> {
    let (stream, gpu) = init_gpu()?;
    let image = image::open("assets/photos/face.jpg")?;
    let (width, height) = image.dimensions();
    let rgb = image.to_rgb8();
    let input = gpu.upload_u8(rgb.as_raw())?;
    let mut frame = gpu.alloc_zeros(3 * width as usize * height as usize)?;
    gpu.hwc_u8_to_chw_f32(&input, &mut frame, height, width)?;

    let mut manager = ModelManager::with_provider("models", ExecutionProvider::Cuda);
    manager.set_compute_stream(stream.cu_stream() as *mut ());
    manager.load("YoloFace8n", "yoloface_8n.onnx")?;

    let detector = YoloFaceDetector::new(0.5);
    let mut workspace = GpuWorkspace::new(&stream)?;
    let (faces, _) =
        detector.detect_gpu(&mut manager, &gpu, &frame, &mut workspace, height, width)?;
    let face = faces
        .first()
        .ok_or_else(|| anyhow::anyhow!("CUDA detector found no face"))?;

    let factor = 4.0f32;
    let mut template = ARCFACE_DST;
    for point in &mut template {
        point[0] *= factor;
        point[1] *= factor;
    }
    template[0][0] += factor * 8.0;
    template[0][1] += factor * 8.0;
    let transform = affine::estimate_face_affine(&face.kps_5, &template);
    gpu.warp_affine_npp(
        &frame,
        &mut workspace.face_512,
        height,
        width,
        512,
        512,
        &transform,
    )?;
    let aligned = gpu.download(&workspace.face_512)?;

    eprintln!("Rust landmarks={:?}", face.kps_5);
    eprintln!("Rust src→aligned transform={transform:?}");

    std::fs::create_dir_all("artifacts")?;
    save_rgb(
        "artifacts/rust-aligned-512.png",
        &chw_f32_to_rgb_u8(&aligned[..3 * 512 * 512], 512, 512),
        512,
        512,
    )?;
    Ok(())
}

#[test]
#[ignore = "requires CUDA, TensorRT, ONNX models, and face.jpg"]
fn provider_stage_cosines_locate_first_identity_divergence() -> anyhow::Result<()> {
    let (stream, gpu) = init_gpu()?;
    let image = image::open("assets/photos/face.jpg")?;
    let (width, height) = image.dimensions();
    let rgb = image.to_rgb8();
    let input = gpu.upload_u8(rgb.as_raw())?;
    let mut frame = gpu.alloc_zeros(3 * width as usize * height as usize)?;
    gpu.hwc_u8_to_chw_f32(&input, &mut frame, height, width)?;

    let mut cuda_manager = ModelManager::with_provider("models", ExecutionProvider::Cuda);
    cuda_manager.set_compute_stream(stream.cu_stream() as *mut ());
    cuda_manager.load("YoloFace8n", "yoloface_8n.onnx")?;
    cuda_manager.load("Inswapper128ArcFace", "w600k_r50.onnx")?;
    cuda_manager.load_emap(CANONICAL_SWAPPER_FILENAME)?;

    let mut tensorrt_manager = ModelManager::with_provider("models", ExecutionProvider::TensorRT);
    tensorrt_manager.set_compute_stream(stream.cu_stream() as *mut ());
    tensorrt_manager.load("YoloFace8n", "yoloface_8n.onnx")?;
    tensorrt_manager.load("Inswapper128ArcFace", "w600k_r50.onnx")?;
    tensorrt_manager.load_emap(CANONICAL_SWAPPER_FILENAME)?;

    let detector = YoloFaceDetector::new(0.5);
    let mut cuda_workspace = GpuWorkspace::new(&stream)?;
    let mut tensorrt_workspace = GpuWorkspace::new(&stream)?;
    let (cuda_faces, _) = detector.detect_gpu(
        &mut cuda_manager,
        &gpu,
        &frame,
        &mut cuda_workspace,
        height,
        width,
    )?;
    let cuda_detector_activation = cuda_workspace.host_detect_candidates.clone();
    let (tensorrt_faces, _) = detector.detect_gpu(
        &mut tensorrt_manager,
        &gpu,
        &frame,
        &mut tensorrt_workspace,
        height,
        width,
    )?;
    let tensorrt_detector_activation = tensorrt_workspace.host_detect_candidates.clone();
    let cuda_face = cuda_faces
        .first()
        .ok_or_else(|| anyhow::anyhow!("CUDA detector found no face"))?;
    let tensorrt_face = tensorrt_faces
        .first()
        .ok_or_else(|| anyhow::anyhow!("TensorRT detector found no face"))?;

    let (detector_mae, detector_max, detector_cosine) =
        tensor_similarity(&cuda_detector_activation, &tensorrt_detector_activation);
    let landmark_max_delta = cuda_face
        .kps_5
        .iter()
        .flatten()
        .zip(tensorrt_face.kps_5.iter().flatten())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "detector activation CUDA↔TRT: MAE={detector_mae:.9}, max={detector_max:.9}, cosine={detector_cosine:.9}, landmark max delta={landmark_max_delta:.6}px"
    );

    let cuda_embedding = FaceRecognizer::recognize_gpu(
        &mut cuda_manager,
        &gpu,
        &frame,
        height,
        width,
        &cuda_face.kps_5,
        &mut cuda_workspace,
    )?;
    let tensorrt_embedding = FaceRecognizer::recognize_gpu(
        &mut tensorrt_manager,
        &gpu,
        &frame,
        height,
        width,
        &cuda_face.kps_5,
        &mut tensorrt_workspace,
    )?;
    let arcface_cosine = FaceRecognizer::cosine_similarity(&cuda_embedding, &tensorrt_embedding);
    let cuda_latent =
        FaceRecognizer::calc_latent(&cuda_embedding, cuda_manager.emap.as_ref().unwrap());
    let tensorrt_latent =
        FaceRecognizer::calc_latent(&tensorrt_embedding, tensorrt_manager.emap.as_ref().unwrap());
    let latent_cosine = FaceRecognizer::cosine_similarity(&cuda_latent, &tensorrt_latent);
    eprintln!(
        "same-crop ArcFace CUDA↔TRT cosine={arcface_cosine:.9}, latent cosine={latent_cosine:.9}"
    );

    assert!(
        detector_cosine > 0.99999,
        "detector activation is the first provider divergence: cosine={detector_cosine:.9}"
    );
    assert!(
        landmark_max_delta < 0.1,
        "detector landmarks diverge between providers: {landmark_max_delta:.6}px"
    );
    assert!(
        arcface_cosine > 0.9999,
        "ArcFace activation diverges on the same crop: cosine={arcface_cosine:.9}"
    );
    assert!(
        latent_cosine > 0.9999,
        "Inswapper latent diverges between providers: cosine={latent_cosine:.9}"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA, TensorRT, ONNX models, and face.jpg"]
fn raw_inswapper_provider_activations_have_the_same_direction() -> anyhow::Result<()> {
    let (stream, gpu) = init_gpu()?;
    let mut cuda_manager = load_cuda_reference_models(&stream)?;
    let (crop, latent) = prepare_identical_swap_inputs(&gpu, &stream, &mut cuda_manager)?;
    let cuda_output = FaceSwapper::swap(&mut cuda_manager, &crop, &latent, 1)?;

    let mut tensorrt_manager = ModelManager::with_provider("models", ExecutionProvider::TensorRT);
    tensorrt_manager.set_compute_stream(stream.cu_stream() as *mut ());
    tensorrt_manager.load("Inswapper128", CANONICAL_SWAPPER_FILENAME)?;
    let tensorrt_output = FaceSwapper::swap(&mut tensorrt_manager, &crop, &latent, 1)?;

    let (mae, max_error, tensor_cosine) = tensor_similarity(&cuda_output, &tensorrt_output);
    eprintln!(
        "raw Inswapper CUDA↔TRT: MAE={mae:.6}, max={max_error:.6}, tensor cosine={tensor_cosine:.9}"
    );

    assert!(
        tensor_cosine > 0.9999,
        "TensorRT raw activation direction diverged from CUDA: cosine={tensor_cosine:.9}"
    );
    Ok(())
}

#[test]
#[ignore = "requires CUDA, TensorRT, ONNX models, and face.jpg"]
fn elon_self_swap_cuda_and_tensorrt_preserve_the_same_identity() -> anyhow::Result<()> {
    let (stream, gpu) = init_gpu()?;
    let image = image::open("assets/photos/face.jpg")?;
    let (width, height) = image.dimensions();
    let rgb = image.to_rgb8();

    let mut cuda_engine = LiveEngine::new_with_provider(
        gpu.clone(),
        Path::new("models"),
        Path::new("assets/photos/face.jpg"),
        FaceSwapParams::default(),
        ExecutionProvider::Cuda,
        &stream,
    )?;
    let cuda_output = cuda_engine.process_rgb(rgb.as_raw(), width, height)?;
    drop(cuda_engine);
    gpu.sync()?;

    let mut tensorrt_engine = LiveEngine::new_with_provider(
        gpu.clone(),
        Path::new("models"),
        Path::new("assets/photos/face.jpg"),
        FaceSwapParams::default(),
        ExecutionProvider::TensorRT,
        &stream,
    )?;
    let tensorrt_output = tensorrt_engine.process_rgb(rgb.as_raw(), width, height)?;

    assert_eq!(cuda_output.faces_detected, 1);
    assert_eq!(cuda_output.faces_swapped, 1);
    assert_eq!(tensorrt_output.faces_detected, 1);
    assert_eq!(tensorrt_output.faces_swapped, 1);

    drop(tensorrt_engine);
    gpu.sync()?;
    let mut identity_manager = ModelManager::with_provider("models", ExecutionProvider::Cuda);
    identity_manager.set_compute_stream(stream.cu_stream() as *mut ());
    identity_manager.load("YoloFace8n", "yoloface_8n.onnx")?;
    identity_manager.load("Inswapper128ArcFace", "w600k_r50.onnx")?;

    let source_embedding = identity_embedding(
        &gpu,
        &stream,
        &mut identity_manager,
        rgb.as_raw(),
        width,
        height,
    )?;
    let cuda_embedding = identity_embedding(
        &gpu,
        &stream,
        &mut identity_manager,
        &cuda_output.data,
        width,
        height,
    )?;
    let tensorrt_embedding = identity_embedding(
        &gpu,
        &stream,
        &mut identity_manager,
        &tensorrt_output.data,
        width,
        height,
    )?;
    let source_to_cuda = FaceRecognizer::cosine_similarity(&source_embedding, &cuda_embedding);
    let source_to_tensorrt =
        FaceRecognizer::cosine_similarity(&source_embedding, &tensorrt_embedding);
    let cuda_to_tensorrt = FaceRecognizer::cosine_similarity(&cuda_embedding, &tensorrt_embedding);
    eprintln!(
        "ArcFace identity cosine: source↔CUDA={source_to_cuda:.9}, source↔TRT={source_to_tensorrt:.9}, CUDA↔TRT={cuda_to_tensorrt:.9}"
    );

    if std::env::var_os("NOPERSON_DUMP_TEST_IMAGES").is_some() {
        std::fs::create_dir_all("artifacts")?;
        save_rgb(
            "artifacts/elon-self-swap-cuda.png",
            &cuda_output.data,
            width,
            height,
        )?;
        save_rgb(
            "artifacts/elon-self-swap-tensorrt.png",
            &tensorrt_output.data,
            width,
            height,
        )?;
    }

    if let Some(reference_path) = std::env::var_os("NOPERSON_PYTHON_REFERENCE") {
        let python = image::open(&reference_path)?.to_rgb8();
        anyhow::ensure!(
            python.width() == width && python.height() == height,
            "Python reference dimensions differ from face.jpg"
        );
        let python_embedding = identity_embedding(
            &gpu,
            &stream,
            &mut identity_manager,
            python.as_raw(),
            width,
            height,
        )?;
        let source_to_python =
            FaceRecognizer::cosine_similarity(&source_embedding, &python_embedding);
        let python_to_cuda = FaceRecognizer::cosine_similarity(&python_embedding, &cuda_embedding);
        let python_to_tensorrt =
            FaceRecognizer::cosine_similarity(&python_embedding, &tensorrt_embedding);
        eprintln!(
            "ArcFace Python baseline: source↔Python={source_to_python:.9}, Python↔CUDA={python_to_cuda:.9}, Python↔TRT={python_to_tensorrt:.9}"
        );
        assert!(
            source_to_cuda + 0.005 >= source_to_python,
            "Rust CUDA loses identity against Python baseline: source↔CUDA={source_to_cuda:.9}, source↔Python={source_to_python:.9}"
        );
        assert!(
            source_to_tensorrt + 0.005 >= source_to_python,
            "Rust TRT loses identity against Python baseline: source↔TRT={source_to_tensorrt:.9}, source↔Python={source_to_python:.9}"
        );
    }

    assert!(
        source_to_cuda > 0.90,
        "CUDA output lost the source identity: cosine={source_to_cuda:.9}"
    );
    assert!(
        source_to_tensorrt > 0.90,
        "TensorRT output lost the source identity: cosine={source_to_tensorrt:.9}"
    );
    assert!(
        cuda_to_tensorrt > 0.999,
        "CUDA and TensorRT changed face identity differently: cosine={cuda_to_tensorrt:.9}"
    );
    Ok(())
}
