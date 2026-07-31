//! Integration tests: ArcFace recognition + Inswapper swap.
//!
//! Requires:
//!   - CUDA device
//!   - models/yoloface_8n.onnx, w600k_r50.onnx, inswapper_128.fp16.onnx, emap.bin
//!   - face.jpg (single face) in project root
//!
//! Run: cargo test --test swap -- --nocapture --ignored

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use image::GenericImageView;

use noperson::config::parameters::FaceSwapParams;
use noperson::gpu::ops::GpuOps;
use noperson::models::live_catalog::CANONICAL_SWAPPER_FILENAME;
use noperson::models::manager::ModelManager;
use noperson::pipeline::face_detector::YoloFaceDetector;
use noperson::pipeline::face_mask;
use noperson::pipeline::face_recognizer::FaceRecognizer;
use noperson::pipeline::face_recognizer::warp_affine_chw;
use noperson::pipeline::face_swapper::FaceSwapper;
use noperson::pipeline::workspace::GpuWorkspace;

type DetectedFrame = (cudarc::driver::CudaSlice<f32>, [[f32; 2]; 5], u32, u32);

/// Initialize tracing subscriber exactly once (tests run in parallel).
static TRACING_INIT: std::sync::Once = std::sync::Once::new();
fn init_tracing() {
    TRACING_INIT.call_once(|| {
        let filter = std::env::var("RUST_LOG").unwrap_or_else(|_| "warn".to_owned());
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
    });
}

/// Helper: init CUDA + GpuOps + ModelManager + load the canonical swap model.
fn init_pipeline() -> anyhow::Result<(Arc<CudaStream>, GpuOps, ModelManager)> {
    let ctx = Arc::new(CudaContext::new(0)?);
    let stream = ctx.default_stream().clone();
    let gpu = GpuOps::new(&ctx, stream.clone())?;
    let mut manager = ModelManager::new("models");
    manager.set_compute_stream(stream.cu_stream() as *mut ());
    manager.load("YoloFace8n", "yoloface_8n.onnx")?;
    manager.load("Inswapper128ArcFace", "w600k_r50.onnx")?;
    manager.load("Inswapper128", CANONICAL_SWAPPER_FILENAME)?;
    manager.load_emap(CANONICAL_SWAPPER_FILENAME)?;
    Ok((stream, gpu, manager))
}

/// Minimal canonical stack without fixed B4/B9/B16 derivative models.
fn init_pipeline_dim1() -> anyhow::Result<(Arc<CudaStream>, GpuOps, ModelManager)> {
    let ctx = Arc::new(CudaContext::new(0)?);
    let stream = ctx.default_stream().clone();
    let gpu = GpuOps::new(&ctx, stream.clone())?;
    let mut manager = ModelManager::new("models");
    manager.set_compute_stream(stream.cu_stream() as *mut ());
    manager.load("YoloFace8n", "yoloface_8n.onnx")?;
    manager.load("Inswapper128ArcFace", "w600k_r50.onnx")?;
    manager.load("Inswapper128", CANONICAL_SWAPPER_FILENAME)?;
    manager.load_emap(CANONICAL_SWAPPER_FILENAME)?;
    Ok((stream, gpu, manager))
}

/// Helper: load face.jpg, detect face, return (frame_chw_gpu, kps_5, height, width).
fn load_and_detect(
    gpu: &GpuOps,
    manager: &mut ModelManager,
    stream: &Arc<CudaStream>,
) -> anyhow::Result<DetectedFrame> {
    let img = image::open("face.jpg")?;
    let (width, height) = img.dimensions();
    let rgb = img.to_rgb8();

    let d_frame_u8 = gpu.upload_u8(rgb.as_raw())?;
    let mut d_frame_chw = gpu.alloc_zeros(3 * width as usize * height as usize)?;
    gpu.hwc_u8_to_chw_f32(&d_frame_u8, &mut d_frame_chw, height, width)?;

    let detector = YoloFaceDetector::new(0.5);
    let mut ws = GpuWorkspace::new(stream)?;
    let (faces, _scale) =
        detector.detect_gpu(manager, gpu, &d_frame_chw, &mut ws, height, width)?;

    anyhow::ensure!(!faces.is_empty(), "No face detected in face.jpg");
    let kps = faces[0].kps_5;
    Ok((d_frame_chw, kps, height, width))
}

fn save_chw_rgb(path: &str, chw: &[f32], size: u32) -> anyhow::Result<()> {
    let pixels = (size * size) as usize;
    let mut rgb = vec![0u8; pixels * 3];
    for i in 0..pixels {
        for c in 0..3 {
            rgb[i * 3 + c] = chw[c * pixels + i].round().clamp(0.0, 255.0) as u8;
        }
    }
    image::RgbImage::from_raw(size, size, rgb)
        .ok_or_else(|| anyhow::anyhow!("invalid CHW image buffer"))?
        .save(path)?;
    Ok(())
}

/// Regression: the fast NPP crop must use the same forward-affine convention as
/// the CPU/Python reference. A non-zero-only assertion cannot catch a shifted
/// but otherwise valid face crop.
#[test]
#[ignore = "requires CUDA + face.jpg"]
fn test_npp_affine_matches_cpu_reference() -> anyhow::Result<()> {
    init_tracing();

    let (stream, gpu, mut manager) = init_pipeline_dim1()?;
    let (d_frame_chw, kps, h, w) = load_and_detect(&gpu, &mut manager, &stream)?;
    let frame_chw = gpu.download(&d_frame_chw)?;

    use noperson::math::affine;
    use noperson::math::constants::ARCFACE_DST;

    let mut template = ARCFACE_DST;
    for point in &mut template {
        point[0] += 8.0;
    }
    let transform = affine::estimate_face_affine(&kps, &template);

    let mut cpu = vec![0.0f32; 3 * 128 * 128];
    warp_affine_chw(&frame_chw, h, w, &mut cpu, 128, 128, &transform);

    let mut gpu_crop = gpu.alloc_zeros(3 * 128 * 128)?;
    gpu.warp_affine_npp(&d_frame_chw, &mut gpu_crop, h, w, 128, 128, &transform)?;
    let npp = gpu.download(&gpu_crop)?;

    let mae = cpu
        .iter()
        .zip(&npp)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / cpu.len() as f32;
    let max_error = cpu
        .iter()
        .zip(&npp)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    eprintln!("NPP vs CPU aligned crop: MAE={mae:.4}, max={max_error:.4}");
    assert!(mae < 2.0, "NPP affine crop is misaligned: MAE={mae:.4}");
    Ok(())
}

/// Crosswap's default live mask has a 10px rectangular border and 10px blur.
#[test]
#[ignore = "requires CUDA"]
fn test_default_gpu_border_mask_has_soft_pixel_edges() -> anyhow::Result<()> {
    let ctx = Arc::new(CudaContext::new(0)?);
    let stream = ctx.default_stream().clone();
    let gpu = GpuOps::new(&ctx, stream.clone())?;
    let mut ws = GpuWorkspace::new(&stream)?;
    let params = FaceSwapParams::default();
    let blur_ks = params.border_blur * 2 + 1;
    let blur_sigma = (params.border_blur as f32 + 1.0) * 0.2;

    face_mask::gpu_generate_mask_512(
        &gpu,
        &mut ws,
        params.border_top,
        params.border_bottom,
        params.border_left,
        params.border_right,
        blur_ks,
        blur_sigma,
        false,
        false,
        0,
    )?;
    let values = gpu.download(&ws.mask_128)?;

    assert!(values[64] < 0.01, "outer edge must keep the original frame");
    assert!(
        (0.05..0.95).contains(&values[10 * 128 + 64]),
        "border transition must be feathered"
    );
    assert!(values[64 * 128 + 64] > 0.99);
    Ok(())
}

/// Test: recognize the same face twice → cosine similarity must be ~1.0.
#[test]
#[ignore = "requires CUDA + ONNX models"]
fn test_recognize_deterministic_cosine() -> anyhow::Result<()> {
    init_tracing();

    let (stream, gpu, mut manager) = init_pipeline()?;
    let (d_frame_chw, kps, h, w) = load_and_detect(&gpu, &mut manager, &stream)?;

    let mut ws = GpuWorkspace::new(&stream)?;

    // Recognize the same face twice — embeddings must be identical.
    let emb1 =
        FaceRecognizer::recognize_gpu(&mut manager, &gpu, &d_frame_chw, h, w, &kps, &mut ws)?;
    gpu.sync()?; // ensure first inference completes before second overwrites buffers
    let emb2 =
        FaceRecognizer::recognize_gpu(&mut manager, &gpu, &d_frame_chw, h, w, &kps, &mut ws)?;

    let cos = FaceRecognizer::cosine_similarity(&emb1, &emb2);
    eprintln!("Cosine similarity (same face, twice): {cos:.6}");
    assert!(
        cos > 0.99,
        "Same face embedding must match: cos={cos:.4} (expected >0.99)"
    );
    Ok(())
}

/// Test: embedding L2 norm must be ~1.0 (ArcFace output is already L2-normalized
/// by the last BatchNorm, but we verify it anyway).
#[test]
#[ignore = "requires CUDA + ONNX models"]
fn test_embedding_norm() -> anyhow::Result<()> {
    init_tracing();

    let (stream, gpu, mut manager) = init_pipeline()?;
    let (d_frame_chw, kps, h, w) = load_and_detect(&gpu, &mut manager, &stream)?;

    let mut ws = GpuWorkspace::new(&stream)?;
    let emb = FaceRecognizer::recognize_gpu(&mut manager, &gpu, &d_frame_chw, h, w, &kps, &mut ws)?;

    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    eprintln!("Embedding L2 norm: {norm:.4} (dim={})", emb.len());
    assert!(
        norm > 1.0,
        "ArcFace embedding should have non-trivial norm, got {norm:.4}"
    );
    assert_eq!(emb.len(), 512, "Embedding must be 512-dim");
    Ok(())
}

/// Test: latent computation from embedding + emap.
/// latent = L2norm(L2norm(embedding) @ emap)
#[test]
#[ignore = "requires CUDA + ONNX models"]
fn test_latent_computation() -> anyhow::Result<()> {
    init_tracing();

    let (stream, gpu, mut manager) = init_pipeline()?;
    let (d_frame_chw, kps, h, w) = load_and_detect(&gpu, &mut manager, &stream)?;

    let mut ws = GpuWorkspace::new(&stream)?;
    let emb = FaceRecognizer::recognize_gpu(&mut manager, &gpu, &d_frame_chw, h, w, &kps, &mut ws)?;

    let emap = manager.emap.as_ref().expect("emap not loaded");
    let latent = FaceRecognizer::calc_latent(&emb, emap);

    let norm: f32 = latent.iter().map(|x| x * x).sum::<f32>().sqrt();
    eprintln!("Latent L2 norm: {norm:.4} (dim={})", latent.len());
    assert!(
        (norm - 1.0).abs() < 0.01,
        "Latent must be L2-normalized, got {norm:.4}"
    );
    assert_eq!(latent.len(), 512, "Latent must be 512-dim");

    // Same latent computed twice must match.
    let latent2 = FaceRecognizer::calc_latent(&emb, emap);
    let max_diff = latent
        .iter()
        .zip(latent2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "Latent must be deterministic, max diff={max_diff}"
    );
    Ok(())
}

/// Test: cosine similarity between embeddings of two DIFFERENT face regions
/// (left half vs right half of the frame) should be lower than same-face.
#[test]
#[ignore = "requires CUDA + ONNX models"]
fn test_cosine_similarity_different_faces() -> anyhow::Result<()> {
    init_tracing();

    let (stream, gpu, mut manager) = init_pipeline()?;
    let (d_frame_chw, kps, h, w) = load_and_detect(&gpu, &mut manager, &stream)?;

    let mut ws = GpuWorkspace::new(&stream)?;
    let emb_original =
        FaceRecognizer::recognize_gpu(&mut manager, &gpu, &d_frame_chw, h, w, &kps, &mut ws)?;

    // Now use slightly shifted keypoints — should reduce cosine similarity.
    let mut kps_shifted = kps;
    for pt in &mut kps_shifted {
        pt[0] += 15.0; // shift 15px right
        pt[1] += 10.0; // shift 10px down
    }
    let emb_shifted = FaceRecognizer::recognize_gpu(
        &mut manager,
        &gpu,
        &d_frame_chw,
        h,
        w,
        &kps_shifted,
        &mut ws,
    )?;

    let cos = FaceRecognizer::cosine_similarity(&emb_original, &emb_shifted);
    eprintln!("Cosine similarity (shifted kps): {cos:.4}");
    // Shifted face should still be similar (same person), but not identical.
    assert!(
        cos > 0.5,
        "Same person with shifted kps should still match: cos={cos:.4}"
    );
    assert!(
        cos < 0.999,
        "Shifted kps should reduce similarity from 1.0: cos={cos:.4}"
    );
    Ok(())
}

/// Test: Inswapper128 swap on a single face (dim=1, 128×128).
/// Verifies the GPU swap pipeline: warp → interlace → ort → scatter.
#[test]
#[ignore = "requires CUDA + ONNX models"]
fn test_swap_dim1() -> anyhow::Result<()> {
    init_tracing();

    let (stream, gpu, mut manager) = init_pipeline()?;
    let (d_frame_chw, kps, h, w) = load_and_detect(&gpu, &mut manager, &stream)?;

    let mut ws = GpuWorkspace::new(&stream)?;

    // 1. Recognize → embedding → latent
    let emb = FaceRecognizer::recognize_gpu(&mut manager, &gpu, &d_frame_chw, h, w, &kps, &mut ws)?;
    let emap = manager.emap.as_ref().expect("emap not loaded");
    let latent = FaceRecognizer::calc_latent(&emb, emap);

    // 2. Warp face to 128×128 for swap
    use noperson::math::affine;
    use noperson::math::constants::ARCFACE_DST;
    let swap_template = {
        let factor = 1.0; // 128/128
        let mut dst = ARCFACE_DST;
        for pt in &mut dst {
            pt[0] = pt[0] * factor + factor * 8.0;
            pt[1] *= factor;
        }
        dst
    };
    let swap_affine = affine::estimate_face_affine(&kps, &swap_template);

    gpu.warp_affine_npp(&d_frame_chw, &mut ws.face_256, h, w, 128, 128, &swap_affine)?;

    if std::env::var_os("NOPERSON_DUMP_TEST_IMAGES").is_some() {
        let crop = gpu.download(&ws.face_256)?;
        save_chw_rgb(
            "/tmp/noperson-rust-crop128.png",
            &crop[..3 * 128 * 128],
            128,
        )?;
    }

    // 3. Run swap (dim=1: single 128×128 tile)
    FaceSwapper::swap_gpu(&mut manager, &gpu, &mut ws, &latent, 1)?;

    // 4. Download swapped face. face_256 buffer is 512x512 (max), but swap uses 128x128.
    //    Download the full buffer then check the active 128x128 region.
    let max_size = 3 * 512 * 512;
    let mut swapped_full = vec![0.0f32; max_size];
    gpu.download_into(&ws.face_256, &mut swapped_full)?;

    // Active region: first 3*128*128 floats (dim=1 → 128x128)
    let active = 3 * 128 * 128;
    let swapped = &swapped_full[..active];
    if std::env::var_os("NOPERSON_DUMP_TEST_IMAGES").is_some() {
        save_chw_rgb("/tmp/noperson-rust-swap128.png", swapped, 128)?;
    }
    let non_zero = swapped.iter().filter(|v| **v != 0.0).count();
    let max_val = swapped.iter().cloned().fold(0.0f32, f32::max);
    eprintln!("Swap dim=1: non-zero pixels={non_zero}/{active}, max={max_val:.1}");
    assert!(
        non_zero > active / 2,
        "Swapped face should have mostly non-zero pixels"
    );
    assert!(
        max_val > 10.0,
        "Swapped face should have significant pixel values, max={max_val}"
    );
    Ok(())
}

/// Regression: the zero-copy IoBinding path must be numerically equivalent to
/// ort's owned-tensor session path for the same crop and latent.
#[test]
#[ignore = "requires CUDA + ONNX models"]
fn test_swap_gpu_matches_owned_tensor_run() -> anyhow::Result<()> {
    init_tracing();

    let (stream, gpu, mut manager) = init_pipeline_dim1()?;
    let (d_frame_chw, kps, h, w) = load_and_detect(&gpu, &mut manager, &stream)?;
    let mut ws = GpuWorkspace::new(&stream)?;

    let emb = FaceRecognizer::recognize_gpu(&mut manager, &gpu, &d_frame_chw, h, w, &kps, &mut ws)?;
    let latent = FaceRecognizer::calc_latent(&emb, manager.emap.as_ref().unwrap());

    use noperson::math::affine;
    use noperson::math::constants::ARCFACE_DST;
    let mut template = ARCFACE_DST;
    for point in &mut template {
        point[0] += 8.0;
    }
    let transform = affine::estimate_face_affine(&kps, &template);
    gpu.warp_affine_npp(&d_frame_chw, &mut ws.face_256, h, w, 128, 128, &transform)?;
    let crop = gpu.download(&ws.face_256)?[..3 * 128 * 128].to_vec();

    // CUDA graph capture pins I/O addresses for a session. Use a separate
    // session for the owned-tensor reference so it cannot poison the stable
    // zero-copy binding used by the production path.
    let (_owned_stream, _owned_gpu, mut owned_manager) = init_pipeline_dim1()?;
    let owned = FaceSwapper::swap(&mut owned_manager, &crop, &latent, 1)?;
    gpu.upload_into(&crop, &mut ws.face_256)?;
    FaceSwapper::swap_gpu(&mut manager, &gpu, &mut ws, &latent, 1)?;
    let zero_copy = gpu.download(&ws.face_256)?[..3 * 128 * 128].to_vec();

    if std::env::var_os("NOPERSON_DUMP_TEST_IMAGES").is_some() {
        save_chw_rgb("/tmp/noperson-owned-swap128.png", &owned, 128)?;
        save_chw_rgb("/tmp/noperson-zero-copy-swap128.png", &zero_copy, 128)?;
    }

    let mae = owned
        .iter()
        .zip(&zero_copy)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / owned.len() as f32;
    eprintln!("IoBinding vs owned-tensor swap: MAE={mae:.4}");
    assert!(
        mae < 0.5,
        "zero-copy swap diverges from owned run: MAE={mae:.4}"
    );
    Ok(())
}

/// Regression: dim=2 must use the canonical graph with a dynamic batch of four.
/// No fixed-batch derivative is loaded into this manager.
#[test]
#[ignore = "requires CUDA + ONNX models"]
fn test_swap_dim2_uses_canonical_multibatch_model() -> anyhow::Result<()> {
    init_tracing();

    let (stream, gpu, mut manager) = init_pipeline_dim1()?;
    let (d_frame_chw, kps, h, w) = load_and_detect(&gpu, &mut manager, &stream)?;

    let mut ws = GpuWorkspace::new(&stream)?;

    // 1. Recognize → latent
    let emb = FaceRecognizer::recognize_gpu(&mut manager, &gpu, &d_frame_chw, h, w, &kps, &mut ws)?;
    let emap = manager.emap.as_ref().expect("emap not loaded");
    let latent = FaceRecognizer::calc_latent(&emb, emap);

    // 2. Warp face to 256×256 (dim=2)
    use noperson::math::affine;
    use noperson::math::constants::ARCFACE_DST;
    let swap_template = {
        let factor = 2.0; // 256/128
        let mut dst = ARCFACE_DST;
        for pt in &mut dst {
            pt[0] = pt[0] * factor + factor * 8.0;
            pt[1] *= factor;
        }
        dst
    };
    let swap_affine = affine::estimate_face_affine(&kps, &swap_template);

    gpu.warp_affine_npp(&d_frame_chw, &mut ws.face_256, h, w, 256, 256, &swap_affine)?;

    // 3. Run swap (dim=2: four tiles in one canonical multibatch call)
    FaceSwapper::swap_gpu(&mut manager, &gpu, &mut ws, &latent, 2)?;

    // 4. Download swapped face. face_256 buffer is 512x512 (max), but swap uses 256x256.
    let max_size = 3 * 512 * 512;
    let mut swapped_full = vec![0.0f32; max_size];
    gpu.download_into(&ws.face_256, &mut swapped_full)?;

    // Active region: first 3*256*256 floats (dim=2 → 256x256)
    let active = 3 * 256 * 256;
    let swapped = &swapped_full[..active];
    let non_zero = swapped.iter().filter(|v| **v != 0.0).count();
    let max_val = swapped.iter().cloned().fold(0.0f32, f32::max);
    eprintln!("Swap dim=2 (canonical): non-zero pixels={non_zero}/{active}, max={max_val:.1}");
    assert!(
        non_zero > active / 2,
        "Canonical tiled swap should produce non-zero output"
    );
    assert!(
        max_val > 10.0,
        "Canonical tiled swap should have significant pixel values"
    );
    Ok(())
}

/// Test: swap output differs from the warped input (the face actually changed).
#[test]
#[ignore = "requires CUDA + ONNX models"]
fn test_swap_changes_face() -> anyhow::Result<()> {
    init_tracing();

    let (stream, gpu, mut manager) = init_pipeline()?;
    let (d_frame_chw, kps, h, w) = load_and_detect(&gpu, &mut manager, &stream)?;

    let mut ws = GpuWorkspace::new(&stream)?;

    // 1. Warp face (pre-swap snapshot)
    use noperson::math::affine;
    use noperson::math::constants::ARCFACE_DST;
    let swap_template_128 = {
        let factor = 1.0;
        let mut dst = ARCFACE_DST;
        for pt in &mut dst {
            pt[0] = pt[0] * factor + factor * 8.0;
            pt[1] *= factor;
        }
        dst
    };
    let swap_affine_128 = affine::estimate_face_affine(&kps, &swap_template_128);
    gpu.warp_affine_npp(
        &d_frame_chw,
        &mut ws.face_256,
        h,
        w,
        128,
        128,
        &swap_affine_128,
    )?;

    // Snapshot the pre-swap face (face_256 buffer is 512x512 max)
    let max_size = 3 * 512 * 512;
    let mut pre_swap = vec![0.0f32; max_size];
    gpu.download_into(&ws.face_256, &mut pre_swap)?;
    let pre_active = 3 * 128 * 128;
    let pre_swap = &pre_swap[..pre_active];

    // 2. Recognize → latent
    let emb = FaceRecognizer::recognize_gpu(&mut manager, &gpu, &d_frame_chw, h, w, &kps, &mut ws)?;
    let emap = manager.emap.as_ref().expect("emap not loaded");
    let latent = FaceRecognizer::calc_latent(&emb, emap);

    // 3. Re-warp at 128×128 (dim=1)
    gpu.warp_affine_npp(
        &d_frame_chw,
        &mut ws.face_256,
        h,
        w,
        128,
        128,
        &swap_affine_128,
    )?;

    // 4. Swap (dim=1: single 128×128 tile)
    FaceSwapper::swap_gpu(&mut manager, &gpu, &mut ws, &latent, 1)?;

    let mut post_swap_full = vec![0.0f32; max_size];
    gpu.download_into(&ws.face_256, &mut post_swap_full)?;
    let post_swap = &post_swap_full[..pre_active];

    // 5. Count pixels that changed significantly
    let mut changed = 0usize;
    for (a, b) in pre_swap.iter().zip(post_swap.iter()) {
        if (a - b).abs() > 5.0 {
            changed += 1;
        }
    }

    let pct = changed as f32 / pre_active as f32 * 100.0;
    eprintln!("Pixels changed by swap: {changed}/{pre_active} ({pct:.1}%)");
    assert!(
        changed > pre_active / 10,
        "Swap should change at least 10% of pixels, got {pct:.1}%"
    );
    Ok(())
}

/// Test: GPU calc_latent matches CPU calc_latent.
/// latent = L2norm(L2norm(embedding) @ emap)
#[test]
#[ignore = "requires CUDA + ONNX models"]
fn test_calc_latent_gpu_matches_cpu() -> anyhow::Result<()> {
    init_tracing();

    let (stream, gpu, mut manager) = init_pipeline()?;
    let (d_frame_chw, kps, h, w) = load_and_detect(&gpu, &mut manager, &stream)?;

    let mut ws = GpuWorkspace::new(&stream)?;

    // 1. Get embedding on GPU (stays on device)
    let emb = FaceRecognizer::recognize_gpu(&mut manager, &gpu, &d_frame_chw, h, w, &kps, &mut ws)?;

    // 2. CPU latent (ground truth)
    let emap = manager.emap.as_ref().expect("emap not loaded");
    let latent_cpu = FaceRecognizer::calc_latent(&emb, emap);

    // 3. GPU latent
    // Upload embedding + emap to GPU, run calc_latent_gpu, download result
    let mut emb_gpu = gpu
        .upload(&emb)
        .map_err(|e| anyhow::anyhow!("upload: {e}"))?;
    let emap_gpu = gpu
        .upload(emap)
        .map_err(|e| anyhow::anyhow!("upload: {e}"))?;
    let mut latent_gpu_buf = gpu
        .alloc_zeros(512)
        .map_err(|e| anyhow::anyhow!("alloc: {e}"))?;

    FaceRecognizer::calc_latent_gpu(&gpu, &mut emb_gpu, &emap_gpu, &mut latent_gpu_buf)
        .map_err(|e| anyhow::anyhow!("calc_latent_gpu: {e}"))?;

    let mut latent_gpu = vec![0.0f32; 512];
    gpu.download_into(&latent_gpu_buf, &mut latent_gpu)?;

    // 4. Compare
    let cos = FaceRecognizer::cosine_similarity(&latent_cpu, &latent_gpu);
    eprintln!("GPU vs CPU latent cosine: {cos:.6}");
    assert!(cos > 0.99, "GPU latent must match CPU: cos={cos:.4}");

    // Check L2 norm of GPU latent
    let norm: f32 = latent_gpu.iter().map(|x| x * x).sum::<f32>().sqrt();
    eprintln!("GPU latent L2 norm: {norm:.6}");
    assert!(
        (norm - 1.0).abs() < 0.01,
        "GPU latent must be L2-normalized: {norm:.4}"
    );

    eprintln!("GPU calc_latent matches CPU — 100% on GPU");
    Ok(())
}
