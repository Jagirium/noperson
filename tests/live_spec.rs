use std::fs;
use std::path::PathBuf;

use noperson::config::parameters::{EnhancerModel, FaceSwapParams};
use noperson::config::settings::{DetectorModel, ExecutionProvider};
use noperson::engine::ModelRole;
use noperson::live::build_live_spec;

fn fixture_dir() -> PathBuf {
    std::env::temp_dir().join(format!(
        "noperson-live-spec-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ))
}

#[test]
fn live_spec_is_content_addressed_before_gpu_allocation() {
    let root = fixture_dir();
    let models = root.join("models");
    fs::create_dir_all(&models).unwrap();
    for (name, content) in [
        ("yoloface_8n.onnx", b"detector".as_slice()),
        ("w600k_r50.onnx", b"recognizer".as_slice()),
        ("inswapper_128.fp16.onnx", b"swapper".as_slice()),
        ("emap.bin", b"emap".as_slice()),
    ] {
        fs::write(models.join(name), content).unwrap();
    }
    let identity = root.join("identity.jpg");
    fs::write(&identity, b"identity-a").unwrap();

    let first = build_live_spec(
        &models,
        &identity,
        FaceSwapParams::default(),
        ExecutionProvider::Cuda,
        DetectorModel::YoloFace8n,
        0,
    )
    .unwrap();
    first.validate().unwrap();
    let first_generation = first.generation_digest().unwrap();
    assert_eq!(
        first.models[&ModelRole::Detector].filename,
        "yoloface_8n.onnx"
    );

    fs::write(&identity, b"identity-b").unwrap();
    let second = build_live_spec(
        &models,
        &identity,
        FaceSwapParams::default(),
        ExecutionProvider::Cuda,
        DetectorModel::YoloFace8n,
        0,
    )
    .unwrap();
    assert_ne!(first.identity_sha256, second.identity_sha256);
    assert_ne!(first_generation, second.generation_digest().unwrap());

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn live_spec_addresses_the_enabled_frame_enhancer() {
    let root = fixture_dir();
    let models = root.join("models");
    fs::create_dir_all(&models).unwrap();
    for (name, content) in [
        ("yoloface_8n.onnx", b"detector".as_slice()),
        ("w600k_r50.onnx", b"recognizer".as_slice()),
        ("inswapper_128.fp16.onnx", b"swapper".as_slice()),
        ("emap.bin", b"emap".as_slice()),
        ("4x-UltraMix_Smooth.fp16.onnx", b"enhancer".as_slice()),
    ] {
        fs::write(models.join(name), content).unwrap();
    }
    let identity = root.join("identity.jpg");
    fs::write(&identity, b"identity").unwrap();
    let mut params = FaceSwapParams::default();
    params.enhancer_enabled = true;
    params.enhancer_model = EnhancerModel::UltraMixX4;

    let spec = build_live_spec(
        &models,
        &identity,
        params,
        ExecutionProvider::Cuda,
        DetectorModel::YoloFace8n,
        0,
    )
    .unwrap();

    assert_eq!(
        spec.models[&ModelRole::Enhancer].filename,
        "4x-UltraMix_Smooth.fp16.onnx"
    );
    spec.validate().unwrap();
    fs::remove_dir_all(root).unwrap();
}
