use std::collections::BTreeMap;

use noperson::config::parameters::{FaceSwapParams, RestorerSize};
use noperson::config::settings::{DetectorModel, ExecutionProvider};
use noperson::engine::{EngineSpec, EngineSpecError, ModelArtifact, ModelRole};

const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn artifact(logical_name: &str, filename: &str, sha256: &str) -> ModelArtifact {
    ModelArtifact {
        logical_name: logical_name.to_owned(),
        filename: filename.to_owned(),
        sha256: sha256.to_owned(),
    }
}

fn required_models(reverse: bool) -> BTreeMap<ModelRole, ModelArtifact> {
    let entries = [
        (
            ModelRole::Detector,
            artifact("YoloFace8n", "yoloface_8n.onnx", SHA_A),
        ),
        (
            ModelRole::Recognizer,
            artifact("ArcFaceW600kR50", "w600k_r50.onnx", SHA_A),
        ),
        (
            ModelRole::Swapper,
            artifact("Inswapper128", "inswapper_128.fp16.onnx", SHA_B),
        ),
        (
            ModelRole::Emap,
            artifact("InswapperEmap", "emap.bin", SHA_B),
        ),
    ];
    if reverse {
        entries.into_iter().rev().collect()
    } else {
        entries.into_iter().collect()
    }
}

fn valid_spec(reverse: bool) -> EngineSpec {
    EngineSpec {
        provider: ExecutionProvider::Cuda,
        device_id: 0,
        detector: DetectorModel::YoloFace8n,
        identity_sha256: SHA_A.to_owned(),
        models: required_models(reverse),
        params: FaceSwapParams::default(),
    }
}

#[test]
fn valid_spec_has_stable_generation_digest() {
    let forward = valid_spec(false);
    let reverse = valid_spec(true);

    forward.validate().expect("valid engine spec");
    reverse.validate().expect("valid engine spec");
    assert_eq!(forward.generation_digest().unwrap().len(), 64);
    assert_eq!(forward.generation_digest(), reverse.generation_digest());
}

#[test]
fn missing_required_model_is_rejected() {
    let mut spec = valid_spec(false);
    spec.models.remove(&ModelRole::Swapper);

    assert_eq!(
        spec.validate(),
        Err(EngineSpecError::MissingModel(ModelRole::Swapper))
    );
}

#[test]
fn malformed_digest_is_rejected_before_model_loading() {
    let mut spec = valid_spec(false);
    spec.models.get_mut(&ModelRole::Detector).unwrap().sha256 = "A".repeat(64);

    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidSha256 { field, .. }) if field == "models.detector.sha256"
    ));
}

#[test]
fn gpen_1024_and_2048_are_excluded_from_new_generations() {
    let mut by_params = valid_spec(false);
    by_params.params.restorer_size = RestorerSize::Gpen1024;
    assert_eq!(
        by_params.validate(),
        Err(EngineSpecError::UnsupportedRestorer("GPEN-1024".to_owned()))
    );

    let mut by_artifact = valid_spec(false);
    by_artifact.models.insert(
        ModelRole::Restorer,
        artifact("GPEN-2048", "gpen_2048.onnx", SHA_A),
    );
    assert_eq!(
        by_artifact.validate(),
        Err(EngineSpecError::UnsupportedRestorer("GPEN-2048".to_owned()))
    );
}

#[test]
fn enabled_optional_stage_requires_its_artifact() {
    let mut spec = valid_spec(false);
    spec.params.restorer_enabled = true;

    assert_eq!(
        spec.validate(),
        Err(EngineSpecError::MissingModel(ModelRole::Restorer))
    );
}
