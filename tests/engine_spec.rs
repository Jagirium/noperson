use std::collections::BTreeMap;

use noperson::config::parameters::{
    EnhancerModel, FaceSwapParams, RestorerAlignment, RestorerSize, SwapperModel,
};
use noperson::config::settings::{DetectorModel, ExecutionProvider};
use noperson::engine::{EngineSpec, EngineSpecError, FaceAssignmentSpec, ModelArtifact, ModelRole};

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
        assignments: Vec::new(),
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
fn face_assignments_are_generation_identity_and_swap_all_is_last() {
    let mut first = valid_spec(false);
    first.assignments = vec![
        FaceAssignmentSpec {
            source_identity_sha256: SHA_A.to_owned(),
            target_identity_sha256: Some(SHA_B.to_owned()),
            similarity_threshold: 0.6,
            params: None,
            models: BTreeMap::new(),
        },
        FaceAssignmentSpec {
            source_identity_sha256: SHA_B.to_owned(),
            target_identity_sha256: None,
            similarity_threshold: 0.5,
            params: None,
            models: BTreeMap::new(),
        },
    ];
    first.validate().unwrap();

    let mut changed = first.clone();
    changed.assignments[0].similarity_threshold = 0.61;
    assert_ne!(first.generation_digest(), changed.generation_digest());

    first.assignments.swap(0, 1);
    assert!(matches!(
        first.validate(),
        Err(EngineSpecError::InvalidAssignments(_))
    ));
}

#[test]
fn per_face_parameters_are_part_of_generation_identity() {
    let mut first = valid_spec(false);
    first.assignments = vec![FaceAssignmentSpec {
        source_identity_sha256: SHA_A.to_owned(),
        target_identity_sha256: Some(SHA_B.to_owned()),
        similarity_threshold: 0.6,
        params: Some(FaceSwapParams::default()),
        models: BTreeMap::new(),
    }];
    let mut changed = first.clone();
    changed.assignments[0].params.as_mut().unwrap().strength = 2.0;

    assert_ne!(first.generation_digest(), changed.generation_digest());
}

#[test]
fn per_face_parameters_are_validated_and_can_select_content_addressed_models() {
    let mut spec = valid_spec(false);
    let params = FaceSwapParams {
        restorer_alpha: 2.0,
        ..FaceSwapParams::default()
    };
    spec.assignments = vec![FaceAssignmentSpec {
        source_identity_sha256: SHA_A.to_owned(),
        target_identity_sha256: Some(SHA_B.to_owned()),
        similarity_threshold: 0.6,
        params: Some(params),
        models: BTreeMap::new(),
    }];
    assert!(spec.validate().is_err());

    spec.assignments[0].params.as_mut().unwrap().restorer_alpha = 1.0;
    spec.assignments[0].params.as_mut().unwrap().swapper_model = SwapperModel::Dfm;
    spec.assignments[0].params.as_mut().unwrap().dfm_model = "statham.dfm".to_owned();
    spec.assignments[0]
        .models
        .insert(ModelRole::Dfm, artifact("StathamDFM", "statham.dfm", SHA_B));
    spec.validate().expect("assignment-local DFM is valid");

    let digest = spec.generation_digest().unwrap();
    spec.assignments[0]
        .models
        .get_mut(&ModelRole::Dfm)
        .unwrap()
        .sha256 = SHA_A.to_owned();
    assert_ne!(digest, spec.generation_digest().unwrap());
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
fn targeted_dfm_generation_requires_arcface_for_assignment_matching() {
    let mut spec = valid_spec(false);
    spec.params.swapper_model = SwapperModel::Dfm;
    spec.params.dfm_model = "statham.dfm".to_owned();
    spec.models.remove(&ModelRole::Swapper);
    spec.models.remove(&ModelRole::Emap);
    spec.models.remove(&ModelRole::Recognizer);
    spec.models
        .insert(ModelRole::Dfm, artifact("StathamDFM", "statham.dfm", SHA_B));
    spec.assignments = vec![FaceAssignmentSpec {
        source_identity_sha256: SHA_A.to_owned(),
        target_identity_sha256: Some(SHA_B.to_owned()),
        similarity_threshold: 0.6,
        params: None,
        models: BTreeMap::new(),
    }];
    assert_eq!(
        spec.validate(),
        Err(EngineSpecError::MissingModel(ModelRole::Recognizer))
    );
}

#[test]
fn assignment_model_union_rejects_runtime_session_name_collisions() {
    let mut spec = valid_spec(false);
    let params = FaceSwapParams {
        occluder_enabled: true,
        ..FaceSwapParams::default()
    };
    spec.assignments = vec![
        FaceAssignmentSpec {
            source_identity_sha256: SHA_A.to_owned(),
            target_identity_sha256: Some(SHA_A.to_owned()),
            similarity_threshold: 0.6,
            params: Some(params.clone()),
            models: BTreeMap::from([(
                ModelRole::Occluder,
                artifact("Occluder", "occluder-a.onnx", SHA_A),
            )]),
        },
        FaceAssignmentSpec {
            source_identity_sha256: SHA_B.to_owned(),
            target_identity_sha256: Some(SHA_B.to_owned()),
            similarity_threshold: 0.6,
            params: Some(params),
            models: BTreeMap::from([(
                ModelRole::Occluder,
                artifact("Occluder", "occluder-b.onnx", SHA_B),
            )]),
        },
    ];
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidAssignments(message)) if message.contains("session Occluder")
    ));
}

#[test]
fn assignment_cannot_override_generation_wide_detection_or_enhancement() {
    let mut spec = valid_spec(false);
    let params = FaceSwapParams {
        detector_score: 0.7,
        ..FaceSwapParams::default()
    };
    spec.assignments = vec![FaceAssignmentSpec {
        source_identity_sha256: SHA_A.to_owned(),
        target_identity_sha256: Some(SHA_B.to_owned()),
        similarity_threshold: 0.6,
        params: Some(params),
        models: BTreeMap::new(),
    }];
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidAssignments(message)) if message.contains("generation-wide")
    ));
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

#[test]
fn enabled_second_restorer_requires_an_independent_artifact() {
    let mut spec = valid_spec(false);
    spec.params.restorer2_enabled = true;

    assert_eq!(
        spec.validate(),
        Err(EngineSpecError::MissingModel(ModelRole::Restorer2))
    );

    spec.models.insert(
        ModelRole::Restorer2,
        artifact("GPENBFR256", "GPEN-BFR-256.onnx", SHA_A),
    );
    spec.validate().expect("valid second-restorer generation");
}

#[test]
fn reference_restorer_alignment_requires_five_point_landmarks() {
    let mut spec = valid_spec(false);
    spec.params.restorer_enabled = true;
    spec.params.restorer_alignment = RestorerAlignment::Reference;
    spec.models.insert(
        ModelRole::Restorer,
        artifact("GPENBFR256", "GPEN-BFR-256.onnx", SHA_A),
    );

    assert_eq!(
        spec.validate(),
        Err(EngineSpecError::MissingModel(ModelRole::RestorerLandmark))
    );

    spec.models.insert(
        ModelRole::RestorerLandmark,
        artifact("FaceLandmark5", "res50.onnx", SHA_B),
    );
    spec.validate().expect("valid reference-aligned restorer");
}

#[test]
fn enabled_landmark_refiner_requires_its_generation_artifact() {
    let mut spec = valid_spec(false);
    spec.params.landmark_enabled = true;

    assert_eq!(
        spec.validate(),
        Err(EngineSpecError::MissingModel(ModelRole::Landmark))
    );

    spec.models.insert(
        ModelRole::Landmark,
        artifact("FaceLandmark203", "landmark.onnx", SHA_A),
    );
    spec.validate().expect("valid landmark generation");

    spec.models
        .get_mut(&ModelRole::Landmark)
        .unwrap()
        .logical_name = "FaceLandmark68".to_owned();
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::LandmarkModelMismatch { .. })
    ));
}

#[test]
fn dfm_generation_replaces_the_inswapper_model_group() {
    let mut spec = valid_spec(false);
    spec.params.swapper_model = SwapperModel::Dfm;
    spec.params.dfm_model = "JasonStatham320.dfm".to_owned();
    spec.models.remove(&ModelRole::Recognizer);
    spec.models.remove(&ModelRole::Swapper);
    spec.models.remove(&ModelRole::Emap);

    assert_eq!(
        spec.validate(),
        Err(EngineSpecError::MissingModel(ModelRole::Dfm))
    );

    spec.models.insert(
        ModelRole::Dfm,
        artifact("JasonStatham320", "JasonStatham320.dfm", SHA_A),
    );
    spec.validate().expect("valid DFM generation");
}

#[test]
fn dfm_controls_and_artifact_selection_are_generation_identity() {
    let mut spec = valid_spec(false);
    spec.params.swapper_model = SwapperModel::Dfm;
    spec.params.dfm_model = "JasonStatham320.dfm".to_owned();
    spec.params.dfm_morph = f32::NAN;
    spec.models.insert(
        ModelRole::Dfm,
        artifact("JasonStatham320", "JasonStatham320.dfm", SHA_A),
    );
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidFloatControl { control, .. }) if control == "dfm_morph"
    ));

    spec.params.dfm_morph = 0.5;
    spec.params.dfm_model = "Other.dfm".to_owned();
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::DfmModelMismatch { .. })
    ));
}

#[test]
fn enabled_enhancer_requires_the_selected_model_artifact() {
    let mut spec = valid_spec(false);
    spec.params.enhancer_enabled = true;
    spec.params.enhancer_model = EnhancerModel::UltraMixX4;

    assert_eq!(
        spec.validate(),
        Err(EngineSpecError::MissingModel(ModelRole::Enhancer))
    );

    spec.models.insert(
        ModelRole::Enhancer,
        artifact("RealEsrganx2Plus", "RealESRGAN_x2plus.fp16.onnx", SHA_A),
    );
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::EnhancerModelMismatch { .. })
    ));
}

#[test]
fn enhancer_selection_and_blend_are_generation_identity() {
    let mut x2 = valid_spec(false);
    x2.params.enhancer_enabled = true;
    x2.models.insert(
        ModelRole::Enhancer,
        artifact("RealEsrganx2Plus", "RealESRGAN_x2plus.fp16.onnx", SHA_A),
    );
    let mut blended = x2.clone();
    blended.params.enhancer_blend = 0.5;

    assert_ne!(
        x2.generation_digest().unwrap(),
        blended.generation_digest().unwrap()
    );
}

#[test]
fn learned_mask_controls_are_validated_before_gpu_allocation() {
    let mut spec = valid_spec(false);
    spec.params.occluder_size = 101;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidMaskControl { control, .. })
            if control == "occluder_size"
    ));

    spec.params.occluder_size = 0;
    spec.params.xseg_size = -101;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidMaskControl { control, .. })
            if control == "xseg_size"
    ));

    spec.params.xseg_size = 0;
    spec.params.occluder_xseg_blur = 101;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidMaskControl { control, .. })
            if control == "occluder_xseg_blur"
    ));
}

#[test]
fn detector_controls_are_generation_validated() {
    let mut spec = valid_spec(false);
    spec.params.detector_score = 0.0;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidFloatControl { control, .. }) if control == "detector_score"
    ));

    spec.params.detector_score = 0.5;
    spec.params.max_faces = 51;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidMaskControl { control, .. }) if control == "max_faces"
    ));

    spec.params.max_faces = 20;
    spec.params.manual_rotation_angle = 45;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidMaskControl { control, .. })
            if control == "manual_rotation_angle"
    ));
}

#[test]
fn strength_and_similarity_are_generation_validated() {
    let mut spec = valid_spec(false);
    spec.params.strength = 5.01;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidFloatControl { control, .. }) if control == "strength"
    ));

    spec.params.strength = 1.0;
    spec.params.similarity_threshold = -0.01;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidFloatControl { control, .. }) if control == "similarity_threshold"
    ));

    spec.params.similarity_threshold = 0.6;
    spec.params.face_likeness_enabled = true;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidAssignments(_))
    ));
}

#[test]
fn auto_color_blend_is_generation_validated() {
    let mut spec = valid_spec(false);
    spec.params.auto_color.enabled = true;
    spec.params.auto_color.blend = 1.01;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidFloatControl { control, .. })
            if control == "auto_color.blend"
    ));

    spec.params.auto_color.blend = 0.8;
    spec.validate().expect("valid auto-color generation");

    spec.params.color_adjust.hue = 0.51;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidFloatControl { control, .. })
            if control == "color_adjust.hue"
    ));

    spec.params.color_adjust.hue = 0.0;
    spec.params.jpeg_quality = 0;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidMaskControl { control, .. }) if control == "jpeg_quality"
    ));
}

#[test]
fn face_parser_controls_are_validated_before_gpu_allocation() {
    let mut spec = valid_spec(false);
    spec.params.faceparser.background = 51;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidMaskControl { control, .. })
            if control == "faceparser.background"
    ));

    spec.params.faceparser.background = 0;
    spec.params.faceparser.left_eye = 31;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidMaskControl { control, .. })
            if control == "faceparser.left_eye"
    ));

    spec.params.faceparser.left_eye = 0;
    spec.params.faceparser.face_blur = 101;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidMaskControl { control, .. })
            if control == "faceparser.face_blur"
    ));
}

#[test]
fn restore_region_controls_are_validated_before_gpu_allocation() {
    let mut spec = valid_spec(false);
    spec.params.restore_eyes_params.blend = f32::NAN;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidFloatControl { control, .. })
            if control == "restore_eyes.blend"
    ));

    spec.params.restore_eyes_params.blend = 0.5;
    spec.params.restore_mouth_params.size_factor = 0.61;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidFloatControl { control, .. })
            if control == "restore_mouth.size_factor"
    ));

    spec.params.restore_mouth_params.size_factor = 0.25;
    spec.params.restore_eyes_params.radius_x = 0.29;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidFloatControl { control, .. })
            if control == "restore_eyes.radius_x"
    ));

    spec.params.restore_eyes_params.radius_x = 1.0;
    spec.params.restore_mouth_params.offset_y = 301;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidMaskControl { control, .. })
            if control == "restore_mouth.offset_y"
    ));

    spec.params.restore_mouth_params.offset_y = 0;
    spec.params.restore_eyes_params.spacing_offset = -201;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidMaskControl { control, .. })
            if control == "restore_eyes.spacing_offset"
    ));

    spec.params.restore_eyes_params.spacing_offset = 0;
    spec.params.restore_eyes_mouth_blur = 51;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidMaskControl { control, .. })
            if control == "restore_eyes_mouth_blur"
    ));
}

#[test]
fn differencing_controls_are_validated_before_gpu_allocation() {
    let mut spec = valid_spec(false);
    spec.params.differencing_amount = 101;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidMaskControl { control, .. })
            if control == "differencing_amount"
    ));

    spec.params.differencing_amount = 4;
    spec.params.differencing_blur = 101;
    assert!(matches!(
        spec.validate(),
        Err(EngineSpecError::InvalidMaskControl { control, .. })
            if control == "differencing_blur"
    ));
}

#[test]
fn model_filenames_cannot_escape_the_generation_root() {
    let mut absolute = valid_spec(false);
    absolute
        .models
        .get_mut(&ModelRole::Detector)
        .unwrap()
        .filename = "/tmp/detector.onnx".to_owned();
    assert!(matches!(
        absolute.validate(),
        Err(EngineSpecError::InvalidFilename { .. })
    ));

    let mut parent = valid_spec(false);
    parent
        .models
        .get_mut(&ModelRole::Detector)
        .unwrap()
        .filename = "../detector.onnx".to_owned();
    assert!(matches!(
        parent.validate(),
        Err(EngineSpecError::InvalidFilename { .. })
    ));
}
