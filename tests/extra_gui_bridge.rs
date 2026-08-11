use std::collections::BTreeMap;
use std::fs;

use noperson::config::parameters::{
    AutoColorMode, EnhancerModel, LandmarkMode, RestorerAlignment, RestorerSize, SimilarityType,
    SwapDim, SwapperModel,
};
use noperson::config::settings::{DetectorModel, ExecutionProvider};
use noperson::extra_gui::{
    ARC_FACE_MODEL, ControlState, ControlValue, EditorRuntimeConfig, EmbeddingMergeMethod,
    FaceWorkspace, control_catalog,
};
use noperson::live::FaceIdentityInput;

fn configured(changes: &[(&str, ControlValue)]) -> EditorRuntimeConfig {
    let catalog = control_catalog().unwrap();
    let mut controls = ControlState::from_catalog(&catalog).unwrap();
    for (id, value) in changes {
        controls.set(id, value.clone(), &catalog).unwrap();
    }
    EditorRuntimeConfig::from_controls(&controls).unwrap()
}

#[test]
fn defaults_bridge_to_gpu_only_crossswap_runtime() {
    let runtime = configured(&[]);
    assert_eq!(runtime.provider, ExecutionProvider::Cuda);
    assert_eq!(runtime.detector, DetectorModel::YoloFace8n);
    assert_eq!(runtime.worker_threads, 2);
    assert_eq!(runtime.playback_fps, None);
    assert_eq!(runtime.params.dim, SwapDim::Dim1);
    assert_eq!(runtime.params.strength, 1.0);
    assert_eq!(runtime.params.similarity_threshold, 0.6);
    assert!(!runtime.params.restorer_enabled);
}

#[test]
fn model_detection_and_enhancement_controls_bridge_without_loss() {
    let runtime = configured(&[
        (
            "ProvidersPrioritySelection",
            ControlValue::Choice("TensorRT".into()),
        ),
        (
            "DetectorModelSelection",
            ControlValue::Choice("SCRFD".into()),
        ),
        ("SwapperResSelection", ControlValue::Choice("512".into())),
        ("LandmarkDetectToggle", ControlValue::Toggle(true)),
        (
            "LandmarkDetectModelSelection",
            ControlValue::Choice("478".into()),
        ),
        ("FrameEnhancerEnableToggle", ControlValue::Toggle(true)),
        (
            "FrameEnhancerTypeSelection",
            ControlValue::Choice("UltraMix-x4".into()),
        ),
        ("FrameEnhancerBlendSlider", ControlValue::Slider(63.0)),
        (
            "SimilarityTypeSelection",
            ControlValue::Choice("Pearl".into()),
        ),
    ]);
    assert_eq!(runtime.provider, ExecutionProvider::TensorRt);
    assert_eq!(runtime.detector, DetectorModel::Scrfd2_5g);
    assert_eq!(runtime.params.dim, SwapDim::Dim4);
    assert_eq!(runtime.params.landmark_mode, LandmarkMode::Points478);
    assert_eq!(runtime.params.enhancer_model, EnhancerModel::UltraMixX4);
    assert_eq!(runtime.params.enhancer_blend, 0.63);
    assert_eq!(runtime.params.similarity_type, SimilarityType::Pearl);
}

#[test]
fn masks_restorers_geometry_and_color_bridge_to_typed_params() {
    let runtime = configured(&[
        ("FaceRestorerEnableToggle", ControlValue::Toggle(true)),
        (
            "FaceRestorerTypeSelection",
            ControlValue::Choice("GPEN-512".into()),
        ),
        (
            "FaceRestorerDetTypeSelection",
            ControlValue::Choice("Reference".into()),
        ),
        ("FaceRestorerBlendSlider", ControlValue::Slider(72.0)),
        ("FaceAdjEnableToggle", ControlValue::Toggle(true)),
        ("KpsXSlider", ControlValue::Slider(12.0)),
        ("FaceParserEnableToggle", ControlValue::Toggle(true)),
        ("HairParserSlider", ControlValue::Slider(19.0)),
        ("AutoColorEnableToggle", ControlValue::Toggle(true)),
        (
            "AutoColorTransferTypeSelection",
            ControlValue::Choice("DFL_Orig".into()),
        ),
        ("AutoColorBlendAmountSlider", ControlValue::Slider(55.0)),
        ("ColorEnableToggle", ControlValue::Toggle(true)),
        ("ColorGammaDecimalSlider", ControlValue::Slider(1.25)),
    ]);
    let params = runtime.params;
    assert!(params.restorer_enabled);
    assert_eq!(params.restorer_size, RestorerSize::Gpen512);
    assert_eq!(params.restorer_alignment, RestorerAlignment::Reference);
    assert_eq!(params.restorer_alpha, 0.72);
    assert_eq!(params.geometry.keypoints_x, 12.0);
    assert_eq!(params.faceparser.hair, 19);
    assert_eq!(params.auto_color.mode, AutoColorMode::DflMasked);
    assert_eq!(params.auto_color.blend, 0.55);
    assert_eq!(params.color_adjust.gamma, 1.25);
}

#[test]
fn dfm_requires_a_concrete_model_while_inswapper_does_not() {
    let catalog = control_catalog().unwrap();
    let mut controls = ControlState::from_catalog(&catalog).unwrap();
    controls
        .set(
            "SwapModelSelection",
            ControlValue::Choice("DeepFaceLive (DFM)".into()),
            &catalog,
        )
        .unwrap();
    assert!(EditorRuntimeConfig::from_controls(&controls).is_err());
    controls
        .set(
            "DFMModelSelection",
            ControlValue::Choice("JasonStatham320.dfm".into()),
            &catalog,
        )
        .unwrap();
    let runtime = EditorRuntimeConfig::from_controls(&controls).unwrap();
    assert_eq!(runtime.params.swapper_model, SwapperModel::Dfm);
    assert_eq!(runtime.params.dfm_model, "JasonStatham320.dfm");
}

#[test]
fn face_workspace_compiles_to_precomputed_atomic_assignments() {
    let catalog = control_catalog().unwrap();
    let controls = ControlState::from_catalog(&catalog).unwrap();
    let runtime = EditorRuntimeConfig::from_controls(&controls).unwrap();
    let store = |value: f32| BTreeMap::from([(ARC_FACE_MODEL.to_owned(), vec![value; 512])]);
    let mut faces = FaceWorkspace::default();
    faces
        .add_source("source", "Source", None, store(0.25))
        .unwrap();
    faces
        .add_target("target", None, store(0.75), &catalog)
        .unwrap();
    faces.assign_source("target", "source", true).unwrap();
    faces
        .target_mut("target")
        .unwrap()
        .controls
        .set(
            "SimilarityThresholdSlider",
            ControlValue::Slider(73.0),
            &catalog,
        )
        .unwrap();

    let assignments = runtime
        .assignment_inputs(&faces, EmbeddingMergeMethod::Mean)
        .unwrap();
    assert_eq!(assignments.len(), 1);
    assert_eq!(assignments[0].similarity_threshold, 0.73);
    assert!(matches!(
        assignments[0].source,
        FaceIdentityInput::Embedding(ref values) if values == &vec![0.25; 512]
    ));
    assert!(matches!(
        assignments[0].target,
        Some(FaceIdentityInput::Embedding(ref values)) if values == &vec![0.75; 512]
    ));
}

#[test]
fn editor_request_contains_content_addressed_models_and_atomic_inputs() {
    let catalog = control_catalog().unwrap();
    let controls = ControlState::from_catalog(&catalog).unwrap();
    let runtime = EditorRuntimeConfig::from_controls(&controls).unwrap();
    let store = |value: f32| BTreeMap::from([(ARC_FACE_MODEL.to_owned(), vec![value; 512])]);
    let mut faces = FaceWorkspace::default();
    faces
        .add_source("source", "Source", None, store(0.25))
        .unwrap();
    faces
        .add_target("target", None, store(0.75), &catalog)
        .unwrap();
    faces.assign_source("target", "source", true).unwrap();

    let models = tempfile::tempdir().unwrap();
    for filename in [
        "yoloface_8n.onnx",
        "w600k_r50.onnx",
        "inswapper_128.fp16.onnx",
        "emap.bin",
    ] {
        fs::write(models.path().join(filename), filename).unwrap();
    }
    let request = runtime
        .compile_engine_request(&faces, EmbeddingMergeMethod::Mean, models.path(), 0)
        .unwrap();
    assert_eq!(request.worker_threads, runtime.worker_threads);
    assert_eq!(request.assignments.len(), 1);
    assert!(
        request
            .spec
            .models
            .contains_key(&noperson::engine::ModelRole::Detector)
    );
    assert_eq!(request.spec.assignments.len(), 0);
}
