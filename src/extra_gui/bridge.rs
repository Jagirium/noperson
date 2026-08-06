use std::collections::BTreeMap;
use std::path::Path;

use thiserror::Error;

use crate::config::parameters::{
    AutoColorMode, EnhancerModel, FaceSwapParams, LandmarkMode, RestorerAlignment, RestorerSize,
    SimilarityType, SwapDim, SwapperModel,
};
use crate::config::settings::{DetectorModel, ExecutionProvider};
use crate::engine::{EngineSpec, ModelRole};
use crate::launch::PredecodeMode;
use crate::live::{
    FaceAssignmentInputs, FaceIdentityInput, build_live_spec_for_digest, embedding_blake3,
};

use super::controls::{ControlState, ControlValue};
use super::faces::{ARC_FACE_MODEL, EmbeddingMergeMethod, FaceWorkspace, FaceWorkspaceError};

#[derive(Debug, Clone, PartialEq)]
pub struct EditorRuntimeConfig {
    pub provider: ExecutionProvider,
    pub detector: DetectorModel,
    pub worker_threads: usize,
    pub playback_fps: Option<f32>,
    pub auto_swap: bool,
    pub show_landmarks: bool,
    pub show_bounding_boxes: bool,
    pub max_dfm_models: usize,
    pub similarity_type: SimilarityType,
    pub embedding_merge_method: String,
    pub target_media_recursive: bool,
    pub source_media_recursive: bool,
    pub scroll_changes_values: bool,
    pub params: FaceSwapParams,
}

#[derive(Debug, Clone)]
pub struct EditorEngineRequest {
    pub spec: EngineSpec,
    pub assignments: Vec<FaceAssignmentInputs>,
    /// CPU threads granted to FFmpeg decode/encode. The GPU pipeline remains
    /// a single ordered worker so models and VRAM are never duplicated.
    pub worker_threads: usize,
    pub predecode: PredecodeMode,
}

impl EditorRuntimeConfig {
    pub fn from_controls(controls: &ControlState) -> Result<Self, ControlBridgeError> {
        let provider = match choice(controls, "ProvidersPrioritySelection")? {
            "CUDA" => ExecutionProvider::Cuda,
            "TensorRT" => ExecutionProvider::TensorRT,
            value => return Err(unknown_choice("ProvidersPrioritySelection", value)),
        };
        let detector = match choice(controls, "DetectorModelSelection")? {
            "Yolov8" => DetectorModel::YoloFace8n,
            "RetinaFace" => DetectorModel::RetinaFace,
            "SCRFD" => DetectorModel::Scrfd2_5g,
            value => return Err(unknown_choice("DetectorModelSelection", value)),
        };
        let playback_fps = toggle(controls, "VideoPlaybackCustomFpsToggle")?
            .then(|| number(controls, "VideoPlaybackCustomFpsSlider").map(|value| value as f32))
            .transpose()?;

        let mut params = FaceSwapParams::default();
        params.detector_score = percent(controls, "DetectorScoreSlider")?;
        params.max_faces = number(controls, "MaxFacesToDetectSlider")? as usize;
        params.auto_rotation = toggle(controls, "AutoRotationToggle")?;
        params.manual_rotation_enabled = toggle(controls, "ManualRotationEnableToggle")?;
        params.manual_rotation_angle = number(controls, "ManualRotationAngleSlider")? as u16;
        params.landmark_enabled = toggle(controls, "LandmarkDetectToggle")?;
        params.landmark_mode = match choice(controls, "LandmarkDetectModelSelection")? {
            "5" => LandmarkMode::Points5,
            "68" => LandmarkMode::Points68,
            "3d68" => LandmarkMode::Points3d68,
            "98" => LandmarkMode::Points98,
            "106" => LandmarkMode::Points106,
            "203" => LandmarkMode::Points203,
            "478" => LandmarkMode::Points478,
            value => return Err(unknown_choice("LandmarkDetectModelSelection", value)),
        };
        params.landmark_score = percent(controls, "LandmarkDetectScoreSlider")?;
        params.landmark_from_points = toggle(controls, "DetectFromPointsToggle")?;

        params.enhancer_enabled = toggle(controls, "FrameEnhancerEnableToggle")?;
        params.enhancer_model =
            EnhancerModel::from_crosswap_name(choice(controls, "FrameEnhancerTypeSelection")?)
                .map_err(|error| ControlBridgeError::InvalidChoice {
                    control: "FrameEnhancerTypeSelection",
                    value: error.to_string(),
                })?;
        params.enhancer_blend = percent(controls, "FrameEnhancerBlendSlider")?;

        params.restorer_enabled = toggle(controls, "FaceRestorerEnableToggle")?;
        params.restorer_size = restorer_size(choice(controls, "FaceRestorerTypeSelection")?)?;
        params.restorer_alignment =
            restorer_alignment(choice(controls, "FaceRestorerDetTypeSelection")?)?;
        params.restorer_alpha = percent(controls, "FaceRestorerBlendSlider")?;
        params.restorer_fidelity = number(controls, "FaceFidelityWeightDecimalSlider")? as f32;
        params.restorer2_enabled = toggle(controls, "FaceRestorerEnable2Toggle")?;
        params.restorer2_size = restorer_size(choice(controls, "FaceRestorerType2Selection")?)?;
        params.restorer2_alignment =
            restorer_alignment(choice(controls, "FaceRestorerDetType2Selection")?)?;
        params.restorer2_alpha = percent(controls, "FaceRestorerBlend2Slider")?;
        params.restorer2_fidelity = number(controls, "FaceFidelityWeight2DecimalSlider")? as f32;

        params.swapper_model = match choice(controls, "SwapModelSelection")? {
            "Inswapper128" => SwapperModel::Inswapper128,
            "DeepFaceLive (DFM)" => SwapperModel::Dfm,
            value => return Err(unknown_choice("SwapModelSelection", value)),
        };
        params.dim = match choice(controls, "SwapperResSelection")? {
            "128" => SwapDim::Dim1,
            "256" => SwapDim::Dim2,
            "384" => SwapDim::Dim3,
            "512" => SwapDim::Dim4,
            value => return Err(unknown_choice("SwapperResSelection", value)),
        };
        if params.swapper_model == SwapperModel::Dfm {
            let model = choice(controls, "DFMModelSelection")?;
            if model.is_empty() {
                return Err(ControlBridgeError::MissingDfmModel);
            }
            params.dfm_model = model.to_owned();
        }
        params.dfm_morph = percent(controls, "DFMAmpMorphSlider")?;
        params.dfm_rct = toggle(controls, "DFMRCTColorToggle")?;

        params.geometry.enabled = toggle(controls, "FaceAdjEnableToggle")?;
        params.geometry.keypoints_x = number(controls, "KpsXSlider")? as f32;
        params.geometry.keypoints_y = number(controls, "KpsYSlider")? as f32;
        params.geometry.keypoints_scale = number(controls, "KpsScaleSlider")? as f32;
        params.geometry.face_scale = number(controls, "FaceScaleAmountSlider")? as f32;
        params.geometry.landmark_offsets_enabled =
            toggle(controls, "LandmarksPositionAdjEnableToggle")?;
        for (index, (x, y)) in [
            ("EyeLeftXAmountSlider", "EyeLeftYAmountSlider"),
            ("EyeRightXAmountSlider", "EyeRightYAmountSlider"),
            ("NoseXAmountSlider", "NoseYAmountSlider"),
            ("MouthLeftXAmountSlider", "MouthLeftYAmountSlider"),
            ("MouthRightXAmountSlider", "MouthRightYAmountSlider"),
        ]
        .into_iter()
        .enumerate()
        {
            params.geometry.landmark_offsets[index] =
                [number(controls, x)? as f32, number(controls, y)? as f32];
        }

        params.similarity_threshold = percent(controls, "SimilarityThresholdSlider")?;
        params.strength = if toggle(controls, "StrengthEnableToggle")? {
            percent(controls, "StrengthAmountSlider")?
        } else {
            1.0
        };
        params.face_likeness_enabled = toggle(controls, "FaceLikenessEnableToggle")?;
        params.face_likeness_factor = number(controls, "FaceLikenessFactorDecimalSlider")? as f32;
        params.differencing_enabled = toggle(controls, "DifferencingEnableToggle")?;
        params.differencing_amount = number(controls, "DifferencingAmountSlider")? as u32;
        params.differencing_blur = number(controls, "DifferencingBlendAmountSlider")? as u32;

        params.border_bottom = number(controls, "BorderBottomSlider")? as u32;
        params.border_left = number(controls, "BorderLeftSlider")? as u32;
        params.border_right = number(controls, "BorderRightSlider")? as u32;
        params.border_top = number(controls, "BorderTopSlider")? as u32;
        params.border_blur = number(controls, "BorderBlurSlider")? as u32;
        params.occluder_enabled = toggle(controls, "OccluderEnableToggle")?;
        params.occluder_size = number(controls, "OccluderSizeSlider")? as i32;
        params.xseg_enabled = toggle(controls, "DFLXSegEnableToggle")?;
        params.xseg_size = number(controls, "DFLXSegSizeSlider")? as i32;
        params.occluder_xseg_blur = number(controls, "OccluderXSegBlurSlider")? as u32;
        params.faceparser_enabled = toggle(controls, "FaceParserEnableToggle")?;
        params.faceparser.background = number(controls, "BackgroundParserSlider")? as i32;
        for (target, id) in [
            (&mut params.faceparser.face, "FaceParserSlider"),
            (
                &mut params.faceparser.left_eyebrow,
                "LeftEyebrowParserSlider",
            ),
            (
                &mut params.faceparser.right_eyebrow,
                "RightEyebrowParserSlider",
            ),
            (&mut params.faceparser.left_eye, "LeftEyeParserSlider"),
            (&mut params.faceparser.right_eye, "RightEyeParserSlider"),
            (&mut params.faceparser.eyeglasses, "EyeGlassesParserSlider"),
            (&mut params.faceparser.nose, "NoseParserSlider"),
            (&mut params.faceparser.mouth, "MouthParserSlider"),
            (&mut params.faceparser.upper_lip, "UpperLipParserSlider"),
            (&mut params.faceparser.lower_lip, "LowerLipParserSlider"),
            (&mut params.faceparser.neck, "NeckParserSlider"),
            (&mut params.faceparser.hair, "HairParserSlider"),
            (
                &mut params.faceparser.background_blur,
                "BackgroundBlurParserSlider",
            ),
            (&mut params.faceparser.face_blur, "FaceBlurParserSlider"),
        ] {
            *target = number(controls, id)? as u32;
        }
        params.faceparser.hair_makeup.enabled =
            toggle(controls, "FaceParserHairMakeupEnableToggle")?;
        params.faceparser.hair_makeup.color = [
            number(controls, "FaceParserHairMakeupRedSlider")? as f32,
            number(controls, "FaceParserHairMakeupGreenSlider")? as f32,
            number(controls, "FaceParserHairMakeupBlueSlider")? as f32,
        ];
        params.faceparser.hair_makeup.blend =
            number(controls, "FaceParserHairMakeupBlendAmountDecimalSlider")? as f32;
        params.faceparser.lips_makeup.enabled =
            toggle(controls, "FaceParserLipsMakeupEnableToggle")?;
        params.faceparser.lips_makeup.color = [
            number(controls, "FaceParserLipsMakeupRedSlider")? as f32,
            number(controls, "FaceParserLipsMakeupGreenSlider")? as f32,
            number(controls, "FaceParserLipsMakeupBlueSlider")? as f32,
        ];
        params.faceparser.lips_makeup.blend =
            number(controls, "FaceParserLipsMakeupBlendAmountDecimalSlider")? as f32;

        params.restore_eyes = toggle(controls, "RestoreEyesEnableToggle")?;
        params.restore_eyes_params.blend = percent(controls, "RestoreEyesBlendAmountSlider")?;
        params.restore_eyes_params.size_factor =
            number(controls, "RestoreEyesSizeFactorDecimalSlider")? as f32;
        params.restore_eyes_params.feather =
            number(controls, "RestoreEyesFeatherBlendSlider")? as u32;
        params.restore_eyes_params.radius_x =
            number(controls, "RestoreXEyesRadiusFactorDecimalSlider")? as f32;
        params.restore_eyes_params.radius_y =
            number(controls, "RestoreYEyesRadiusFactorDecimalSlider")? as f32;
        params.restore_eyes_params.offset_x = number(controls, "RestoreXEyesOffsetSlider")? as i32;
        params.restore_eyes_params.offset_y = number(controls, "RestoreYEyesOffsetSlider")? as i32;
        params.restore_eyes_params.spacing_offset =
            number(controls, "RestoreEyesSpacingOffsetSlider")? as i32;
        params.restore_mouth = toggle(controls, "RestoreMouthEnableToggle")?;
        params.restore_mouth_params.blend = percent(controls, "RestoreMouthBlendAmountSlider")?;
        params.restore_mouth_params.size_factor =
            percent(controls, "RestoreMouthSizeFactorSlider")?;
        params.restore_mouth_params.feather =
            number(controls, "RestoreMouthFeatherBlendSlider")? as u32;
        params.restore_mouth_params.radius_x =
            number(controls, "RestoreXMouthRadiusFactorDecimalSlider")? as f32;
        params.restore_mouth_params.radius_y =
            number(controls, "RestoreYMouthRadiusFactorDecimalSlider")? as f32;
        params.restore_mouth_params.offset_x =
            number(controls, "RestoreXMouthOffsetSlider")? as i32;
        params.restore_mouth_params.offset_y =
            number(controls, "RestoreYMouthOffsetSlider")? as i32;
        params.restore_eyes_mouth_blur = number(controls, "RestoreEyesMouthBlurSlider")? as u32;

        params.auto_color.enabled = toggle(controls, "AutoColorEnableToggle")?;
        params.auto_color.mode = match choice(controls, "AutoColorTransferTypeSelection")? {
            "Test" => AutoColorMode::Histogram,
            "Test_Mask" => AutoColorMode::HistogramMasked,
            "DFL_Test" => AutoColorMode::Dfl,
            "DFL_Orig" => AutoColorMode::DflMasked,
            value => return Err(unknown_choice("AutoColorTransferTypeSelection", value)),
        };
        params.auto_color.blend = percent(controls, "AutoColorBlendAmountSlider")?;
        params.color_adjust.enabled = toggle(controls, "ColorEnableToggle")?;
        params.color_adjust.red = number(controls, "ColorRedSlider")? as f32;
        params.color_adjust.green = number(controls, "ColorGreenSlider")? as f32;
        params.color_adjust.blue = number(controls, "ColorBlueSlider")? as f32;
        params.color_adjust.brightness = number(controls, "ColorBrightnessDecimalSlider")? as f32;
        params.color_adjust.contrast = number(controls, "ColorContrastDecimalSlider")? as f32;
        params.color_adjust.saturation = number(controls, "ColorSaturationDecimalSlider")? as f32;
        params.color_adjust.sharpness = number(controls, "ColorSharpnessDecimalSlider")? as f32;
        params.color_adjust.hue = number(controls, "ColorHueDecimalSlider")? as f32;
        params.color_adjust.gamma = number(controls, "ColorGammaDecimalSlider")? as f32;
        params.color_adjust.noise = number(controls, "ColorNoiseDecimalSlider")? as f32;
        params.jpeg_compression_enabled = toggle(controls, "JPEGCompressionEnableToggle")?;
        params.jpeg_quality = number(controls, "JPEGCompressionAmountSlider")? as u8;
        params.final_blur_enabled = toggle(controls, "FinalBlendAdjEnableToggle")?;
        params.final_blur = number(controls, "FinalBlendAmountSlider")? as u32;
        params.overall_mask_blur = number(controls, "OverallMaskBlendAmountSlider")? as u32;

        let similarity_type =
            SimilarityType::from_crossswap_name(choice(controls, "SimilarityTypeSelection")?)
                .map_err(|error| ControlBridgeError::InvalidChoice {
                    control: "SimilarityTypeSelection",
                    value: error.to_string(),
                })?;
        params.similarity_type = similarity_type;

        Ok(Self {
            provider,
            detector,
            worker_threads: number(controls, "nThreadsSlider")? as usize,
            playback_fps,
            auto_swap: toggle(controls, "AutoSwapToggle")?,
            show_landmarks: toggle(controls, "ShowLandmarksEnableToggle")?,
            show_bounding_boxes: toggle(controls, "ShowAllDetectedFacesBBoxToggle")?,
            max_dfm_models: number(controls, "MaxDFMModelsSlider")? as usize,
            similarity_type,
            embedding_merge_method: choice(controls, "EmbMergeMethodSelection")?.to_owned(),
            target_media_recursive: toggle(controls, "TargetMediaFolderRecursiveToggle")?,
            source_media_recursive: toggle(controls, "InputFacesFolderRecursiveToggle")?,
            scroll_changes_values: toggle(controls, "ScrollChangesValuesToggle")?,
            params,
        })
    }

    pub fn assignment_inputs(
        &self,
        faces: &FaceWorkspace,
        merge_method: EmbeddingMergeMethod,
    ) -> Result<Vec<FaceAssignmentInputs>, ControlBridgeError> {
        if faces.swap_all() {
            if !faces.has_assignment(None) {
                return Err(ControlBridgeError::NoFaceAssignments);
            }
            let source = faces.swap_all_embedding(ARC_FACE_MODEL, merge_method)?;
            embedding_blake3(&source).map_err(ControlBridgeError::InvalidEmbedding)?;
            return Ok(vec![FaceAssignmentInputs {
                source: FaceIdentityInput::Embedding(source),
                target: None,
                similarity_threshold: self.params.similarity_threshold,
                params: Some(self.params.clone()),
                models: BTreeMap::new(),
            }]);
        }

        let mut assignments = Vec::new();
        for target in faces.targets() {
            if !faces.has_assignment(Some(&target.id)) {
                continue;
            }
            let source = faces.assigned_embedding(&target.id, ARC_FACE_MODEL, merge_method)?;
            let target_embedding = target
                .embeddings
                .get(ARC_FACE_MODEL)
                .cloned()
                .ok_or_else(|| ControlBridgeError::MissingTargetEmbedding(target.id.clone()))?;
            embedding_blake3(&source).map_err(ControlBridgeError::InvalidEmbedding)?;
            embedding_blake3(&target_embedding).map_err(ControlBridgeError::InvalidEmbedding)?;

            let mut params = Self::from_controls(&target.controls)?.params;
            copy_generation_controls(&self.params, &mut params);
            assignments.push(FaceAssignmentInputs {
                source: FaceIdentityInput::Embedding(source),
                target: Some(FaceIdentityInput::Embedding(target_embedding)),
                similarity_threshold: params.similarity_threshold,
                params: Some(params),
                models: BTreeMap::new(),
            });
        }
        if assignments.is_empty() {
            return Err(ControlBridgeError::NoFaceAssignments);
        }
        Ok(assignments)
    }

    pub fn compile_engine_request(
        &self,
        faces: &FaceWorkspace,
        merge_method: EmbeddingMergeMethod,
        models_dir: &Path,
        device_id: i32,
    ) -> Result<EditorEngineRequest, ControlBridgeError> {
        let mut assignments = self.assignment_inputs(faces, merge_method)?;
        let primary_digest = assignment_source_digest(
            assignments
                .first()
                .expect("assignment_inputs rejects an empty route"),
        )?;
        let spec = build_live_spec_for_digest(
            models_dir,
            primary_digest,
            self.params.clone(),
            self.provider,
            self.detector,
            device_id,
        )
        .map_err(ControlBridgeError::EngineRequest)?;

        for assignment in &mut assignments {
            let params = assignment.params.as_ref().unwrap_or(&self.params);
            let mut local = build_live_spec_for_digest(
                models_dir,
                assignment_source_digest(assignment)?,
                params.clone(),
                self.provider,
                self.detector,
                device_id,
            )
            .map_err(ControlBridgeError::EngineRequest)?;
            local
                .models
                .remove(&ModelRole::Detector)
                .expect("every live spec owns its detector");
            local.models.remove(&ModelRole::Enhancer);
            local
                .models
                .retain(|role, artifact| spec.models.get(role) != Some(artifact));
            assignment.models = local.models;
        }
        let dfm_models = spec
            .models
            .get(&ModelRole::Dfm)
            .into_iter()
            .chain(
                assignments
                    .iter()
                    .filter_map(|assignment| assignment.models.get(&ModelRole::Dfm)),
            )
            .map(|artifact| artifact.blake3.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        if dfm_models.len() > self.max_dfm_models {
            return Err(ControlBridgeError::TooManyDfmModels {
                requested: dfm_models.len(),
                maximum: self.max_dfm_models,
            });
        }
        Ok(EditorEngineRequest {
            spec,
            assignments,
            worker_threads: self.worker_threads.clamp(1, 32),
            predecode: PredecodeMode::Auto,
        })
    }
}

fn assignment_source_digest(
    assignment: &FaceAssignmentInputs,
) -> Result<String, ControlBridgeError> {
    match &assignment.source {
        FaceIdentityInput::Embedding(embedding) => {
            embedding_blake3(embedding).map_err(ControlBridgeError::InvalidEmbedding)
        }
        FaceIdentityInput::Image(path) => crate::models::digest::file_blake3(path)
            .map_err(|error| ControlBridgeError::EngineRequest(error.into())),
    }
}

fn copy_generation_controls(source: &FaceSwapParams, target: &mut FaceSwapParams) {
    target.detector_score = source.detector_score;
    target.max_faces = source.max_faces;
    target.auto_rotation = source.auto_rotation;
    target.manual_rotation_enabled = source.manual_rotation_enabled;
    target.manual_rotation_angle = source.manual_rotation_angle;
    target.enhancer_enabled = source.enhancer_enabled;
    target.enhancer_model = source.enhancer_model;
    target.enhancer_blend = source.enhancer_blend;
}

fn toggle(controls: &ControlState, id: &'static str) -> Result<bool, ControlBridgeError> {
    match controls.get(id) {
        Some(ControlValue::Toggle(value)) => Ok(*value),
        _ => Err(ControlBridgeError::MissingOrWrongType(id)),
    }
}

fn number(controls: &ControlState, id: &'static str) -> Result<f64, ControlBridgeError> {
    match controls.get(id) {
        Some(ControlValue::Slider(value)) => Ok(*value),
        _ => Err(ControlBridgeError::MissingOrWrongType(id)),
    }
}

fn percent(controls: &ControlState, id: &'static str) -> Result<f32, ControlBridgeError> {
    Ok((number(controls, id)? / 100.0) as f32)
}

fn choice<'a>(controls: &'a ControlState, id: &'static str) -> Result<&'a str, ControlBridgeError> {
    match controls.get(id) {
        Some(ControlValue::Choice(value)) => Ok(value),
        _ => Err(ControlBridgeError::MissingOrWrongType(id)),
    }
}

fn restorer_size(value: &str) -> Result<RestorerSize, ControlBridgeError> {
    match value {
        "GPEN-256" => Ok(RestorerSize::Gpen256),
        "GPEN-512" => Ok(RestorerSize::Gpen512),
        value => Err(unknown_choice("FaceRestorerTypeSelection", value)),
    }
}

fn restorer_alignment(value: &str) -> Result<RestorerAlignment, ControlBridgeError> {
    match value {
        "Original" => Ok(RestorerAlignment::Original),
        "Blend" => Ok(RestorerAlignment::Blend),
        "Reference" => Ok(RestorerAlignment::Reference),
        value => Err(unknown_choice("FaceRestorerDetTypeSelection", value)),
    }
}

fn unknown_choice(control: &'static str, value: &str) -> ControlBridgeError {
    ControlBridgeError::InvalidChoice {
        control,
        value: value.to_owned(),
    }
}

#[derive(Debug, Error)]
pub enum ControlBridgeError {
    #[error("control {0} is missing or has the wrong type")]
    MissingOrWrongType(&'static str),
    #[error("control {control} has unsupported choice {value}")]
    InvalidChoice {
        control: &'static str,
        value: String,
    },
    #[error("DFM swapper requires a selected .dfm model")]
    MissingDfmModel,
    #[error("engine request uses {requested} DFM models but the configured maximum is {maximum}")]
    TooManyDfmModels { requested: usize, maximum: usize },
    #[error("no source identities are assigned to an active face route")]
    NoFaceAssignments,
    #[error("target face {0} has no ArcFace embedding")]
    MissingTargetEmbedding(String),
    #[error("invalid precomputed ArcFace embedding: {0}")]
    InvalidEmbedding(anyhow::Error),
    #[error("could not compile an immutable engine request: {0}")]
    EngineRequest(anyhow::Error),
    #[error(transparent)]
    Faces(#[from] FaceWorkspaceError),
}
