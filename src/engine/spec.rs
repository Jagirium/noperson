use std::collections::BTreeMap;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::config::parameters::{FaceSwapParams, RestorerSize, SwapperModel};
use crate::config::settings::{DetectorModel, ExecutionProvider};

/// A model's purpose inside one immutable engine generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelRole {
    Detector,
    Landmark,
    Recognizer,
    Swapper,
    Emap,
    Restorer,
    Restorer2,
    RestorerLandmark,
    Occluder,
    Xseg,
    FaceParser,
    Enhancer,
    Dfm,
}

impl ModelRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Detector => "detector",
            Self::Landmark => "landmark",
            Self::Recognizer => "recognizer",
            Self::Swapper => "swapper",
            Self::Emap => "emap",
            Self::Restorer => "restorer",
            Self::Restorer2 => "restorer-2",
            Self::RestorerLandmark => "restorer-landmark",
            Self::Occluder => "occluder",
            Self::Xseg => "xseg",
            Self::FaceParser => "face-parser",
            Self::Enhancer => "enhancer",
            Self::Dfm => "dfm",
        }
    }
}

/// One content-addressed file consumed by an engine generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub logical_name: String,
    pub filename: String,
    pub blake3: String,
}

/// One immutable source-to-target identity assignment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FaceAssignmentSpec {
    /// Identity to render into the selected target face.
    pub source_identity_blake3: String,
    /// Reference identity used to select a target. `None` explicitly means all faces.
    pub target_identity_blake3: Option<String>,
    /// CrossSwap similarity score in the normalized `[0, 1]` domain.
    pub similarity_threshold: f32,
    /// Optional target-local controls. `None` inherits generation defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<FaceSwapParams>,
    /// Content-addressed model overrides used only by this target assignment.
    /// Empty preserves the legacy generation-wide model set.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub models: BTreeMap<ModelRole, ModelArtifact>,
}

/// Complete immutable configuration for a buildable engine generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSpec {
    pub provider: ExecutionProvider,
    pub device_id: i32,
    pub detector: DetectorModel,
    pub identity_blake3: String,
    /// Empty preserves the legacy single-source, swap-all generation contract.
    #[serde(default)]
    pub assignments: Vec<FaceAssignmentSpec>,
    pub models: BTreeMap<ModelRole, ModelArtifact>,
    pub params: FaceSwapParams,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EngineSpecError {
    #[error("engine spec requires model role {0:?}")]
    MissingModel(ModelRole),
    #[error("{field} must be a lowercase BLAKE3 digest, got {value}")]
    InvalidBlake3 { field: String, value: String },
    #[error("{0} must not be empty")]
    EmptyField(String),
    #[error("model filename for {role:?} must stay inside the generation root: {filename}")]
    InvalidFilename { role: ModelRole, filename: String },
    #[error("CUDA device id must be non-negative, got {0}")]
    InvalidDeviceId(i32),
    #[error("restorer {0} is excluded from new engine generations")]
    UnsupportedRestorer(String),
    #[error("enhancer blend must be finite and in [0,1], got {0}")]
    InvalidEnhancerBlend(String),
    #[error("enhancer selection {selected} does not match artifact {artifact}")]
    EnhancerModelMismatch { selected: String, artifact: String },
    #[error("landmark selection {selected} does not match artifact {artifact}")]
    LandmarkModelMismatch { selected: String, artifact: String },
    #[error("DFM selection {selected} does not match artifact {artifact}")]
    DfmModelMismatch { selected: String, artifact: String },
    #[error("mask control {control} is outside {min}..={max}: {value}")]
    InvalidMaskControl {
        control: String,
        value: i64,
        min: i64,
        max: i64,
    },
    #[error("mask control {control} must be finite and inside {min}..={max}, got {value}")]
    InvalidFloatControl {
        control: String,
        value: String,
        min: String,
        max: String,
    },
    #[error("failed to serialize engine spec: {0}")]
    Serialization(String),
    #[error("invalid face assignments: {0}")]
    InvalidAssignments(String),
}

impl EngineSpec {
    const REQUIRED_MODELS: [ModelRole; 1] = [ModelRole::Detector];

    /// Reject incomplete or unsupported generations before allocating GPU memory.
    pub fn validate(&self) -> Result<(), EngineSpecError> {
        if self.device_id < 0 {
            return Err(EngineSpecError::InvalidDeviceId(self.device_id));
        }

        for role in Self::REQUIRED_MODELS {
            self.require(role)?;
        }
        match self.params.swapper_model {
            SwapperModel::Inswapper128 => {
                for role in [ModelRole::Recognizer, ModelRole::Swapper, ModelRole::Emap] {
                    self.require(role)?;
                }
            }
            SwapperModel::Dfm => {
                self.require(ModelRole::Dfm)?;
            }
        }
        if self.params.landmark_enabled {
            self.require(ModelRole::Landmark)?;
            let artifact = &self.models[&ModelRole::Landmark];
            let selected = self.params.landmark_mode.registry_name();
            if artifact.logical_name != selected {
                return Err(EngineSpecError::LandmarkModelMismatch {
                    selected: selected.to_owned(),
                    artifact: artifact.logical_name.clone(),
                });
            }
        }
        if (self.params.restorer_enabled
            && self.params.restorer_alignment
                == crate::config::parameters::RestorerAlignment::Reference)
            || (self.params.restorer2_enabled
                && self.params.restorer2_alignment
                    == crate::config::parameters::RestorerAlignment::Reference)
        {
            self.require(ModelRole::RestorerLandmark)?;
        }

        validate_blake3("identity_blake3", &self.identity_blake3)?;
        for (index, assignment) in self.assignments.iter().enumerate() {
            validate_blake3(
                &format!("assignments[{index}].source_identity_blake3"),
                &assignment.source_identity_blake3,
            )?;
            if let Some(target) = &assignment.target_identity_blake3 {
                validate_blake3(
                    &format!("assignments[{index}].target_identity_blake3"),
                    target,
                )?;
            }
            validate_float_range(
                &format!("assignments[{index}].similarity_threshold"),
                assignment.similarity_threshold,
                0.0,
                1.0,
            )?;
            if assignment.target_identity_blake3.is_none() && index + 1 != self.assignments.len() {
                return Err(EngineSpecError::InvalidAssignments(
                    "an unscoped swap-all assignment must be last".to_owned(),
                ));
            }
            if let Some(params) = &assignment.params {
                if !same_generation_controls(&self.params, params) {
                    return Err(EngineSpecError::InvalidAssignments(format!(
                        "assignments[{index}] overrides generation-wide detection, rotation, or enhancement controls"
                    )));
                }
                if params.face_likeness_enabled && assignment.target_identity_blake3.is_none() {
                    return Err(EngineSpecError::InvalidAssignments(format!(
                        "assignments[{index}] enables face likeness without a target identity"
                    )));
                }

                // Reuse the complete parameter validator without recursing into
                // assignments. Target presence is checked directly above.
                let mut scoped = self.clone();
                scoped.assignments.clear();
                scoped.params = params.clone();
                scoped.models.extend(assignment.models.clone());
                scoped.params.face_likeness_enabled = false;
                scoped.validate()?;
            } else if !assignment.models.is_empty() {
                return Err(EngineSpecError::InvalidAssignments(format!(
                    "assignments[{index}] provides model overrides without target-local parameters"
                )));
            }
        }
        if self
            .assignments
            .iter()
            .any(|assignment| assignment.target_identity_blake3.is_some())
            && !self.models.contains_key(&ModelRole::Recognizer)
            && !self
                .assignments
                .iter()
                .any(|assignment| assignment.models.contains_key(&ModelRole::Recognizer))
        {
            return Err(EngineSpecError::MissingModel(ModelRole::Recognizer));
        }
        let mut sessions = BTreeMap::<String, String>::new();
        register_runtime_sessions(&mut sessions, &self.params, &self.models)?;
        for assignment in &self.assignments {
            let Some(params) = &assignment.params else {
                continue;
            };
            let mut models = self.models.clone();
            models.extend(assignment.models.clone());
            register_runtime_sessions(&mut sessions, params, &models)?;
        }

        if matches!(self.params.restorer_size, RestorerSize::Gpen1024)
            || matches!(self.params.restorer2_size, RestorerSize::Gpen1024)
        {
            return Err(EngineSpecError::UnsupportedRestorer("GPEN-1024".to_owned()));
        }
        if !self.params.enhancer_blend.is_finite()
            || !(0.0..=1.0).contains(&self.params.enhancer_blend)
        {
            return Err(EngineSpecError::InvalidEnhancerBlend(
                self.params.enhancer_blend.to_string(),
            ));
        }
        validate_float_range("dfm_morph", self.params.dfm_morph, 0.01, 1.0)?;
        validate_float_range("detector_score", self.params.detector_score, 0.01, 1.0)?;
        validate_range("max_faces", self.params.max_faces as i64, 1, 50)?;
        if self.params.manual_rotation_angle > 270
            || !self.params.manual_rotation_angle.is_multiple_of(90)
        {
            return Err(EngineSpecError::InvalidMaskControl {
                control: "manual_rotation_angle".to_owned(),
                value: self.params.manual_rotation_angle as i64,
                min: 0,
                max: 270,
            });
        }
        validate_float_range("landmark_score", self.params.landmark_score, 0.01, 1.0)?;
        validate_float_range("strength", self.params.strength, 0.0, 5.0)?;
        validate_float_range(
            "face_likeness_factor",
            self.params.face_likeness_factor,
            -1.0,
            1.0,
        )?;
        if self.params.face_likeness_enabled
            && (self.assignments.is_empty()
                || self
                    .assignments
                    .iter()
                    .any(|assignment| assignment.target_identity_blake3.is_none()))
        {
            return Err(EngineSpecError::InvalidAssignments(
                "face likeness requires a target identity for every assignment".to_owned(),
            ));
        }
        validate_float_range("auto_color.blend", self.params.auto_color.blend, 0.0, 1.0)?;
        validate_float_range("restorer_alpha", self.params.restorer_alpha, 0.0, 1.0)?;
        validate_float_range("restorer2_alpha", self.params.restorer2_alpha, 0.0, 1.0)?;
        for (control, value, min, max) in [
            (
                "color_adjust.red",
                self.params.color_adjust.red,
                -100.0,
                100.0,
            ),
            (
                "color_adjust.green",
                self.params.color_adjust.green,
                -100.0,
                100.0,
            ),
            (
                "color_adjust.blue",
                self.params.color_adjust.blue,
                -100.0,
                100.0,
            ),
            (
                "color_adjust.brightness",
                self.params.color_adjust.brightness,
                0.0,
                2.0,
            ),
            (
                "color_adjust.contrast",
                self.params.color_adjust.contrast,
                0.0,
                2.0,
            ),
            (
                "color_adjust.saturation",
                self.params.color_adjust.saturation,
                0.0,
                2.0,
            ),
            (
                "color_adjust.sharpness",
                self.params.color_adjust.sharpness,
                0.0,
                2.0,
            ),
            ("color_adjust.hue", self.params.color_adjust.hue, -0.5, 0.5),
            (
                "color_adjust.gamma",
                self.params.color_adjust.gamma,
                0.0,
                2.0,
            ),
            (
                "color_adjust.noise",
                self.params.color_adjust.noise,
                0.0,
                20.0,
            ),
        ] {
            validate_float_range(control, value, min, max)?;
        }
        validate_range("jpeg_quality", self.params.jpeg_quality as i64, 1, 100)?;
        validate_range("final_blur", self.params.final_blur as i64, 1, 50)?;
        validate_range(
            "overall_mask_blur",
            self.params.overall_mask_blur as i64,
            0,
            100,
        )?;
        for (control, value, min, max) in [
            (
                "geometry.keypoints_x",
                self.params.geometry.keypoints_x,
                -100.0,
                100.0,
            ),
            (
                "geometry.keypoints_y",
                self.params.geometry.keypoints_y,
                -100.0,
                100.0,
            ),
            (
                "geometry.keypoints_scale",
                self.params.geometry.keypoints_scale,
                -100.0,
                100.0,
            ),
            (
                "geometry.face_scale",
                self.params.geometry.face_scale,
                -20.0,
                20.0,
            ),
        ] {
            validate_float_range(control, value, min, max)?;
        }
        for (index, offset) in self.params.geometry.landmark_offsets.iter().enumerate() {
            validate_float_range(
                &format!("geometry.landmark_offsets[{index}].x"),
                offset[0],
                -100.0,
                100.0,
            )?;
            validate_float_range(
                &format!("geometry.landmark_offsets[{index}].y"),
                offset[1],
                -100.0,
                100.0,
            )?;
        }
        validate_float_range(
            "similarity_threshold",
            self.params.similarity_threshold,
            0.0,
            1.0,
        )?;
        validate_range("occluder_size", self.params.occluder_size as i64, -100, 100)?;
        validate_range("xseg_size", self.params.xseg_size as i64, -100, 100)?;
        validate_range(
            "occluder_xseg_blur",
            self.params.occluder_xseg_blur as i64,
            0,
            100,
        )?;
        validate_range(
            "faceparser.background",
            self.params.faceparser.background as i64,
            -50,
            50,
        )?;
        for (control, value) in [
            ("face", self.params.faceparser.face),
            ("left_eyebrow", self.params.faceparser.left_eyebrow),
            ("right_eyebrow", self.params.faceparser.right_eyebrow),
            ("left_eye", self.params.faceparser.left_eye),
            ("right_eye", self.params.faceparser.right_eye),
            ("eyeglasses", self.params.faceparser.eyeglasses),
            ("nose", self.params.faceparser.nose),
            ("mouth", self.params.faceparser.mouth),
            ("upper_lip", self.params.faceparser.upper_lip),
            ("lower_lip", self.params.faceparser.lower_lip),
            ("neck", self.params.faceparser.neck),
            ("hair", self.params.faceparser.hair),
        ] {
            validate_range(&format!("faceparser.{control}"), value as i64, 0, 30)?;
        }
        for (control, value) in [
            ("background_blur", self.params.faceparser.background_blur),
            ("face_blur", self.params.faceparser.face_blur),
        ] {
            validate_range(&format!("faceparser.{control}"), value as i64, 0, 100)?;
        }
        for (control, value, min, max) in [
            (
                "restore_mouth.blend",
                self.params.restore_mouth_params.blend,
                0.0,
                1.0,
            ),
            (
                "restore_mouth.size_factor",
                self.params.restore_mouth_params.size_factor,
                0.05,
                0.60,
            ),
            (
                "restore_mouth.radius_x",
                self.params.restore_mouth_params.radius_x,
                0.3,
                3.0,
            ),
            (
                "restore_mouth.radius_y",
                self.params.restore_mouth_params.radius_y,
                0.3,
                3.0,
            ),
            (
                "restore_eyes.blend",
                self.params.restore_eyes_params.blend,
                0.0,
                1.0,
            ),
            (
                "restore_eyes.size_factor",
                self.params.restore_eyes_params.size_factor,
                2.0,
                4.0,
            ),
            (
                "restore_eyes.radius_x",
                self.params.restore_eyes_params.radius_x,
                0.3,
                3.0,
            ),
            (
                "restore_eyes.radius_y",
                self.params.restore_eyes_params.radius_y,
                0.3,
                3.0,
            ),
        ] {
            validate_float_range(control, value, min, max)?;
        }
        for (control, value, min, max) in [
            (
                "restore_mouth.feather",
                self.params.restore_mouth_params.feather as i64,
                1,
                100,
            ),
            (
                "restore_mouth.offset_x",
                self.params.restore_mouth_params.offset_x as i64,
                -300,
                300,
            ),
            (
                "restore_mouth.offset_y",
                self.params.restore_mouth_params.offset_y as i64,
                -300,
                300,
            ),
            (
                "restore_eyes.feather",
                self.params.restore_eyes_params.feather as i64,
                1,
                100,
            ),
            (
                "restore_eyes.offset_x",
                self.params.restore_eyes_params.offset_x as i64,
                -300,
                300,
            ),
            (
                "restore_eyes.offset_y",
                self.params.restore_eyes_params.offset_y as i64,
                -300,
                300,
            ),
            (
                "restore_eyes.spacing_offset",
                self.params.restore_eyes_params.spacing_offset as i64,
                -200,
                200,
            ),
            (
                "restore_eyes_mouth_blur",
                self.params.restore_eyes_mouth_blur as i64,
                0,
                50,
            ),
            (
                "differencing_amount",
                self.params.differencing_amount as i64,
                0,
                100,
            ),
            (
                "differencing_blur",
                self.params.differencing_blur as i64,
                0,
                100,
            ),
        ] {
            validate_range(control, value, min, max)?;
        }

        for (role, artifact) in &self.models {
            if artifact.logical_name.trim().is_empty() {
                return Err(EngineSpecError::EmptyField(format!(
                    "models.{}.logical_name",
                    role.as_str()
                )));
            }
            if artifact.filename.trim().is_empty() {
                return Err(EngineSpecError::EmptyField(format!(
                    "models.{}.filename",
                    role.as_str()
                )));
            }
            if Path::new(&artifact.filename).is_absolute()
                || Path::new(&artifact.filename)
                    .components()
                    .any(|component| !matches!(component, Component::Normal(_)))
            {
                return Err(EngineSpecError::InvalidFilename {
                    role: *role,
                    filename: artifact.filename.clone(),
                });
            }
            validate_blake3(
                &format!("models.{}.blake3", role.as_str()),
                &artifact.blake3,
            )?;

            if matches!(role, ModelRole::Restorer | ModelRole::Restorer2) {
                validate_restorer(&artifact.logical_name)?;
            }
        }

        for (enabled, role) in [
            (self.params.restorer_enabled, ModelRole::Restorer),
            (self.params.restorer2_enabled, ModelRole::Restorer2),
            (self.params.occluder_enabled, ModelRole::Occluder),
            (self.params.xseg_enabled, ModelRole::Xseg),
            (self.params.faceparser_enabled, ModelRole::FaceParser),
            (self.params.enhancer_enabled, ModelRole::Enhancer),
        ] {
            if enabled {
                self.require(role)?;
            }
        }

        if self.params.enhancer_enabled {
            let artifact = self.require(ModelRole::Enhancer)?;
            let selected = self.params.enhancer_model.registry_name();
            if artifact.logical_name != selected {
                return Err(EngineSpecError::EnhancerModelMismatch {
                    selected: selected.to_owned(),
                    artifact: artifact.logical_name.clone(),
                });
            }
        }
        if self.params.swapper_model == SwapperModel::Dfm {
            let artifact = self.require(ModelRole::Dfm)?;
            if artifact.filename != self.params.dfm_model {
                return Err(EngineSpecError::DfmModelMismatch {
                    selected: self.params.dfm_model.clone(),
                    artifact: artifact.filename.clone(),
                });
            }
        }

        Ok(())
    }

    /// Stable identifier used to coalesce and compare generation requests.
    pub fn generation_digest(&self) -> Result<String, EngineSpecError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| EngineSpecError::Serialization(error.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }

    fn require(&self, role: ModelRole) -> Result<&ModelArtifact, EngineSpecError> {
        self.models
            .get(&role)
            .ok_or(EngineSpecError::MissingModel(role))
    }
}

fn same_generation_controls(a: &FaceSwapParams, b: &FaceSwapParams) -> bool {
    a.detector_score == b.detector_score
        && a.max_faces == b.max_faces
        && a.auto_rotation == b.auto_rotation
        && a.manual_rotation_enabled == b.manual_rotation_enabled
        && a.manual_rotation_angle == b.manual_rotation_angle
        && a.enhancer_enabled == b.enhancer_enabled
        && a.enhancer_model == b.enhancer_model
        && a.enhancer_blend == b.enhancer_blend
}

fn register_runtime_sessions(
    sessions: &mut BTreeMap<String, String>,
    params: &FaceSwapParams,
    models: &BTreeMap<ModelRole, ModelArtifact>,
) -> Result<(), EngineSpecError> {
    let mut roles = Vec::<(String, ModelRole)>::new();
    if params.swapper_model == SwapperModel::Inswapper128 {
        roles.extend([
            ("Inswapper128ArcFace".to_owned(), ModelRole::Recognizer),
            ("Inswapper128".to_owned(), ModelRole::Swapper),
            ("InswapperEMap".to_owned(), ModelRole::Emap),
        ]);
    }
    if params.landmark_enabled {
        roles.push((
            params.landmark_mode.registry_name().to_owned(),
            ModelRole::Landmark,
        ));
    }
    if params.restorer_enabled {
        roles.push((
            match params.restorer_size {
                RestorerSize::Gpen256 => "GPENBFR256",
                RestorerSize::Gpen512 => "GPENBFR512",
                RestorerSize::Gpen1024 => "GPENBFR1024",
            }
            .to_owned(),
            ModelRole::Restorer,
        ));
    }
    if params.restorer2_enabled {
        roles.push((
            match params.restorer2_size {
                RestorerSize::Gpen256 => "GPENBFR256_2",
                RestorerSize::Gpen512 => "GPENBFR512_2",
                RestorerSize::Gpen1024 => "GPENBFR1024_2",
            }
            .to_owned(),
            ModelRole::Restorer2,
        ));
    }
    for (enabled, name, role) in [
        (params.occluder_enabled, "Occluder", ModelRole::Occluder),
        (params.xseg_enabled, "XSeg", ModelRole::Xseg),
        (
            params.faceparser_enabled,
            "FaceParser",
            ModelRole::FaceParser,
        ),
    ] {
        if enabled {
            roles.push((name.to_owned(), role));
        }
    }
    for (session, role) in roles {
        let Some(artifact) = models.get(&role) else {
            continue;
        };
        if let Some(previous) = sessions.insert(session.clone(), artifact.blake3.clone())
            && previous != artifact.blake3
        {
            return Err(EngineSpecError::InvalidAssignments(format!(
                "runtime session {session} maps to multiple model digests"
            )));
        }
    }
    Ok(())
}

fn validate_range(control: &str, value: i64, min: i64, max: i64) -> Result<(), EngineSpecError> {
    if (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(EngineSpecError::InvalidMaskControl {
            control: control.to_owned(),
            value,
            min,
            max,
        })
    }
}

fn validate_float_range(
    control: &str,
    value: f32,
    min: f32,
    max: f32,
) -> Result<(), EngineSpecError> {
    if value.is_finite() && (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(EngineSpecError::InvalidFloatControl {
            control: control.to_owned(),
            value: value.to_string(),
            min: min.to_string(),
            max: max.to_string(),
        })
    }
}

fn validate_blake3(field: &str, value: &str) -> Result<(), EngineSpecError> {
    if value.len() == 64
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        Ok(())
    } else {
        Err(EngineSpecError::InvalidBlake3 {
            field: field.to_owned(),
            value: value.to_owned(),
        })
    }
}

fn validate_restorer(logical_name: &str) -> Result<(), EngineSpecError> {
    let normalized: String = logical_name
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_uppercase)
        .collect();
    if matches!(
        normalized.as_str(),
        "GPEN256" | "GPENBFR256" | "GPEN512" | "GPENBFR512"
    ) {
        Ok(())
    } else {
        Err(EngineSpecError::UnsupportedRestorer(
            logical_name.to_owned(),
        ))
    }
}
