use std::collections::BTreeMap;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
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
    pub sha256: String,
}

/// Complete immutable configuration for a buildable engine generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineSpec {
    pub provider: ExecutionProvider,
    pub device_id: i32,
    pub detector: DetectorModel,
    pub identity_sha256: String,
    pub models: BTreeMap<ModelRole, ModelArtifact>,
    pub params: FaceSwapParams,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EngineSpecError {
    #[error("engine spec requires model role {0:?}")]
    MissingModel(ModelRole),
    #[error("{field} must be a lowercase SHA-256 digest, got {value}")]
    InvalidSha256 { field: String, value: String },
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

        validate_sha256("identity_sha256", &self.identity_sha256)?;

        if matches!(self.params.restorer_size, RestorerSize::Gpen1024) {
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
        validate_float_range("landmark_score", self.params.landmark_score, 0.01, 1.0)?;
        validate_float_range("strength", self.params.strength, 0.0, 5.0)?;
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
            validate_sha256(
                &format!("models.{}.sha256", role.as_str()),
                &artifact.sha256,
            )?;

            if *role == ModelRole::Restorer {
                validate_restorer(&artifact.logical_name)?;
            }
        }

        for (enabled, role) in [
            (self.params.restorer_enabled, ModelRole::Restorer),
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
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    fn require(&self, role: ModelRole) -> Result<&ModelArtifact, EngineSpecError> {
        self.models
            .get(&role)
            .ok_or(EngineSpecError::MissingModel(role))
    }
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

fn validate_sha256(field: &str, value: &str) -> Result<(), EngineSpecError> {
    if value.len() == 64
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        Ok(())
    } else {
        Err(EngineSpecError::InvalidSha256 {
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
