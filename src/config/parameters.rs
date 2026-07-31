//! Typed per-face swap parameters.
//! Replaces dict-based parameters with string keys.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Full-frame enhancement model selected for photo/video output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnhancerModel {
    #[default]
    RealEsrganX2Plus,
    RealEsrganX4Plus,
    RealEsrGeneralX4V3,
    BsrganX2,
    BsrganX4,
    UltraSharpX4,
    UltraMixX4,
}

impl EnhancerModel {
    pub fn from_crosswap_name(name: &str) -> Result<Self, EnhancerModelParseError> {
        match name {
            "RealEsrgan-x2-Plus" => Ok(Self::RealEsrganX2Plus),
            "RealEsrgan-x4-Plus" => Ok(Self::RealEsrganX4Plus),
            "RealEsr-General-x4v3" => Ok(Self::RealEsrGeneralX4V3),
            "BSRGan-x2" => Ok(Self::BsrganX2),
            "BSRGan-x4" => Ok(Self::BsrganX4),
            "UltraSharp-x4" => Ok(Self::UltraSharpX4),
            "UltraMix-x4" => Ok(Self::UltraMixX4),
            _ => Err(EnhancerModelParseError(name.to_owned())),
        }
    }

    pub const fn crosswap_name(self) -> &'static str {
        match self {
            Self::RealEsrganX2Plus => "RealEsrgan-x2-Plus",
            Self::RealEsrganX4Plus => "RealEsrgan-x4-Plus",
            Self::RealEsrGeneralX4V3 => "RealEsr-General-x4v3",
            Self::BsrganX2 => "BSRGan-x2",
            Self::BsrganX4 => "BSRGan-x4",
            Self::UltraSharpX4 => "UltraSharp-x4",
            Self::UltraMixX4 => "UltraMix-x4",
        }
    }

    pub const fn registry_name(self) -> &'static str {
        match self {
            Self::RealEsrganX2Plus => "RealEsrganx2Plus",
            Self::RealEsrganX4Plus => "RealEsrganx4Plus",
            Self::RealEsrGeneralX4V3 => "RealEsrx4v3",
            Self::BsrganX2 => "BSRGANx2",
            Self::BsrganX4 => "BSRGANx4",
            Self::UltraSharpX4 => "UltraSharpx4",
            Self::UltraMixX4 => "UltraMixx4",
        }
    }

    pub const fn scale(self) -> u32 {
        match self {
            Self::RealEsrganX2Plus | Self::BsrganX2 => 2,
            Self::RealEsrganX4Plus
            | Self::RealEsrGeneralX4V3
            | Self::BsrganX4
            | Self::UltraSharpX4
            | Self::UltraMixX4 => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("unknown CrossSwap frame enhancer {0}")]
pub struct EnhancerModelParseError(String);

/// Swap resolution: how many tiles for Inswapper128.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwapDim {
    /// dim=1: 1 tile 128×128, 1 inference call.
    Dim1 = 1,
    /// dim=2: 4 tiles 128×128 → 256×256, 1 batched call.
    Dim2 = 2,
    /// dim=3: 9 tiles 128×128 → 384×384, 1 batched call.
    Dim3 = 3,
    /// dim=4: 16 tiles 128×128 → 512×512, 1 batched call.
    Dim4 = 4,
}

/// Face restorer model size.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RestorerSize {
    Gpen256,
    Gpen512,
    Gpen1024,
}

/// Restorer hot-path policy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RestorerMode {
    /// Run GPEN for every frame. Best temporal detail, lower throughput.
    Quality,
    /// Run GPEN on keyframes and reuse its aligned GPU output in between.
    #[default]
    Realtime,
}

/// Per-face swap parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceSwapParams {
    pub enabled: bool,
    pub dim: SwapDim,

    // Restorer
    pub restorer_enabled: bool,
    pub restorer_size: RestorerSize,
    pub restorer_alpha: f32,
    #[serde(default)]
    pub restorer_mode: RestorerMode,

    // Full-frame enhancement (photo/video; disabled in realtime by CrossSwap)
    #[serde(default)]
    pub enhancer_enabled: bool,
    #[serde(default)]
    pub enhancer_model: EnhancerModel,
    #[serde(default = "default_enhancer_blend")]
    pub enhancer_blend: f32,

    // Masks
    pub occluder_enabled: bool,
    pub xseg_enabled: bool,
    pub faceparser_enabled: bool,
    pub restore_mouth: bool,
    pub restore_eyes: bool,

    // Border mask
    pub border_top: u32,
    pub border_bottom: u32,
    pub border_left: u32,
    pub border_right: u32,
    pub border_blur: u32,

    // Color correction
    pub color_correction: bool,
    pub histogram_matching: bool,

    // Strength
    pub strength: f32,

    // Similarity threshold for face matching
    pub similarity_threshold: f32,
}

impl Default for FaceSwapParams {
    fn default() -> Self {
        Self {
            enabled: true,
            dim: SwapDim::Dim1,
            restorer_enabled: false,
            restorer_size: RestorerSize::Gpen512,
            restorer_alpha: 1.0,
            restorer_mode: RestorerMode::Realtime,
            enhancer_enabled: false,
            enhancer_model: EnhancerModel::default(),
            enhancer_blend: default_enhancer_blend(),
            occluder_enabled: false,
            xseg_enabled: false,
            faceparser_enabled: false,
            restore_mouth: false,
            restore_eyes: false,
            border_top: 10,
            border_bottom: 10,
            border_left: 10,
            border_right: 10,
            border_blur: 10,
            color_correction: false,
            histogram_matching: false,
            strength: 1.0,
            similarity_threshold: 0.6,
        }
    }
}

const fn default_enhancer_blend() -> f32 {
    1.0
}
