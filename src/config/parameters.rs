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

/// FaceParser class-mask controls, matching CrossSwap's parser sliders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceParserMaskParams {
    pub background: i32,
    pub face: u32,
    pub left_eyebrow: u32,
    pub right_eyebrow: u32,
    pub left_eye: u32,
    pub right_eye: u32,
    pub eyeglasses: u32,
    pub nose: u32,
    pub mouth: u32,
    pub upper_lip: u32,
    pub lower_lip: u32,
    pub neck: u32,
    pub hair: u32,
    pub background_blur: u32,
    pub face_blur: u32,
}

impl Default for FaceParserMaskParams {
    fn default() -> Self {
        Self {
            background: 0,
            face: 0,
            left_eyebrow: 0,
            right_eyebrow: 0,
            left_eye: 0,
            right_eye: 0,
            eyeglasses: 0,
            nose: 0,
            mouth: 0,
            upper_lip: 0,
            lower_lip: 0,
            neck: 0,
            hair: 0,
            background_blur: 5,
            face_blur: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreMouthParams {
    pub blend: f32,
    pub feather: u32,
    pub size_factor: f32,
    pub radius_x: f32,
    pub radius_y: f32,
    pub offset_x: i32,
    pub offset_y: i32,
}

impl Default for RestoreMouthParams {
    fn default() -> Self {
        Self {
            blend: 0.5,
            feather: 10,
            size_factor: 0.25,
            radius_x: 1.0,
            radius_y: 1.0,
            offset_x: 0,
            offset_y: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreEyesParams {
    pub blend: f32,
    pub feather: u32,
    pub size_factor: f32,
    pub radius_x: f32,
    pub radius_y: f32,
    pub offset_x: i32,
    pub offset_y: i32,
    pub spacing_offset: i32,
}

impl Default for RestoreEyesParams {
    fn default() -> Self {
        Self {
            blend: 0.5,
            feather: 10,
            size_factor: 3.0,
            radius_x: 1.0,
            radius_y: 1.0,
            offset_x: 0,
            offset_y: 0,
            spacing_offset: 0,
        }
    }
}

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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwapperModel {
    #[default]
    Inswapper128,
    Dfm,
}

/// Detailed landmark refiner selected for one immutable engine generation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LandmarkMode {
    #[serde(rename = "5")]
    Points5,
    #[serde(rename = "68")]
    Points68,
    #[serde(rename = "3d68")]
    Points3d68,
    #[serde(rename = "98")]
    Points98,
    #[serde(rename = "106")]
    Points106,
    #[default]
    #[serde(rename = "203")]
    Points203,
    #[serde(rename = "478")]
    Points478,
}

impl LandmarkMode {
    pub const fn registry_name(self) -> &'static str {
        match self {
            Self::Points5 => "FaceLandmark5",
            Self::Points68 => "FaceLandmark68",
            Self::Points3d68 => "FaceLandmark3d68",
            Self::Points98 => "FaceLandmark98",
            Self::Points106 => "FaceLandmark106",
            Self::Points203 => "FaceLandmark203",
            Self::Points478 => "FaceLandmark478",
        }
    }

    pub const fn filename(self) -> &'static str {
        match self {
            Self::Points5 => "res50.onnx",
            Self::Points68 => "2dfan4.onnx",
            Self::Points3d68 => "1k3d68.onnx",
            Self::Points98 => "peppapig_teacher_Nx3x256x256.onnx",
            Self::Points106 => "2d106det.onnx",
            Self::Points203 => "landmark.onnx",
            Self::Points478 => "face_landmarks_detector_Nx3x256x256.onnx",
        }
    }
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
    #[default]
    Quality,
    /// Run GPEN on keyframes and reuse its aligned GPU output in between.
    Realtime,
}

/// Landmark source used to align a face before restoration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RestorerAlignment {
    /// Restore the swap crop without a second alignment pass.
    #[default]
    Original,
    /// Re-align the canonical ArcFace points to the FFHQ restorer template.
    Blend,
    /// Detect fresh five-point landmarks on the swapped crop, then align them.
    Reference,
}

/// CrossSwap auto-color algorithm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutoColorMode {
    /// Global empirical-CDF histogram matching (CrossSwap `Test`).
    #[default]
    Histogram,
    /// Histogram matching blended through the face mask (`Test_Mask`).
    HistogramMasked,
    /// Global LAB mean/std transfer (`DFL_Test`).
    Dfl,
    /// LAB mean/std transfer using masked statistics (`DFL_Orig`).
    DflMasked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoColorParams {
    pub enabled: bool,
    pub mode: AutoColorMode,
    /// CrossSwap's AutoColor blend slider normalized to 0..1.
    pub blend: f32,
}

impl Default for AutoColorParams {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: AutoColorMode::Histogram,
            blend: 0.8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorAdjustParams {
    pub enabled: bool,
    pub red: f32,
    pub green: f32,
    pub blue: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub sharpness: f32,
    pub hue: f32,
    pub gamma: f32,
    pub noise: f32,
}

impl Default for ColorAdjustParams {
    fn default() -> Self {
        Self {
            enabled: false,
            red: 0.0,
            green: 0.0,
            blue: 0.0,
            brightness: 1.0,
            contrast: 1.0,
            saturation: 1.0,
            sharpness: 1.0,
            hue: 0.0,
            gamma: 1.0,
            noise: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceGeometryParams {
    pub enabled: bool,
    pub keypoints_x: f32,
    pub keypoints_y: f32,
    pub keypoints_scale: f32,
    pub face_scale: f32,
    pub landmark_offsets_enabled: bool,
    /// Left eye, right eye, nose, left mouth, right mouth offsets.
    pub landmark_offsets: [[f32; 2]; 5],
}

impl Default for FaceGeometryParams {
    fn default() -> Self {
        Self {
            enabled: false,
            keypoints_x: 0.0,
            keypoints_y: 0.0,
            keypoints_scale: 0.0,
            face_scale: 0.0,
            landmark_offsets_enabled: false,
            landmark_offsets: [[0.0; 2]; 5],
        }
    }
}

/// Per-face swap parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceSwapParams {
    pub enabled: bool,
    pub dim: SwapDim,
    #[serde(default)]
    pub swapper_model: SwapperModel,
    #[serde(default = "default_dfm_model")]
    pub dfm_model: String,
    #[serde(default = "default_dfm_morph")]
    pub dfm_morph: f32,
    #[serde(default)]
    pub dfm_rct: bool,

    // Detection
    #[serde(default = "default_detector_score")]
    pub detector_score: f32,
    #[serde(default = "default_max_faces")]
    pub max_faces: usize,
    #[serde(default)]
    pub auto_rotation: bool,
    #[serde(default)]
    pub manual_rotation_enabled: bool,
    #[serde(default)]
    pub manual_rotation_angle: u16,

    // Detailed face alignment
    #[serde(default)]
    pub landmark_enabled: bool,
    #[serde(default)]
    pub landmark_mode: LandmarkMode,
    #[serde(default = "default_landmark_score")]
    pub landmark_score: f32,
    #[serde(default)]
    pub landmark_from_points: bool,

    // Restorer
    pub restorer_enabled: bool,
    pub restorer_size: RestorerSize,
    pub restorer_alpha: f32,
    #[serde(default)]
    pub restorer_mode: RestorerMode,
    #[serde(default)]
    pub restorer_alignment: RestorerAlignment,
    #[serde(default)]
    pub restorer2_enabled: bool,
    #[serde(default = "default_restorer_size")]
    pub restorer2_size: RestorerSize,
    #[serde(default = "default_restorer_alpha")]
    pub restorer2_alpha: f32,
    #[serde(default)]
    pub restorer2_mode: RestorerMode,
    #[serde(default)]
    pub restorer2_alignment: RestorerAlignment,

    // Full-frame enhancement (photo/video; disabled in realtime by CrossSwap)
    #[serde(default)]
    pub enhancer_enabled: bool,
    #[serde(default)]
    pub enhancer_model: EnhancerModel,
    #[serde(default = "default_enhancer_blend")]
    pub enhancer_blend: f32,

    // Masks
    pub occluder_enabled: bool,
    #[serde(default)]
    pub occluder_size: i32,
    pub xseg_enabled: bool,
    #[serde(default)]
    pub xseg_size: i32,
    #[serde(default)]
    pub occluder_xseg_blur: u32,
    pub faceparser_enabled: bool,
    #[serde(default)]
    pub faceparser: FaceParserMaskParams,
    pub restore_mouth: bool,
    pub restore_eyes: bool,
    #[serde(default)]
    pub restore_mouth_params: RestoreMouthParams,
    #[serde(default)]
    pub restore_eyes_params: RestoreEyesParams,
    #[serde(default)]
    pub restore_eyes_mouth_blur: u32,
    #[serde(default)]
    pub differencing_enabled: bool,
    #[serde(default = "default_differencing_amount")]
    pub differencing_amount: u32,
    #[serde(default = "default_differencing_blur")]
    pub differencing_blur: u32,

    // Border mask
    pub border_top: u32,
    pub border_bottom: u32,
    pub border_left: u32,
    pub border_right: u32,
    pub border_blur: u32,

    // Color correction
    #[serde(default)]
    pub auto_color: AutoColorParams,
    #[serde(default)]
    pub color_adjust: ColorAdjustParams,
    #[serde(default)]
    pub geometry: FaceGeometryParams,
    /// Legacy compact-GUI toggle. Kept for config compatibility.
    pub color_correction: bool,
    /// Legacy compact-GUI toggle. Kept for config compatibility.
    pub histogram_matching: bool,

    // Final face/mask degradation controls
    #[serde(default)]
    pub jpeg_compression_enabled: bool,
    #[serde(default = "default_jpeg_quality")]
    pub jpeg_quality: u8,
    #[serde(default)]
    pub final_blur_enabled: bool,
    #[serde(default = "default_final_blur")]
    pub final_blur: u32,
    #[serde(default)]
    pub overall_mask_blur: u32,

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
            swapper_model: SwapperModel::default(),
            dfm_model: default_dfm_model(),
            dfm_morph: default_dfm_morph(),
            dfm_rct: false,
            detector_score: default_detector_score(),
            max_faces: default_max_faces(),
            auto_rotation: false,
            manual_rotation_enabled: false,
            manual_rotation_angle: 0,
            landmark_enabled: false,
            landmark_mode: LandmarkMode::default(),
            landmark_score: default_landmark_score(),
            landmark_from_points: false,
            restorer_enabled: false,
            restorer_size: RestorerSize::Gpen256,
            restorer_alpha: 1.0,
            restorer_mode: RestorerMode::Quality,
            restorer_alignment: RestorerAlignment::Original,
            restorer2_enabled: false,
            restorer2_size: default_restorer_size(),
            restorer2_alpha: default_restorer_alpha(),
            restorer2_mode: RestorerMode::Quality,
            restorer2_alignment: RestorerAlignment::Original,
            enhancer_enabled: false,
            enhancer_model: EnhancerModel::default(),
            enhancer_blend: default_enhancer_blend(),
            occluder_enabled: false,
            occluder_size: 0,
            xseg_enabled: false,
            xseg_size: 0,
            occluder_xseg_blur: 0,
            faceparser_enabled: false,
            faceparser: FaceParserMaskParams::default(),
            restore_mouth: false,
            restore_eyes: false,
            restore_mouth_params: RestoreMouthParams::default(),
            restore_eyes_params: RestoreEyesParams::default(),
            restore_eyes_mouth_blur: 0,
            differencing_enabled: false,
            differencing_amount: default_differencing_amount(),
            differencing_blur: default_differencing_blur(),
            border_top: 10,
            border_bottom: 10,
            border_left: 10,
            border_right: 10,
            border_blur: 10,
            auto_color: AutoColorParams::default(),
            color_adjust: ColorAdjustParams::default(),
            geometry: FaceGeometryParams::default(),
            color_correction: false,
            histogram_matching: false,
            jpeg_compression_enabled: false,
            jpeg_quality: default_jpeg_quality(),
            final_blur_enabled: false,
            final_blur: default_final_blur(),
            overall_mask_blur: 0,
            strength: 1.0,
            similarity_threshold: 0.6,
        }
    }
}

const fn default_enhancer_blend() -> f32 {
    1.0
}

const fn default_restorer_size() -> RestorerSize {
    RestorerSize::Gpen256
}

const fn default_restorer_alpha() -> f32 {
    1.0
}

const fn default_jpeg_quality() -> u8 {
    50
}

const fn default_final_blur() -> u32 {
    1
}

const fn default_landmark_score() -> f32 {
    0.5
}

const fn default_detector_score() -> f32 {
    0.5
}

const fn default_max_faces() -> usize {
    20
}

const fn default_differencing_amount() -> u32 {
    4
}

const fn default_differencing_blur() -> u32 {
    5
}

fn default_dfm_model() -> String {
    "JasonStatham320.dfm".to_owned()
}

const fn default_dfm_morph() -> f32 {
    0.5
}
