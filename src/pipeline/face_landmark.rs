//! Face landmark detectors: 5/68/3d68/98/106/203/478 points.
//!
//! Port of crosswap/app/processors/face_landmark_detectors.py
//! Each model has unique input size, normalization, and output decoding.

use thiserror::Error;

/// Landmark model types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LandmarkModel {
    /// res50.onnx — 5 points, input 512×512, norm: subtract BGR mean [104,117,123]
    Points5,
    /// 2dfan4.onnx — 68 points, input 256×256, norm: /255
    Points68,
    /// 1k3d68.onnx — 68 3D points, input 192×192, identity normalization
    Points3d68,
    /// peppapig — 98 points, input 256×256, norm: /255
    Points98,
    /// 2d106det.onnx — 106 points, input 192×192, identity normalization
    Points106,
    /// landmark.onnx — 203 points, input 224×224, norm: /255
    Points203,
    /// face_landmarks_detector — 478 MediaPipe points, input 256×256, norm: /255
    Points478,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LandmarkError {
    #[error("unsupported landmark mode {0}")]
    UnsupportedMode(String),
    #[error("landmark model expected {expected} points, got {actual}")]
    PointCount { expected: usize, actual: usize },
}

impl LandmarkModel {
    pub fn from_mode(mode: &str) -> Result<Self, LandmarkError> {
        match mode {
            "5" => Ok(Self::Points5),
            "68" => Ok(Self::Points68),
            "3d68" => Ok(Self::Points3d68),
            "98" => Ok(Self::Points98),
            "106" => Ok(Self::Points106),
            "203" => Ok(Self::Points203),
            "478" => Ok(Self::Points478),
            _ => Err(LandmarkError::UnsupportedMode(mode.to_owned())),
        }
    }

    pub fn model_name(&self) -> &'static str {
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

    pub fn input_size(&self) -> u32 {
        match self {
            Self::Points5 => 512,
            Self::Points68 | Self::Points98 | Self::Points478 => 256,
            Self::Points3d68 | Self::Points106 => 192,
            Self::Points203 => 224,
        }
    }

    pub fn num_points(&self) -> usize {
        match self {
            Self::Points5 => 5,
            Self::Points68 | Self::Points3d68 => 68,
            Self::Points98 => 98,
            Self::Points106 => 106,
            Self::Points203 => 203,
            Self::Points478 => 478,
        }
    }

    pub fn onnx_filename(&self) -> &'static str {
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

    /// Apply model-specific normalization to pixel values.
    pub fn normalize(&self, pixel: f32, channel: u32) -> f32 {
        match self {
            Self::Points5 => {
                // CrossSwap subtracts this vector after CHW→HWC, without a channel swap.
                let mean = match channel {
                    0 => 104.0,
                    1 => 117.0,
                    2 => 123.0,
                    _ => 0.0,
                };
                pixel - mean
            }
            Self::Points68 | Self::Points98 | Self::Points203 | Self::Points478 => pixel / 255.0,
            Self::Points3d68 | Self::Points106 => pixel,
        }
    }

    /// Reduce a model's dense output to CrossSwap's canonical ArcFace-5 order.
    pub fn to_five(&self, points: &[[f32; 2]]) -> Result<[[f32; 2]; 5], LandmarkError> {
        let expected = self.num_points();
        if points.len() != expected {
            return Err(LandmarkError::PointCount {
                expected,
                actual: points.len(),
            });
        }

        let result = match self {
            Self::Points5 => [points[0], points[1], points[2], points[3], points[4]],
            Self::Points68 | Self::Points3d68 => [
                mean_pair(points[36], points[39]),
                mean_pair(points[42], points[45]),
                points[30],
                points[48],
                points[54],
            ],
            Self::Points98 => [points[96], points[97], points[54], points[76], points[82]],
            Self::Points106 => [points[38], points[88], points[86], points[52], points[61]],
            Self::Points203 => [
                points[197],
                points[198],
                points[201],
                points[48],
                points[66],
            ],
            Self::Points478 => [points[468], points[473], points[4], points[61], points[291]],
        };
        Ok(result)
    }
}

fn mean_pair(left: [f32; 2], right: [f32; 2]) -> [f32; 2] {
    [(left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5]
}
