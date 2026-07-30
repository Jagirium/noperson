//! Face landmark detectors: 5/68/3d68/98/106/203/478 points.
//!
//! Port of crosswap/app/processors/face_landmark_detectors.py
//! Each model has unique input size, normalization, and output decoding.

/// Landmark model types.
#[derive(Debug, Clone, Copy)]
pub enum LandmarkModel {
    /// res50.onnx — 5 points, input 512×512, norm: subtract BGR mean [104,117,123]
    Points5,
    /// 2dfan4.onnx — 68 points, input 256×256, norm: /255
    Points68,
    /// 1k3d68.onnx — 68 3D points, input 192×192, ImageNet norm
    Points3d68,
    /// peppapig — 98 points, input 256×256, norm: /255
    Points98,
    /// 2d106det.onnx — 106 points, input 192×192, ImageNet norm
    Points106,
    /// landmark.onnx — 203 points, input 224×224, norm: /255
    Points203,
    /// face_landmarks_detector — 478 MediaPipe points, input 256×256, norm: /255
    Points478,
}

impl LandmarkModel {
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
                // Subtract BGR mean (but we're in RGB, so swap channels)
                let mean = match channel {
                    0 => 123.0, // R ← B mean
                    1 => 117.0, // G ← G mean
                    2 => 104.0, // B ← R mean
                    _ => 0.0,
                };
                pixel - mean
            }
            Self::Points68 | Self::Points98 | Self::Points203 | Self::Points478 => pixel / 255.0,
            Self::Points3d68 | Self::Points106 => {
                // ImageNet normalization: (pixel/255 - mean) / std
                let (mean, std) = match channel {
                    0 => (0.485, 0.229),
                    1 => (0.456, 0.224),
                    2 => (0.406, 0.225),
                    _ => (0.0, 1.0),
                };
                (pixel / 255.0 - mean) / std
            }
        }
    }
}
