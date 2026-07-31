//! Pipeline — face swap processing stages.
//!
//! Each stage uses ort (ONNX Runtime) + cudarc (GPU compute).
//! Pipeline orchestration: detect → recognize → swap → mask → paste-back.

pub mod dfm;
pub mod face_detector;
pub mod face_landmark;
pub mod face_mask;
pub mod face_recognizer;
pub mod face_swapper;
pub mod frame_enhancer;
pub mod frame_processor;
pub mod ort_binding;
pub mod workspace;

/// Detected face with bounding box and 5-point landmarks.
#[derive(Debug, Clone)]
pub struct DetectedFace {
    /// Bounding box [x1, y1, x2, y2] in original frame coordinates.
    pub bbox: [f32; 4],
    /// 5-point landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth].
    pub kps_5: [[f32; 2]; 5],
    /// Detection confidence [0, 1].
    pub score: f32,
}

/// A face ready for swapping — has embedding + latent computed.
#[derive(Debug, Clone)]
pub struct PreparedFace {
    pub bbox: [f32; 4],
    pub kps_5: [[f32; 2]; 5],
    pub embedding: Vec<f32>,
    pub latent: Vec<f32>,
}
