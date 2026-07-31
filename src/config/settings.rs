//! Global pipeline settings.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectorModel {
    YoloFace8n,
    RetinaFace,
    Scrfd2_5g,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionProvider {
    Cuda,
    TensorRT,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSettings {
    pub detector: DetectorModel,
    pub provider: ExecutionProvider,
    pub device_id: i32,

    /// Detection confidence threshold.
    pub detect_threshold: f32,

    /// Max faces to process per frame (0 = unlimited).
    pub max_faces: usize,

    /// Temporal detection cache: reuse detections for N frames (webcam mode).
    pub detect_interval: u32,
}

impl Default for PipelineSettings {
    fn default() -> Self {
        Self {
            detector: DetectorModel::YoloFace8n,
            provider: ExecutionProvider::Cuda,
            device_id: 0,
            detect_threshold: 0.5,
            max_faces: 0,
            detect_interval: 1,
        }
    }
}
