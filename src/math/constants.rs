//! Face landmark template constants.
//!
//! Ported from crosswap/app/processors/utils/constants.py

/// ArcFace 5-point template for 112×112 input.
/// [left_eye, right_eye, nose, left_mouth, right_mouth]
pub const ARCFACE_DST: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// FFHQ 5-point template for 512×512 input (used by face restorers).
pub const FFHQ_KPS: [[f32; 2]; 5] = [
    [192.98138, 239.94708],
    [318.90277, 240.193_6],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
];

/// Default face detector input size (YoloFace8n = 640, SCRFD/RetinaFace = 512).
pub const DETECTOR_INPUT_SIZE: u32 = 640;

/// ArcFace embedding dimension.
pub const EMBEDDING_SIZE: usize = 512;

/// Inswapper128 tile size.
pub const SWAP_TILE_SIZE: u32 = 128;

/// Inswapper128 embedding/latent size.
pub const SWAP_LATENT_SIZE: usize = 512;

/// Inswapper128 emap matrix [512×512] — extracted from ONNX initializer at runtime.
/// Stored as flat row-major: emap[i*512 + j]
pub const EMAP_SIZE: usize = 512 * 512;
