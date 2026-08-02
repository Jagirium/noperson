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

/// CrossSwap's five pose-aware ArcFace templates used by the `Optimal` mode.
pub const ARCFACE_MAP_TEMPLATES: [[[f32; 2]; 5]; 5] = [
    [
        [51.642, 50.115],
        [57.617, 49.990],
        [35.740, 69.007],
        [51.157, 89.050],
        [57.025, 89.702],
    ],
    [
        [45.031, 50.118],
        [65.568, 50.872],
        [39.677, 68.111],
        [45.177, 86.190],
        [64.246, 86.758],
    ],
    [
        [39.730, 51.138],
        [72.270, 51.138],
        [56.000, 68.493],
        [42.463, 87.010],
        [69.537, 87.010],
    ],
    [
        [46.845, 50.872],
        [67.382, 50.118],
        [72.737, 68.111],
        [48.167, 86.758],
        [67.236, 86.190],
    ],
    [
        [54.796, 49.990],
        [60.771, 50.115],
        [76.673, 69.007],
        [55.388, 89.702],
        [61.257, 89.050],
    ],
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
