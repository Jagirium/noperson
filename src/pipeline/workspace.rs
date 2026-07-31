//! Pre-allocated GPU workspace — zero allocations in hot path.
//!
//! All buffers are allocated once at startup. The pipeline reuses them
//! frame after frame. Buffer sizes are based on max resolution (1080p default).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DriverError, PinnedHostSlice};

/// Maximum supported frame dimensions.
pub const MAX_WIDTH: usize = 1920;
pub const MAX_HEIGHT: usize = 1080;
pub const MAX_SWAP_DIM: usize = 4;
/// Maximum faces processed per frame (ArcFace batch size).
pub const MAX_FACES: usize = 8;

/// Pre-allocated GPU buffers for the entire pipeline.
/// Note: the main frame buffers (frame_chw, frame_u8_in/out) live OUTSIDE the
/// workspace so we can borrow them independently from the scratch buffers.
pub struct GpuWorkspace {
    // Detection (YoloFace input: 640×640, 0..1 normalized)
    pub detect_input: CudaSlice<f32>, // [1, 3, 640, 640]
    /// YoloFace output: [1, 20, 8400] for 640 input — 4 bbox + 1 score + 15 kps
    pub detect_output: CudaSlice<f32>, // [1, 20, 8400]

    // Per-face buffers (reused)
    pub face_112: CudaSlice<f32>,          // ArcFace input [3, 112, 112]
    pub face_256: CudaSlice<f32>,          // swap face [3, dim*128, dim*128] (max 512)
    pub face_512: CudaSlice<f32>,          // Pipeline face [3, 512, 512]
    pub face_512_original: CudaSlice<f32>, // Original aligned crop for learned masks
    pub face_512_pre_restorer: CudaSlice<f32>, // Swapped crop before restoration/parser
    pub face_512_scratch: CudaSlice<f32>,  // Scratch for upscaled swap [3, 512, 512]
    pub restorer_256_input: CudaSlice<f32>,
    pub restorer_256_output: CudaSlice<f32>,
    /// Last GPEN output in aligned 512-space. Reused between inference keyframes.
    pub restorer_cache: CudaSlice<f32>,
    pub restorer_cache_valid: bool,
    pub restorer_frame: u64,

    // ArcFace output: [1, 512] — pre-allocated for IoBinding path
    pub arcface_embedding: CudaSlice<f32>, // [1, 512]

    // Batched ArcFace buffers — for recognize_batch_gpu (multiple faces in 1 ort call)
    pub face_112_batch: CudaSlice<f32>, // [MAX_FACES, 3, 112, 112]
    pub arcface_embeddings_batch: CudaSlice<f32>, // [MAX_FACES, 512]
    pub host_face_112_batch: Vec<f32>,  // [MAX_FACES * 3 * 112 * 112]
    pub host_embeddings_batch: Vec<f32>, // [MAX_FACES * 512]

    // Swap (batched dim=1..4 → up to 16 tiles)
    pub swap_batch_in: CudaSlice<f32>,     // [16, 3, 128, 128]
    pub swap_batch_out: CudaSlice<f32>,    // [16, 3, 128, 128]
    pub swap_latent_batch: CudaSlice<f32>, // [16, 512]
    pub swap_latent_gpu: CudaSlice<f32>,   // [16, 512] — replicated latent for IoBinding input

    // Masks
    pub mask_128: CudaSlice<f32>,                // [128, 128]
    pub mask_128_tmp: CudaSlice<f32>,            // [128, 128] blur scratch
    pub mask_256: CudaSlice<f32>,                // [256, 256] — Occluder output
    pub mask_256_tmp: CudaSlice<f32>,            // [256, 256] morphology scratch
    pub mask_learned_128: CudaSlice<f32>,        // [128, 128] learned-mask composition
    pub parser_logits: CudaSlice<f32>,           // [19, 512, 512]
    pub parser_classes: CudaSlice<u8>,           // [512, 512]
    pub parser_mask_512: CudaSlice<f32>,         // Final parser mask
    pub parser_attribute_512: CudaSlice<f32>,    // Current class mask
    pub parser_tmp_512: CudaSlice<f32>,          // Morphology/blur scratch
    pub semantic_previous_eyes: CudaSlice<f32>,  // Temporal eye mask
    pub semantic_previous_mouth: CudaSlice<f32>, // Temporal mouth mask
    pub semantic_stats: CudaSlice<f32>,          // Mask and RGB reduction totals
    pub semantic_count: CudaSlice<u32>,          // Raw selected-class pixel count
    pub semantic_eyes_valid: CudaSlice<u32>,
    pub semantic_mouth_valid: CudaSlice<u32>,
    pub dfm_morph: CudaSlice<f32>,
    pub dfm_rct_stats: CudaSlice<f32>,
    pub mask_512: CudaSlice<f32>, // [512, 512] — final face mask

    // Blur kernel weights (uploaded once per param change)
    pub blur_kernel: CudaSlice<f32>, // [MAX_KS]
    pub blur_ks_current: u32,
    pub blur_sigma_current: f32,

    // Pre-computed
    pub emap: CudaSlice<f32>, // [512*512]

    // Pre-allocated host staging buffers (reused to avoid per-frame Vec alloc)
    pub host_det_input: Vec<f32>,  // [3*640*640]
    pub host_det_output: Vec<f32>, // [1*20*8400] — YoloFace output
    pub host_face_112: Vec<f32>,   // [3*112*112]
    pub host_embedding: Vec<f32>,  // [512] — ArcFace embedding
    pub host_swap_tiles: Vec<f32>, // [16*3*128*128]

    // Current frame dimensions
    pub width: u32,
    pub height: u32,
}

/// Max Gaussian blur kernel taps.
pub const MAX_BLUR_KS: usize = 201;

impl GpuWorkspace {
    /// Allocate all GPU buffers. Called once at startup.
    pub fn new(stream: &Arc<CudaStream>) -> Result<Self, DriverError> {
        let detect_size = 3 * 640 * 640;
        let face_112_size = 3 * 112 * 112;
        let max_swap_size = MAX_SWAP_DIM * 128;
        let face_256_size = 3 * max_swap_size * max_swap_size;
        let face_512_size = 3 * 512 * 512;
        let max_swap_tiles = MAX_SWAP_DIM * MAX_SWAP_DIM;
        let swap_batch_size = max_swap_tiles * 3 * 128 * 128;

        // YoloFace output: 1 * 20 * 8400 = 168000 for 640 input
        let det_output_size = 20 * 8400;

        Ok(Self {
            detect_input: stream.alloc_zeros::<f32>(detect_size)?,
            face_112_batch: stream.alloc_zeros::<f32>(MAX_FACES * face_112_size)?,
            arcface_embeddings_batch: stream.alloc_zeros::<f32>(MAX_FACES * 512)?,
            detect_output: stream.alloc_zeros::<f32>(det_output_size)?,

            face_112: stream.alloc_zeros::<f32>(face_112_size)?,
            face_256: stream.alloc_zeros::<f32>(face_256_size)?,
            face_512: stream.alloc_zeros::<f32>(face_512_size)?,
            face_512_original: stream.alloc_zeros::<f32>(face_512_size)?,
            face_512_pre_restorer: stream.alloc_zeros::<f32>(face_512_size)?,
            face_512_scratch: stream.alloc_zeros::<f32>(face_512_size)?,
            restorer_256_input: stream.alloc_zeros::<f32>(3 * 256 * 256)?,
            restorer_256_output: stream.alloc_zeros::<f32>(3 * 256 * 256)?,
            restorer_cache: stream.alloc_zeros::<f32>(face_512_size)?,
            restorer_cache_valid: false,
            restorer_frame: 0,

            arcface_embedding: stream.alloc_zeros::<f32>(512)?,

            swap_batch_in: stream.alloc_zeros::<f32>(swap_batch_size)?,
            swap_batch_out: stream.alloc_zeros::<f32>(swap_batch_size)?,
            swap_latent_batch: stream.alloc_zeros::<f32>(max_swap_tiles * 512)?,
            swap_latent_gpu: stream.alloc_zeros::<f32>(max_swap_tiles * 512)?,

            mask_128: stream.alloc_zeros::<f32>(128 * 128)?,
            mask_128_tmp: stream.alloc_zeros::<f32>(128 * 128)?,
            mask_256: stream.alloc_zeros::<f32>(256 * 256)?,
            mask_256_tmp: stream.alloc_zeros::<f32>(256 * 256)?,
            mask_learned_128: stream.alloc_zeros::<f32>(128 * 128)?,
            parser_logits: stream.alloc_zeros::<f32>(19 * 512 * 512)?,
            parser_classes: stream.alloc_zeros::<u8>(512 * 512)?,
            parser_mask_512: stream.alloc_zeros::<f32>(512 * 512)?,
            parser_attribute_512: stream.alloc_zeros::<f32>(512 * 512)?,
            parser_tmp_512: stream.alloc_zeros::<f32>(512 * 512)?,
            semantic_previous_eyes: stream.alloc_zeros::<f32>(512 * 512)?,
            semantic_previous_mouth: stream.alloc_zeros::<f32>(512 * 512)?,
            semantic_stats: stream.alloc_zeros::<f32>(8)?,
            semantic_count: stream.alloc_zeros::<u32>(1)?,
            semantic_eyes_valid: stream.alloc_zeros::<u32>(1)?,
            semantic_mouth_valid: stream.alloc_zeros::<u32>(1)?,
            dfm_morph: stream.alloc_zeros::<f32>(1)?,
            dfm_rct_stats: stream.alloc_zeros::<f32>(12)?,
            mask_512: stream.alloc_zeros::<f32>(512 * 512)?,

            blur_kernel: stream.alloc_zeros::<f32>(MAX_BLUR_KS)?,
            host_face_112_batch: vec![0.0f32; MAX_FACES * face_112_size],
            host_embeddings_batch: vec![0.0f32; MAX_FACES * 512],
            blur_ks_current: 0,
            blur_sigma_current: -1.0,

            emap: stream.alloc_zeros::<f32>(512 * 512)?,

            host_det_input: vec![0.0f32; 3 * 640 * 640],
            host_det_output: vec![0.0f32; det_output_size],
            host_face_112: vec![0.0f32; 3 * 112 * 112],
            host_embedding: vec![0.0f32; 512],
            host_swap_tiles: vec![0.0f32; swap_batch_size],

            width: 0,
            height: 0,
        })
    }
}

/// Frame-sized buffers (held separately from the scratch workspace so borrows
/// can be split).
pub struct FrameBuffers {
    pub u8_in: CudaSlice<u8>,     // [H, W, 3]
    pub u8_out: CudaSlice<u8>,    // [H, W, 3]
    pub chw: CudaSlice<f32>,      // [3, H, W]
    pub chw_last: CudaSlice<f32>, // [3, H, W] snapshot for Find Faces
}

impl FrameBuffers {
    pub fn new(stream: &Arc<CudaStream>) -> Result<Self, DriverError> {
        let frame_f32 = 3 * MAX_HEIGHT * MAX_WIDTH;
        let frame_u8 = 3 * MAX_HEIGHT * MAX_WIDTH;
        Ok(Self {
            u8_in: stream.alloc_zeros::<u8>(frame_u8)?,
            u8_out: stream.alloc_zeros::<u8>(frame_u8)?,
            chw: stream.alloc_zeros::<f32>(frame_f32)?,
            chw_last: stream.alloc_zeros::<f32>(frame_f32)?,
        })
    }
}

/// One slot in the live-frame ring. Device addresses never change, which is
/// required for stable I/O binding and future CUDA Graph capture.
pub struct FrameSlot {
    pub u8_in: CudaSlice<u8>,
    pub chw: CudaSlice<f32>,
    pub u8_out: CudaSlice<u8>,
    pub host_out: PinnedHostSlice<u8>,
}

/// Triple-buffered live path. Slots rotate instead of allocating CUDA and
/// host staging memory for every camera frame.
pub struct FrameRing {
    slots: Vec<FrameSlot>,
    cursor: usize,
    max_pixels: usize,
}

impl FrameRing {
    pub const DEFAULT_CAPACITY: usize = 3;

    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        capacity: usize,
    ) -> Result<Self, DriverError> {
        let max_pixels = MAX_WIDTH * MAX_HEIGHT;
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            // SAFETY: u8 has no invalid bit patterns. Pinned memory is owned by
            // the slot and released only after all recorded copies complete.
            let host_out = unsafe { ctx.alloc_pinned::<u8>(max_pixels * 3)? };
            slots.push(FrameSlot {
                u8_in: stream.alloc_zeros::<u8>(max_pixels * 3)?,
                chw: stream.alloc_zeros::<f32>(max_pixels * 3)?,
                u8_out: stream.alloc_zeros::<u8>(max_pixels * 3)?,
                host_out,
            });
        }
        Ok(Self {
            slots,
            cursor: 0,
            max_pixels,
        })
    }

    pub fn acquire(&mut self, width: u32, height: u32) -> anyhow::Result<&mut FrameSlot> {
        let pixels = width as usize * height as usize;
        anyhow::ensure!(
            pixels <= self.max_pixels,
            "frame {width}x{height} exceeds ring capacity"
        );
        let idx = self.cursor;
        self.cursor = (self.cursor + 1) % self.slots.len();
        Ok(&mut self.slots[idx])
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
}
