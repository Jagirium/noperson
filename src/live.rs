//! Shared Rust-native engine used by both photo and webcam live paths.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use image::GenericImageView;
use thiserror::Error;

use crate::config::parameters::FaceSwapParams;
use crate::config::settings::ExecutionProvider;
use crate::gpu::ops::GpuOps;
use crate::models::live_catalog::CANONICAL_SWAPPER_FILENAME;
use crate::models::manager::ModelManager;
use crate::pipeline::face_detector::YoloFaceDetector;
use crate::pipeline::face_recognizer::FaceRecognizer;
use crate::pipeline::frame_processor::{FrameResult, SourceFace, process_frame_gpu};
use crate::pipeline::workspace::GpuWorkspace;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LiveFrameError {
    #[error("RGB frame length mismatch for {width}x{height}: expected {expected}, got {actual}")]
    LengthMismatch {
        width: u32,
        height: u32,
        expected: usize,
        actual: usize,
    },
}

pub fn validate_rgb_frame(data: &[u8], width: u32, height: u32) -> Result<(), LiveFrameError> {
    let expected = width as usize * height as usize * 3;
    if data.len() != expected {
        return Err(LiveFrameError::LengthMismatch {
            width,
            height,
            expected,
            actual: data.len(),
        });
    }
    Ok(())
}

pub struct ProcessedRgb {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub faces_detected: usize,
    pub faces_swapped: usize,
}

pub struct LiveEngine {
    gpu: Arc<GpuOps>,
    manager: ModelManager,
    detector: YoloFaceDetector,
    workspace: GpuWorkspace,
    source_faces: Vec<SourceFace>,
    params: FaceSwapParams,
}

impl LiveEngine {
    pub fn new(
        gpu: Arc<GpuOps>,
        models_dir: &Path,
        identity_path: &Path,
        params: FaceSwapParams,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        Self::new_with_provider(
            gpu,
            models_dir,
            identity_path,
            params,
            ExecutionProvider::Cuda,
            stream,
        )
    }

    pub fn new_with_provider(
        gpu: Arc<GpuOps>,
        models_dir: &Path,
        identity_path: &Path,
        params: FaceSwapParams,
        provider: ExecutionProvider,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let mut manager = ModelManager::with_provider(models_dir, provider);
        manager.set_compute_stream(stream.cu_stream() as *mut ());
        manager.load("YoloFace8n", "yoloface_8n.onnx")?;
        manager.load("Inswapper128ArcFace", "w600k_r50.onnx")?;
        manager.load("Inswapper128", CANONICAL_SWAPPER_FILENAME)?;
        manager.load_emap(CANONICAL_SWAPPER_FILENAME)?;
        if params.restorer_enabled {
            manager.load("GPENBFR512", "GPEN-BFR-512.onnx")?;
        }

        let detector = YoloFaceDetector::new(0.5);
        let mut workspace = GpuWorkspace::new(stream)?;
        let identity = image::open(identity_path)?;
        let (width, height) = identity.dimensions();
        let rgb = identity.to_rgb8();
        let input = gpu.upload_u8(rgb.as_raw())?;
        let mut chw = gpu.alloc_zeros(3 * width as usize * height as usize)?;
        gpu.hwc_u8_to_chw_f32(&input, &mut chw, height, width)?;
        let (faces, _) =
            detector.detect_gpu(&mut manager, &gpu, &chw, &mut workspace, height, width)?;
        let face = faces
            .first()
            .ok_or_else(|| anyhow::anyhow!("No face in identity image"))?;
        let embedding = FaceRecognizer::recognize_gpu(
            &mut manager,
            &gpu,
            &chw,
            height,
            width,
            &face.kps_5,
            &mut workspace,
        )?;
        let emap = manager
            .emap
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Inswapper emap is not loaded"))?;
        let latent = FaceRecognizer::calc_latent(&embedding, emap);
        let source_faces = vec![SourceFace {
            embedding,
            latent,
            threshold: 0.0,
        }];

        Ok(Self {
            gpu,
            manager,
            detector,
            workspace,
            source_faces,
            params,
        })
    }

    pub fn process_chw(
        &mut self,
        frame: &mut CudaSlice<f32>,
        height: u32,
        width: u32,
    ) -> anyhow::Result<FrameResult> {
        process_frame_gpu(
            &self.gpu,
            &mut self.manager,
            &self.detector,
            frame,
            height,
            width,
            &mut self.workspace,
            &self.source_faces,
            &self.params,
        )
    }

    pub fn process_rgb(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> anyhow::Result<ProcessedRgb> {
        validate_rgb_frame(data, width, height)?;
        let input = self.gpu.upload_u8(data)?;
        let mut chw = self.gpu.alloc_zeros(3 * width as usize * height as usize)?;
        self.gpu
            .hwc_u8_to_chw_f32(&input, &mut chw, height, width)?;
        let result = self.process_chw(&mut chw, height, width)?;
        let mut output_gpu = self.gpu.alloc_zeros_u8(data.len())?;
        self.gpu
            .chw_f32_to_hwc_u8(&chw, &mut output_gpu, height, width)?;
        let data = self.gpu.download_u8(&output_gpu)?;
        Ok(ProcessedRgb {
            data,
            width,
            height,
            faces_detected: result.faces_detected,
            faces_swapped: result.faces_swapped,
        })
    }
}
