//! Shared Rust-native engine used by both photo and webcam live paths.

mod atomic;

pub use atomic::{AtomicLiveEngine, FaceAssignmentPaths, LiveShadowBuilder};

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use image::GenericImageView;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::config::parameters::{FaceSwapParams, RestorerSize, SwapperModel};
use crate::config::settings::{DetectorModel, ExecutionProvider};
use crate::engine::{BuildCancellation, EngineSpec, ModelArtifact, ModelRole};
use crate::gpu::ops::GpuOps;
use crate::models::live_catalog::{CANONICAL_SWAPPER_FILENAME, validate_model_file};
use crate::models::manager::ModelManager;
use crate::models::registry::find_model;
use crate::pipeline::dfm::DfmContract;
use crate::pipeline::face_detector::{FaceDetector, FaceDetectorBackend};
use crate::pipeline::face_landmark::LandmarkModel;
use crate::pipeline::face_recognizer::FaceRecognizer;
use crate::pipeline::frame_enhancer::FrameEnhancer;
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
    detector: FaceDetector,
    workspace: GpuWorkspace,
    source_faces: Vec<SourceFace>,
    params: FaceSwapParams,
    enhancer: Option<FrameEnhancer>,
    dfm: Option<DfmContract>,
}

pub(crate) struct ResolvedFaceAssignment {
    pub source_path: PathBuf,
    pub target_path: Option<PathBuf>,
    pub similarity_threshold: f32,
}

pub fn output_dimensions(
    params: &FaceSwapParams,
    width: u32,
    height: u32,
) -> anyhow::Result<(u32, u32)> {
    if !params.enhancer_enabled {
        return Ok((width, height));
    }
    let scale = params.enhancer_model.scale();
    Ok((
        width
            .checked_mul(scale)
            .ok_or_else(|| anyhow::anyhow!("enhanced output width overflows"))?,
        height
            .checked_mul(scale)
            .ok_or_else(|| anyhow::anyhow!("enhanced output height overflows"))?,
    ))
}

pub fn build_live_spec(
    models_dir: &Path,
    identity_path: &Path,
    params: FaceSwapParams,
    provider: ExecutionProvider,
    detector: DetectorModel,
    device_id: i32,
) -> anyhow::Result<EngineSpec> {
    let mut models = BTreeMap::new();
    let (detector_name, detector_filename) = match detector {
        DetectorModel::YoloFace8n => ("YoloFace8n", "yoloface_8n.onnx"),
        DetectorModel::RetinaFace => ("RetinaFace", "det_10g.onnx"),
        DetectorModel::Scrfd2_5g => ("SCRFD2.5g", "scrfd_2.5g_bnkps.onnx"),
    };
    insert_artifact(
        &mut models,
        models_dir,
        ModelRole::Detector,
        detector_name,
        detector_filename,
    )?;
    if params.landmark_enabled {
        insert_artifact(
            &mut models,
            models_dir,
            ModelRole::Landmark,
            params.landmark_mode.registry_name(),
            params.landmark_mode.filename(),
        )?;
    }
    if (params.restorer_enabled
        && params.restorer_alignment == crate::config::parameters::RestorerAlignment::Reference)
        || (params.restorer2_enabled
            && params.restorer2_alignment
                == crate::config::parameters::RestorerAlignment::Reference)
    {
        insert_artifact(
            &mut models,
            models_dir,
            ModelRole::RestorerLandmark,
            "FaceLandmark5",
            "res50.onnx",
        )?;
    }
    match params.swapper_model {
        SwapperModel::Inswapper128 => {
            for (role, logical_name, filename) in [
                (
                    ModelRole::Recognizer,
                    "Inswapper128ArcFace",
                    "w600k_r50.onnx",
                ),
                (
                    ModelRole::Swapper,
                    "Inswapper128",
                    CANONICAL_SWAPPER_FILENAME,
                ),
                (ModelRole::Emap, "InswapperEMap", "emap.bin"),
            ] {
                insert_artifact(&mut models, models_dir, role, logical_name, filename)?;
            }
        }
        SwapperModel::Dfm => {
            let root = dfm_root(models_dir);
            let logical_name = Path::new(&params.dfm_model)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| anyhow::anyhow!("invalid DFM model filename"))?;
            insert_artifact(
                &mut models,
                &root,
                ModelRole::Dfm,
                logical_name,
                &params.dfm_model,
            )?;
        }
    }

    if params.restorer_enabled {
        let (name, filename) = match params.restorer_size {
            RestorerSize::Gpen256 => ("GPENBFR256", "GPEN-BFR-256.onnx"),
            RestorerSize::Gpen512 => ("GPENBFR512", "GPEN-BFR-512.onnx"),
            RestorerSize::Gpen1024 => anyhow::bail!("GPEN-1024 is excluded from the runtime"),
        };
        insert_artifact(&mut models, models_dir, ModelRole::Restorer, name, filename)?;
    }
    if params.restorer2_enabled {
        let (name, filename) = match params.restorer2_size {
            RestorerSize::Gpen256 => ("GPENBFR256", "GPEN-BFR-256.onnx"),
            RestorerSize::Gpen512 => ("GPENBFR512", "GPEN-BFR-512.onnx"),
            RestorerSize::Gpen1024 => anyhow::bail!("GPEN-1024 is excluded from the runtime"),
        };
        insert_artifact(
            &mut models,
            models_dir,
            ModelRole::Restorer2,
            name,
            filename,
        )?;
    }
    if params.enhancer_enabled {
        let model_name = params.enhancer_model.registry_name();
        let artifact = find_model(model_name)
            .ok_or_else(|| anyhow::anyhow!("enhancer model {model_name} is not registered"))?;
        insert_artifact(
            &mut models,
            models_dir,
            ModelRole::Enhancer,
            model_name,
            artifact.filename,
        )?;
    }
    for (enabled, role, name, filename) in [
        (
            params.occluder_enabled,
            ModelRole::Occluder,
            "Occluder",
            "occluder.onnx",
        ),
        (
            params.xseg_enabled,
            ModelRole::Xseg,
            "XSeg",
            "XSeg_model.onnx",
        ),
        (
            params.faceparser_enabled,
            ModelRole::FaceParser,
            "FaceParser",
            "faceparser_resnet34.onnx",
        ),
    ] {
        if enabled {
            insert_artifact(&mut models, models_dir, role, name, filename)?;
        }
    }

    let spec = EngineSpec {
        provider,
        device_id,
        detector,
        identity_sha256: sha256_file(identity_path)?,
        assignments: Vec::new(),
        models,
        params,
    };
    spec.validate()?;
    Ok(spec)
}

fn insert_artifact(
    models: &mut BTreeMap<ModelRole, ModelArtifact>,
    root: &Path,
    role: ModelRole,
    logical_name: &str,
    filename: &str,
) -> anyhow::Result<()> {
    models.insert(
        role,
        ModelArtifact {
            logical_name: logical_name.to_owned(),
            filename: filename.to_owned(),
            sha256: sha256_file(&root.join(filename))?,
        },
    );
    Ok(())
}

fn dfm_root(models_dir: &Path) -> std::path::PathBuf {
    models_dir
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("dfms")
}

fn artifact_path(models_dir: &Path, role: ModelRole, filename: &str) -> std::path::PathBuf {
    if role == ModelRole::Dfm {
        dfm_root(models_dir).join(filename)
    } else {
        models_dir.join(filename)
    }
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn required_artifact(spec: &EngineSpec, role: ModelRole) -> anyhow::Result<&ModelArtifact> {
    spec.models
        .get(&role)
        .ok_or_else(|| anyhow::anyhow!("engine generation is missing {}", role.as_str()))
}

fn restorer_session_name(params: &FaceSwapParams) -> anyhow::Result<Option<&'static str>> {
    if !params.restorer_enabled {
        return Ok(None);
    }
    match params.restorer_size {
        RestorerSize::Gpen256 => Ok(Some("GPENBFR256")),
        RestorerSize::Gpen512 => Ok(Some("GPENBFR512")),
        RestorerSize::Gpen1024 => anyhow::bail!("GPEN-1024 is excluded from the runtime"),
    }
}

fn restorer2_session_name(params: &FaceSwapParams) -> anyhow::Result<Option<&'static str>> {
    if !params.restorer2_enabled {
        return Ok(None);
    }
    match params.restorer2_size {
        RestorerSize::Gpen256 => Ok(Some("GPENBFR256_2")),
        RestorerSize::Gpen512 => Ok(Some("GPENBFR512_2")),
        RestorerSize::Gpen1024 => anyhow::bail!("GPEN-1024 is excluded from the runtime"),
    }
}

fn ensure_build_active(cancellation: Option<&BuildCancellation>) -> anyhow::Result<()> {
    anyhow::ensure!(
        !cancellation.is_some_and(BuildCancellation::is_cancelled),
        "shadow build cancelled"
    );
    Ok(())
}

fn identity_embedding_gpu(
    path: &Path,
    detector: &FaceDetector,
    manager: &mut ModelManager,
    gpu: &GpuOps,
    workspace: &mut GpuWorkspace,
    params: &FaceSwapParams,
) -> anyhow::Result<Vec<f32>> {
    let identity = image::open(path)?;
    let (width, height) = identity.dimensions();
    let rgb = identity.to_rgb8();
    let input = gpu.upload_u8(rgb.as_raw())?;
    let mut chw = gpu.alloc_zeros(3 * width as usize * height as usize)?;
    gpu.hwc_u8_to_chw_f32(&input, &mut chw, height, width)?;
    let (faces, _) = detector.detect_gpu(manager, gpu, &chw, workspace, height, width)?;
    let face = faces
        .first()
        .ok_or_else(|| anyhow::anyhow!("No face in identity image {}", path.display()))?;
    let refined = if params.landmark_enabled {
        LandmarkModel::from(params.landmark_mode).detect_gpu(
            manager,
            gpu,
            workspace,
            &chw,
            height,
            width,
            face.bbox,
            &face.kps_5,
            params.landmark_from_points,
            params.landmark_score,
        )?
    } else {
        None
    };
    let keypoints = refined
        .as_ref()
        .filter(|landmarks| landmarks.is_preferred_to(face.score))
        .map_or(face.kps_5, |landmarks| landmarks.five);
    FaceRecognizer::recognize_gpu(manager, gpu, &chw, height, width, &keypoints, workspace)
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
        Self::new_configured(
            gpu,
            models_dir,
            identity_path,
            params,
            provider,
            DetectorModel::YoloFace8n,
            0,
            stream,
        )
    }

    pub fn new_configured(
        gpu: Arc<GpuOps>,
        models_dir: &Path,
        identity_path: &Path,
        params: FaceSwapParams,
        provider: ExecutionProvider,
        detector_model: DetectorModel,
        device_id: i32,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let spec = build_live_spec(
            models_dir,
            identity_path,
            params,
            provider,
            detector_model,
            device_id,
        )?;
        Self::new_from_spec_inner(
            gpu,
            models_dir,
            identity_path,
            &[],
            &spec,
            stream,
            false,
            None,
        )
    }

    /// Build a fully content-verified generation before it becomes eligible for activation.
    pub fn new_from_spec(
        gpu: Arc<GpuOps>,
        models_dir: &Path,
        identity_path: &Path,
        spec: &EngineSpec,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            spec.assignments.is_empty(),
            "multi-face generations require the atomic identity catalog"
        );
        Self::new_from_spec_inner(
            gpu,
            models_dir,
            identity_path,
            &[],
            spec,
            stream,
            true,
            None,
        )
    }

    pub(crate) fn new_from_spec_assignments_cancellable(
        gpu: Arc<GpuOps>,
        models_dir: &Path,
        identity_path: &Path,
        assignments: &[ResolvedFaceAssignment],
        spec: &EngineSpec,
        stream: &Arc<CudaStream>,
        cancellation: &BuildCancellation,
    ) -> anyhow::Result<Self> {
        Self::new_from_spec_inner(
            gpu,
            models_dir,
            identity_path,
            assignments,
            spec,
            stream,
            true,
            Some(cancellation),
        )
    }

    fn new_from_spec_inner(
        gpu: Arc<GpuOps>,
        models_dir: &Path,
        identity_path: &Path,
        assignments: &[ResolvedFaceAssignment],
        spec: &EngineSpec,
        stream: &Arc<CudaStream>,
        verify_files: bool,
        cancellation: Option<&BuildCancellation>,
    ) -> anyhow::Result<Self> {
        ensure_build_active(cancellation)?;
        spec.validate()?;
        if verify_files {
            validate_model_file(identity_path, &spec.identity_sha256)?;
            anyhow::ensure!(
                assignments.len() == spec.assignments.len(),
                "resolved face assignments do not match the generation spec"
            );
            for (resolved, assignment) in assignments.iter().zip(&spec.assignments) {
                validate_model_file(&resolved.source_path, &assignment.source_identity_sha256)?;
                match (&resolved.target_path, &assignment.target_identity_sha256) {
                    (Some(path), Some(digest)) => validate_model_file(path, digest)?,
                    (None, None) => {}
                    _ => anyhow::bail!("resolved target identity does not match generation spec"),
                }
                ensure_build_active(cancellation)?;
            }
            ensure_build_active(cancellation)?;
            for (role, artifact) in &spec.models {
                validate_model_file(
                    &artifact_path(models_dir, *role, &artifact.filename),
                    &artifact.sha256,
                )?;
                ensure_build_active(cancellation)?;
            }
        }

        let mut manager = ModelManager::with_execution(models_dir, spec.provider, spec.device_id);
        manager.set_compute_stream(stream.cu_stream() as *mut ());
        let detector_artifact = required_artifact(spec, ModelRole::Detector)?;
        match spec.detector {
            DetectorModel::YoloFace8n => manager.load("YoloFace8n", &detector_artifact.filename)?,
            DetectorModel::RetinaFace => manager.load("RetinaFace", &detector_artifact.filename)?,
            DetectorModel::Scrfd2_5g => manager.load("SCRFD2.5g", &detector_artifact.filename)?,
        }
        ensure_build_active(cancellation)?;
        if spec.params.landmark_enabled {
            let artifact = required_artifact(spec, ModelRole::Landmark)?;
            manager.load(
                spec.params.landmark_mode.registry_name(),
                &artifact.filename,
            )?;
            ensure_build_active(cancellation)?;
        }
        if let Some(artifact) = spec.models.get(&ModelRole::RestorerLandmark)
            && manager.get("FaceLandmark5").is_none()
        {
            manager.load("FaceLandmark5", &artifact.filename)?;
            ensure_build_active(cancellation)?;
        }
        let dfm = match spec.params.swapper_model {
            SwapperModel::Inswapper128 => {
                manager.load(
                    "Inswapper128ArcFace",
                    &required_artifact(spec, ModelRole::Recognizer)?.filename,
                )?;
                ensure_build_active(cancellation)?;
                manager.load(
                    "Inswapper128",
                    &required_artifact(spec, ModelRole::Swapper)?.filename,
                )?;
                ensure_build_active(cancellation)?;
                manager.load_emap_file(&required_artifact(spec, ModelRole::Emap)?.filename)?;
                ensure_build_active(cancellation)?;
                None
            }
            SwapperModel::Dfm => {
                let artifact = required_artifact(spec, ModelRole::Dfm)?;
                manager.load_path(
                    "DFM",
                    &artifact_path(models_dir, ModelRole::Dfm, &artifact.filename),
                )?;
                ensure_build_active(cancellation)?;
                Some(DfmContract::from_session(manager.get("DFM").ok_or_else(
                    || anyhow::anyhow!("DFM session was not loaded"),
                )?)?)
            }
        };

        for (role, session_name) in [
            (ModelRole::Restorer, restorer_session_name(&spec.params)?),
            (ModelRole::Restorer2, restorer2_session_name(&spec.params)?),
            (ModelRole::Occluder, Some("Occluder")),
            (ModelRole::Xseg, Some("XSeg")),
            (ModelRole::FaceParser, Some("FaceParser")),
        ] {
            if let (Some(artifact), Some(session_name)) = (spec.models.get(&role), session_name) {
                manager.load(session_name, &artifact.filename)?;
                ensure_build_active(cancellation)?;
            }
        }

        let detector = FaceDetector::configured(
            spec.detector,
            spec.params.detector_score,
            spec.params.max_faces,
        );
        let mut workspace = GpuWorkspace::new(stream)?;
        ensure_build_active(cancellation)?;
        let source_faces = if spec.params.swapper_model == SwapperModel::Inswapper128 {
            let legacy;
            let resolved = if assignments.is_empty() {
                legacy = [ResolvedFaceAssignment {
                    source_path: identity_path.to_path_buf(),
                    target_path: None,
                    similarity_threshold: spec.params.similarity_threshold,
                }];
                &legacy[..]
            } else {
                assignments
            };
            let mut sources = Vec::with_capacity(resolved.len());
            for assignment in resolved {
                let source_embedding = identity_embedding_gpu(
                    &assignment.source_path,
                    &detector,
                    &mut manager,
                    &gpu,
                    &mut workspace,
                    &spec.params,
                )?;
                let target_embedding = match &assignment.target_path {
                    Some(path) => Some(identity_embedding_gpu(
                        path,
                        &detector,
                        &mut manager,
                        &gpu,
                        &mut workspace,
                        &spec.params,
                    )?),
                    None => None,
                };
                let emap = manager
                    .emap
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Inswapper emap is not loaded"))?;
                sources.push(SourceFace {
                    target_embedding,
                    latent: FaceRecognizer::calc_latent(&source_embedding, emap),
                    threshold: assignment.similarity_threshold,
                });
                ensure_build_active(cancellation)?;
            }
            sources
        } else {
            Vec::new()
        };
        let enhancer = if spec.params.enhancer_enabled {
            let artifact = required_artifact(spec, ModelRole::Enhancer)?;
            let enhancer = FrameEnhancer::new_with_filename(
                Arc::clone(&gpu),
                models_dir,
                spec.provider,
                spec.device_id,
                spec.params.enhancer_model,
                &artifact.filename,
            )?;
            ensure_build_active(cancellation)?;
            Some(enhancer)
        } else {
            None
        };

        Ok(Self {
            gpu,
            manager,
            detector,
            workspace,
            source_faces,
            params: spec.params.clone(),
            enhancer,
            dfm,
        })
    }

    pub fn process_chw(
        &mut self,
        frame: &mut CudaSlice<f32>,
        height: u32,
        width: u32,
    ) -> anyhow::Result<FrameResult> {
        let turns = if self.params.manual_rotation_enabled {
            u32::from(self.params.manual_rotation_angle / 90) & 3
        } else {
            0
        };
        if turns == 0 {
            return process_frame_gpu(
                &self.gpu,
                &mut self.manager,
                &self.detector,
                frame,
                height,
                width,
                &mut self.workspace,
                &self.source_faces,
                &self.params,
                self.dfm,
            );
        }

        let (rotated_height, rotated_width) = if turns.is_multiple_of(2) {
            (height, width)
        } else {
            (width, height)
        };
        let mut rotated = self.gpu.alloc_zeros(3 * height as usize * width as usize)?;
        self.gpu
            .rotate_quadrants(frame, &mut rotated, height, width, turns)?;
        let result = process_frame_gpu(
            &self.gpu,
            &mut self.manager,
            &self.detector,
            &mut rotated,
            rotated_height,
            rotated_width,
            &mut self.workspace,
            &self.source_faces,
            &self.params,
            self.dfm,
        )?;
        self.gpu.rotate_quadrants(
            &rotated,
            frame,
            rotated_height,
            rotated_width,
            (4 - turns) & 3,
        )?;
        Ok(result)
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
        let (output_width, output_height) = output_dimensions(&self.params, width, height)?;
        let enhanced = if let Some(enhancer) = &mut self.enhancer {
            let output_elements = 3usize
                .checked_mul(output_width as usize)
                .and_then(|elements| elements.checked_mul(output_height as usize))
                .ok_or_else(|| anyhow::anyhow!("enhanced output allocation overflows"))?;
            let mut output = self.gpu.alloc_zeros(output_elements)?;
            enhancer.enhance_into(&chw, &mut output, width, height, self.params.enhancer_blend)?;
            Some(output)
        } else {
            None
        };
        let output_chw = enhanced.as_ref().unwrap_or(&chw);
        let output_len = 3usize
            .checked_mul(output_width as usize)
            .and_then(|elements| elements.checked_mul(output_height as usize))
            .ok_or_else(|| anyhow::anyhow!("RGB output allocation overflows"))?;
        let mut output_gpu = self.gpu.alloc_zeros_u8(output_len)?;
        self.gpu
            .chw_f32_to_hwc_u8(output_chw, &mut output_gpu, output_height, output_width)?;
        let data = self.gpu.download_u8(&output_gpu)?;
        Ok(ProcessedRgb {
            data,
            width: output_width,
            height: output_height,
            faces_detected: result.faces_detected,
            faces_swapped: result.faces_swapped,
        })
    }
}
