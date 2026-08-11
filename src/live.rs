//! Shared Rust-native engine used by both photo and webcam live paths.

mod atomic;

pub use atomic::{
    AtomicLiveEngine, FaceAssignmentInputs, FaceAssignmentPaths, FaceIdentityInput,
    LiveShadowBuilder, embedding_blake3,
};

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use image::GenericImageView;
use thiserror::Error;

use crate::backend::{Buffer, ComputeOps, ComputeStream};
use crate::config::parameters::{FaceSwapParams, RestorerSize, SwapperModel};
use crate::config::settings::{DetectorModel, ExecutionProvider};
use crate::engine::{BuildCancellation, EngineSpec, ModelArtifact, ModelRole};
use crate::models::digest::file_blake3;
use crate::models::live_catalog::{CANONICAL_SWAPPER_FILENAME, validate_model_file};
use crate::models::manager::ModelManager;
use crate::models::registry::find_model;
use crate::pipeline::dfm::DfmContract;
use crate::pipeline::face_detector::{FaceDetector, FaceDetectorBackend};
use crate::pipeline::face_landmark::LandmarkModel;
use crate::pipeline::face_recognizer::FaceRecognizer;
use crate::pipeline::face_tracker::{TemporalFaceTracker, TrackerPolicy};
use crate::pipeline::frame_enhancer::FrameEnhancer;
use crate::pipeline::frame_processor::{
    AssignmentBackend, FrameResult, GenerationGpuState, SourceFace, process_frame_gpu_with_state,
};
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
    pub overlays: Vec<FaceOverlay>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FaceOverlay {
    /// Normalized [x1, y1, x2, y2] in the displayed output orientation.
    pub bbox: [f32; 4],
    /// Normalized five-point landmarks in the displayed output orientation.
    pub kps_5: [[f32; 2]; 5],
}

#[derive(Debug, Clone)]
pub struct AnalyzedIdentity {
    pub embedding: Vec<f32>,
    pub bbox: [f32; 4],
    pub crop_rgb: Vec<u8>,
    pub crop_width: u32,
    pub crop_height: u32,
}

pub struct IdentityAnalyzer {
    gpu: Arc<ComputeOps>,
    manager: ModelManager,
    detector: FaceDetector,
    workspace: GpuWorkspace,
    params: FaceSwapParams,
}

impl IdentityAnalyzer {
    pub fn new(
        gpu: Arc<ComputeOps>,
        models_dir: &Path,
        params: FaceSwapParams,
        provider: ExecutionProvider,
        detector_model: DetectorModel,
        device_id: i32,
    ) -> anyhow::Result<Self> {
        let mut manager = ModelManager::with_execution(models_dir, provider, device_id);
        manager.set_compute_stream(gpu.stream.cu_stream() as *mut ())?;
        let (detector_name, detector_filename) = match detector_model {
            DetectorModel::YoloFace8n => ("YoloFace8n", "yoloface_8n.onnx"),
            DetectorModel::RetinaFace => ("RetinaFace", "det_10g.onnx"),
            DetectorModel::Scrfd2_5g => ("SCRFD2.5g", "scrfd_2.5g_bnkps.onnx"),
        };
        manager.load(detector_name, detector_filename)?;
        manager.load("Inswapper128ArcFace", "w600k_r50.onnx")?;
        if params.landmark_enabled {
            manager.load(
                params.landmark_mode.registry_name(),
                params.landmark_mode.filename(),
            )?;
        }
        let detector =
            FaceDetector::configured(detector_model, params.detector_score, params.max_faces);
        let workspace = GpuWorkspace::new(&gpu.stream)?;
        Ok(Self {
            gpu,
            manager,
            detector,
            workspace,
            params,
        })
    }

    pub fn analyze(&mut self, image_path: &Path) -> anyhow::Result<Vec<AnalyzedIdentity>> {
        let image = image::open(image_path)?.to_rgb8();
        self.analyze_rgb_image(image)
    }

    pub fn analyze_rgb(
        &mut self,
        rgb: &[u8],
        width: u32,
        height: u32,
    ) -> anyhow::Result<Vec<AnalyzedIdentity>> {
        validate_rgb_frame(rgb, width, height)?;
        let image = image::RgbImage::from_raw(width, height, rgb.to_vec())
            .ok_or_else(|| anyhow::anyhow!("invalid RGB analysis frame"))?;
        self.analyze_rgb_image(image)
    }

    fn analyze_rgb_image(
        &mut self,
        image: image::RgbImage,
    ) -> anyhow::Result<Vec<AnalyzedIdentity>> {
        let image = rotate_analysis_image(
            image,
            self.params.manual_rotation_enabled,
            self.params.manual_rotation_angle,
        );
        let (width, height) = image.dimensions();
        let input = self.gpu.upload_u8(image.as_raw())?;
        let mut chw = self.gpu.alloc_zeros(3 * width as usize * height as usize)?;
        self.gpu
            .hwc_u8_to_chw_f32(&input, &mut chw, height, width)?;
        let (faces, _) = if self.params.auto_rotation {
            self.detector.detect_gpu_auto_rotation(
                &mut self.manager,
                &self.gpu,
                &chw,
                &mut self.workspace,
                height,
                width,
            )?
        } else {
            self.detector.detect_gpu(
                &mut self.manager,
                &self.gpu,
                &chw,
                &mut self.workspace,
                height,
                width,
            )?
        };
        let mut analyzed = Vec::with_capacity(faces.len());
        for face in faces {
            let refined = if self.params.landmark_enabled {
                LandmarkModel::from(self.params.landmark_mode).detect_gpu(
                    &mut self.manager,
                    &self.gpu,
                    &mut self.workspace,
                    &chw,
                    height,
                    width,
                    face.bbox,
                    &face.kps_5,
                    self.params.landmark_from_points,
                    self.params.landmark_score,
                )?
            } else {
                None
            };
            let keypoints = refined
                .as_ref()
                .filter(|landmarks| landmarks.is_preferred_to(face.score))
                .map_or(face.kps_5, |landmarks| landmarks.five);
            let embedding = FaceRecognizer::recognize_gpu_with_similarity(
                &mut self.manager,
                &self.gpu,
                &chw,
                height,
                width,
                &keypoints,
                &mut self.workspace,
                self.params.similarity_type,
            )?
            .to_vec();
            let (crop_rgb, crop_width, crop_height) = crop_bbox(&image, face.bbox);
            analyzed.push(AnalyzedIdentity {
                embedding,
                bbox: face.bbox,
                crop_rgb,
                crop_width,
                crop_height,
            });
        }
        Ok(analyzed)
    }
}

fn rotate_analysis_image(image: image::RgbImage, enabled: bool, angle: u16) -> image::RgbImage {
    if !enabled {
        return image;
    }
    match (angle / 90) & 3 {
        1 => image::imageops::rotate90(&image),
        2 => image::imageops::rotate180(&image),
        3 => image::imageops::rotate270(&image),
        _ => image,
    }
}

pub fn analyze_image_identities(
    gpu: Arc<ComputeOps>,
    models_dir: &Path,
    image_path: &Path,
    params: &FaceSwapParams,
    provider: ExecutionProvider,
    detector_model: DetectorModel,
    device_id: i32,
) -> anyhow::Result<Vec<AnalyzedIdentity>> {
    IdentityAnalyzer::new(
        gpu,
        models_dir,
        params.clone(),
        provider,
        detector_model,
        device_id,
    )?
    .analyze(image_path)
}

fn crop_bbox(image: &image::RgbImage, bbox: [f32; 4]) -> (Vec<u8>, u32, u32) {
    let width = image.width();
    let height = image.height();
    let x1 = bbox[0].floor().max(0.0).min(width.saturating_sub(1) as f32) as u32;
    let y1 = bbox[1]
        .floor()
        .max(0.0)
        .min(height.saturating_sub(1) as f32) as u32;
    let x2 = bbox[2].ceil().max(x1 as f32 + 1.0).min(width as f32) as u32;
    let y2 = bbox[3].ceil().max(y1 as f32 + 1.0).min(height as f32) as u32;
    let crop_width = x2.saturating_sub(x1).max(1);
    let crop_height = y2.saturating_sub(y1).max(1);
    let crop = image::imageops::crop_imm(image, x1, y1, crop_width, crop_height).to_image();
    (crop.into_raw(), crop_width, crop_height)
}

pub struct LiveEngine {
    gpu: Arc<ComputeOps>,
    manager: ModelManager,
    detector: FaceDetector,
    face_tracker: TemporalFaceTracker,
    workspace: GpuWorkspace,
    source_faces: Vec<SourceFace>,
    generation_gpu_state: GenerationGpuState,
    params: FaceSwapParams,
    enhancer: Option<FrameEnhancer>,
    rotation_scratch: Option<Buffer<f32>>,
    rgb_input_scratch: Option<Buffer<u8>>,
    frame_scratch: Option<Buffer<f32>>,
    enhanced_scratch: Option<Buffer<f32>>,
    rgb_output_scratch: Option<Buffer<u8>>,
}

pub(crate) struct ResolvedFaceAssignment {
    pub source: ResolvedIdentity,
    pub target: Option<ResolvedIdentity>,
    pub similarity_threshold: f32,
    pub params: Option<FaceSwapParams>,
}

#[derive(Debug, Clone)]
pub(crate) enum ResolvedIdentity {
    Image(PathBuf),
    Embedding(Vec<f32>),
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
    build_live_spec_for_digest(
        models_dir,
        blake3_file(identity_path)?,
        params,
        provider,
        detector,
        device_id,
    )
}

pub fn build_live_spec_for_digest(
    models_dir: &Path,
    identity_blake3: String,
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
        identity_blake3,
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
            blake3: blake3_file(&root.join(filename))?,
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

fn blake3_file(path: &Path) -> anyhow::Result<String> {
    Ok(file_blake3(path)?)
}

fn required_artifact(spec: &EngineSpec, role: ModelRole) -> anyhow::Result<&ModelArtifact> {
    spec.models
        .get(&role)
        .ok_or_else(|| anyhow::anyhow!("engine generation is missing {}", role.as_str()))
}

fn dfm_session_name(artifact: &ModelArtifact) -> String {
    format!("DFM-{}", &artifact.blake3[..16])
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
    gpu: &ComputeOps,
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
    FaceRecognizer::recognize_gpu_with_similarity(
        manager,
        gpu,
        &chw,
        height,
        width,
        &keypoints,
        workspace,
        params.similarity_type,
    )
    .map(Vec::from)
}

fn resolved_identity_embedding(
    identity: &ResolvedIdentity,
    detector: &FaceDetector,
    manager: &mut ModelManager,
    gpu: &ComputeOps,
    workspace: &mut GpuWorkspace,
    params: &FaceSwapParams,
) -> anyhow::Result<Vec<f32>> {
    match identity {
        ResolvedIdentity::Image(path) => {
            identity_embedding_gpu(path, detector, manager, gpu, workspace, params)
        }
        ResolvedIdentity::Embedding(embedding) => {
            atomic::embedding_blake3(embedding)?;
            Ok(embedding.clone())
        }
    }
}

fn validate_resolved_identity(identity: &ResolvedIdentity, expected: &str) -> anyhow::Result<()> {
    let actual = match identity {
        ResolvedIdentity::Image(path) => blake3_file(path)?,
        ResolvedIdentity::Embedding(embedding) => atomic::embedding_blake3(embedding)?,
    };
    anyhow::ensure!(
        actual == expected,
        "identity BLAKE3 mismatch: expected {expected}, got {actual}"
    );
    Ok(())
}

fn apply_face_likeness(source_latent: &mut [f32], target_latent: &[f32], factor: f32) {
    debug_assert_eq!(source_latent.len(), target_latent.len());
    for (value, target_value) in source_latent.iter_mut().zip(target_latent) {
        *value -= factor * target_value;
    }
}

impl LiveEngine {
    pub fn set_tracking_policy(&mut self, policy: TrackerPolicy) {
        self.face_tracker.set_policy(policy);
    }

    pub fn reset_face_tracker(&mut self) {
        self.face_tracker.reset();
    }

    pub fn new(
        gpu: Arc<ComputeOps>,
        models_dir: &Path,
        identity_path: &Path,
        params: FaceSwapParams,
        stream: &Arc<ComputeStream>,
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
        gpu: Arc<ComputeOps>,
        models_dir: &Path,
        identity_path: &Path,
        params: FaceSwapParams,
        provider: ExecutionProvider,
        stream: &Arc<ComputeStream>,
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
        gpu: Arc<ComputeOps>,
        models_dir: &Path,
        identity_path: &Path,
        params: FaceSwapParams,
        provider: ExecutionProvider,
        detector_model: DetectorModel,
        device_id: i32,
        stream: &Arc<ComputeStream>,
    ) -> anyhow::Result<Self> {
        let spec = build_live_spec(
            models_dir,
            identity_path,
            params,
            provider,
            detector_model,
            device_id,
        )?;
        let identity = ResolvedIdentity::Image(identity_path.to_path_buf());
        Self::new_from_spec_inner(gpu, models_dir, &identity, &[], &spec, stream, false, None)
    }

    /// Build a fully content-verified generation before it becomes eligible for activation.
    pub fn new_from_spec(
        gpu: Arc<ComputeOps>,
        models_dir: &Path,
        identity_path: &Path,
        spec: &EngineSpec,
        stream: &Arc<ComputeStream>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            spec.assignments.is_empty(),
            "multi-face generations require the atomic identity catalog"
        );
        let identity = ResolvedIdentity::Image(identity_path.to_path_buf());
        Self::new_from_spec_inner(gpu, models_dir, &identity, &[], spec, stream, true, None)
    }

    pub(crate) fn new_from_spec_assignments_cancellable(
        gpu: Arc<ComputeOps>,
        models_dir: &Path,
        identity: &ResolvedIdentity,
        assignments: &[ResolvedFaceAssignment],
        spec: &EngineSpec,
        stream: &Arc<ComputeStream>,
        cancellation: &BuildCancellation,
    ) -> anyhow::Result<Self> {
        Self::new_from_spec_inner(
            gpu,
            models_dir,
            identity,
            assignments,
            spec,
            stream,
            true,
            Some(cancellation),
        )
    }

    fn new_from_spec_inner(
        gpu: Arc<ComputeOps>,
        models_dir: &Path,
        identity: &ResolvedIdentity,
        assignments: &[ResolvedFaceAssignment],
        spec: &EngineSpec,
        stream: &Arc<ComputeStream>,
        verify_files: bool,
        cancellation: Option<&BuildCancellation>,
    ) -> anyhow::Result<Self> {
        ensure_build_active(cancellation)?;
        spec.validate()?;
        if verify_files {
            validate_resolved_identity(identity, &spec.identity_blake3)?;
            anyhow::ensure!(
                assignments.len() == spec.assignments.len(),
                "resolved face assignments do not match the generation spec"
            );
            for (resolved, assignment) in assignments.iter().zip(&spec.assignments) {
                validate_resolved_identity(&resolved.source, &assignment.source_identity_blake3)?;
                match (&resolved.target, &assignment.target_identity_blake3) {
                    (Some(identity), Some(digest)) => validate_resolved_identity(identity, digest)?,
                    (None, None) => {}
                    _ => anyhow::bail!("resolved target identity does not match generation spec"),
                }
                ensure_build_active(cancellation)?;
            }
            ensure_build_active(cancellation)?;
            for (role, artifact) in &spec.models {
                validate_model_file(
                    &artifact_path(models_dir, *role, &artifact.filename),
                    &artifact.blake3,
                )?;
                ensure_build_active(cancellation)?;
            }
            for assignment in &spec.assignments {
                for (role, artifact) in &assignment.models {
                    validate_model_file(
                        &artifact_path(models_dir, *role, &artifact.filename),
                        &artifact.blake3,
                    )?;
                    ensure_build_active(cancellation)?;
                }
            }
        }

        let mut manager = ModelManager::with_execution(models_dir, spec.provider, spec.device_id);
        manager.set_compute_stream(stream.cu_stream() as *mut ())?;
        let detector_artifact = required_artifact(spec, ModelRole::Detector)?;
        match spec.detector {
            DetectorModel::YoloFace8n => manager.load("YoloFace8n", &detector_artifact.filename)?,
            DetectorModel::RetinaFace => manager.load("RetinaFace", &detector_artifact.filename)?,
            DetectorModel::Scrfd2_5g => manager.load("SCRFD2.5g", &detector_artifact.filename)?,
        }
        ensure_build_active(cancellation)?;
        let mut effective_models = Vec::with_capacity(spec.assignments.len().max(1));
        if spec.assignments.is_empty() {
            effective_models.push(spec.models.clone());
        } else {
            for assignment in &spec.assignments {
                let mut models = spec.models.clone();
                models.extend(assignment.models.clone());
                effective_models.push(models);
            }
        }
        let effective_params: Vec<&FaceSwapParams> = if assignments.is_empty() {
            vec![&spec.params]
        } else {
            assignments
                .iter()
                .map(|assignment| assignment.params.as_ref().unwrap_or(&spec.params))
                .collect()
        };
        let needs_inswapper = effective_params
            .iter()
            .any(|params| params.swapper_model == SwapperModel::Inswapper128);
        let needs_recognizer = needs_inswapper
            || assignments
                .iter()
                .any(|assignment| assignment.target.is_some());
        if needs_recognizer {
            let artifact = effective_models
                .iter()
                .find_map(|models| models.get(&ModelRole::Recognizer))
                .ok_or_else(|| anyhow::anyhow!("target matching requires a recognizer model"))?;
            manager.load("Inswapper128ArcFace", &artifact.filename)?;
            ensure_build_active(cancellation)?;
        }
        if needs_inswapper {
            let swapper = effective_models
                .iter()
                .find_map(|models| models.get(&ModelRole::Swapper))
                .ok_or_else(|| anyhow::anyhow!("Inswapper assignment requires a swapper model"))?;
            let emap = effective_models
                .iter()
                .find_map(|models| models.get(&ModelRole::Emap))
                .ok_or_else(|| anyhow::anyhow!("Inswapper assignment requires an emap"))?;
            manager.load("Inswapper128", &swapper.filename)?;
            manager.load_emap_file(&emap.filename)?;
            ensure_build_active(cancellation)?;
        }
        for (params, models) in effective_params.iter().zip(&effective_models) {
            if params.landmark_enabled {
                let artifact = models
                    .get(&ModelRole::Landmark)
                    .ok_or_else(|| anyhow::anyhow!("landmark assignment requires a model"))?;
                manager.load(params.landmark_mode.registry_name(), &artifact.filename)?;
                ensure_build_active(cancellation)?;
            }
            if let Some(artifact) = models.get(&ModelRole::RestorerLandmark) {
                manager.load("FaceLandmark5", &artifact.filename)?;
                ensure_build_active(cancellation)?;
            }
            if params.swapper_model == SwapperModel::Dfm {
                let artifact = models
                    .get(&ModelRole::Dfm)
                    .ok_or_else(|| anyhow::anyhow!("DFM assignment requires a model"))?;
                let session_name = dfm_session_name(artifact);
                manager.load_path(
                    &session_name,
                    &artifact_path(models_dir, ModelRole::Dfm, &artifact.filename),
                )?;
                ensure_build_active(cancellation)?;
            }
            for (role, session_name) in [
                (ModelRole::Restorer, restorer_session_name(params)?),
                (ModelRole::Restorer2, restorer2_session_name(params)?),
                (ModelRole::Occluder, Some("Occluder")),
                (ModelRole::Xseg, Some("XSeg")),
                (ModelRole::FaceParser, Some("FaceParser")),
            ] {
                if let (Some(artifact), Some(session_name)) = (models.get(&role), session_name) {
                    manager.load(session_name, &artifact.filename)?;
                    ensure_build_active(cancellation)?;
                }
            }
        }

        let detector = FaceDetector::configured(
            spec.detector,
            spec.params.detector_score,
            spec.params.max_faces,
        );
        let mut workspace = GpuWorkspace::new(stream)?;
        ensure_build_active(cancellation)?;
        let legacy;
        let resolved = if assignments.is_empty() {
            legacy = [ResolvedFaceAssignment {
                source: identity.clone(),
                target: None,
                similarity_threshold: spec.params.similarity_threshold,
                params: None,
            }];
            &legacy[..]
        } else {
            assignments
        };
        let mut source_faces = Vec::with_capacity(resolved.len());
        for (index, assignment) in resolved.iter().enumerate() {
            let face_params = assignment.params.as_ref().unwrap_or(&spec.params);
            let target_embedding = match &assignment.target {
                Some(identity) => Some(resolved_identity_embedding(
                    identity,
                    &detector,
                    &mut manager,
                    &gpu,
                    &mut workspace,
                    &spec.params,
                )?),
                None => None,
            };
            let backend = match face_params.swapper_model {
                SwapperModel::Inswapper128 => {
                    let source_embedding = resolved_identity_embedding(
                        &assignment.source,
                        &detector,
                        &mut manager,
                        &gpu,
                        &mut workspace,
                        &spec.params,
                    )?;
                    let emap = manager
                        .emap
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("Inswapper emap is not loaded"))?;
                    let mut latent = FaceRecognizer::calc_latent(&source_embedding, emap);
                    if face_params.face_likeness_enabled {
                        let target = target_embedding.as_ref().ok_or_else(|| {
                            anyhow::anyhow!("face likeness requires a target identity")
                        })?;
                        let target_latent = FaceRecognizer::calc_latent(target, emap);
                        apply_face_likeness(
                            &mut latent,
                            &target_latent,
                            face_params.face_likeness_factor,
                        );
                    }
                    AssignmentBackend::Inswapper { latent }
                }
                SwapperModel::Dfm => {
                    let artifact = effective_models[index]
                        .get(&ModelRole::Dfm)
                        .ok_or_else(|| anyhow::anyhow!("DFM assignment requires a model"))?;
                    let session_name = dfm_session_name(artifact);
                    let contract =
                        DfmContract::from_session(manager.get(&session_name).ok_or_else(
                            || anyhow::anyhow!("DFM session {session_name} was not loaded"),
                        )?)?;
                    AssignmentBackend::Dfm {
                        session_name,
                        contract,
                    }
                }
            };
            source_faces.push(SourceFace {
                target_embedding,
                backend,
                threshold: assignment.similarity_threshold,
                params: assignment.params.clone(),
            });
            ensure_build_active(cancellation)?;
        }
        let generation_gpu_state = GenerationGpuState::new(&gpu, &source_faces)?;
        ensure_build_active(cancellation)?;
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
            face_tracker: TemporalFaceTracker::new(TrackerPolicy::offline_recovery()),
            workspace,
            source_faces,
            generation_gpu_state,
            params: spec.params.clone(),
            enhancer,
            rotation_scratch: None,
            rgb_input_scratch: None,
            frame_scratch: None,
            enhanced_scratch: None,
            rgb_output_scratch: None,
        })
    }

    pub fn process_chw(
        &mut self,
        frame: &mut Buffer<f32>,
        height: u32,
        width: u32,
    ) -> anyhow::Result<FrameResult> {
        let turns = if self.params.manual_rotation_enabled {
            u32::from(self.params.manual_rotation_angle / 90) & 3
        } else {
            0
        };
        if turns == 0 {
            return process_frame_gpu_with_state(
                &self.gpu,
                &mut self.manager,
                &self.detector,
                frame,
                height,
                width,
                &mut self.workspace,
                &self.source_faces,
                &mut self.generation_gpu_state,
                &self.params,
                &mut self.face_tracker,
            );
        }

        let (rotated_height, rotated_width) = if turns.is_multiple_of(2) {
            (height, width)
        } else {
            (width, height)
        };
        let elements = 3 * height as usize * width as usize;
        let mut rotated = match self.rotation_scratch.take() {
            Some(buffer) if buffer.len() == elements => buffer,
            _ => self.gpu.alloc_zeros(elements)?,
        };
        self.gpu
            .rotate_quadrants(frame, &mut rotated, height, width, turns)?;
        let result = process_frame_gpu_with_state(
            &self.gpu,
            &mut self.manager,
            &self.detector,
            &mut rotated,
            rotated_height,
            rotated_width,
            &mut self.workspace,
            &self.source_faces,
            &mut self.generation_gpu_state,
            &self.params,
            &mut self.face_tracker,
        )?;
        self.gpu.rotate_quadrants(
            &rotated,
            frame,
            rotated_height,
            rotated_width,
            (4 - turns) & 3,
        )?;
        self.rotation_scratch = Some(rotated);
        Ok(result)
    }

    /// Process an already device-resident CHW frame and convert the final
    /// image directly into an NVENC-compatible pitch-linear NV12/P010 surface.
    pub fn process_chw_to_pitched_nv12(
        &mut self,
        frame: &mut Buffer<f32>,
        height: u32,
        width: u32,
        output_device_ptr: u64,
        output_pitch: u32,
        matrix: crate::io::native_video::ColorMatrix,
        range: crate::io::native_video::ColorRange,
        pixel_format: crate::io::native_video::PixelFormat,
    ) -> anyhow::Result<FrameResult> {
        let result = self.process_chw(frame, height, width)?;
        let (output_width, output_height) = output_dimensions(&self.params, width, height)?;
        let bytes_per_sample = if pixel_format == crate::io::native_video::PixelFormat::P010 {
            2
        } else {
            1
        };
        anyhow::ensure!(
            output_device_ptr != 0 && output_pitch >= output_width * bytes_per_sample,
            "NV12 output surface has invalid pointer or pitch"
        );

        let mut enhanced = if let Some(enhancer) = &mut self.enhancer {
            let output_elements = 3usize
                .checked_mul(output_width as usize)
                .and_then(|elements| elements.checked_mul(output_height as usize))
                .ok_or_else(|| anyhow::anyhow!("enhanced output allocation overflows"))?;
            let mut scratch = match self.enhanced_scratch.take() {
                Some(buffer) if buffer.len() == output_elements => buffer,
                _ => self.gpu.alloc_zeros(output_elements)?,
            };
            enhancer.enhance_into(
                frame,
                &mut scratch,
                width,
                height,
                self.params.enhancer_blend,
            )?;
            Some(scratch)
        } else {
            None
        };
        let output_chw = enhanced.as_ref().unwrap_or(frame);
        unsafe {
            self.gpu.chw_f32_to_pitched_nv12_scaled_color(
                output_chw,
                output_device_ptr,
                output_pitch,
                output_height,
                output_width,
                output_height,
                output_width,
                matrix,
                range,
                pixel_format,
            )?;
        }
        self.enhanced_scratch = enhanced.take();
        Ok(result)
    }

    pub fn process_rgb(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> anyhow::Result<ProcessedRgb> {
        validate_rgb_frame(data, width, height)?;
        let input_len = data.len();
        let mut input = match self.rgb_input_scratch.take() {
            Some(buffer) if buffer.len() == input_len => buffer,
            _ => self.gpu.alloc_zeros_u8(input_len)?,
        };
        self.gpu.upload_into_u8(data, &mut input)?;
        let frame_elements = 3 * width as usize * height as usize;
        let mut chw = match self.frame_scratch.take() {
            Some(buffer) if buffer.len() == frame_elements => buffer,
            _ => self.gpu.alloc_zeros(frame_elements)?,
        };
        self.gpu
            .hwc_u8_to_chw_f32(&input, &mut chw, height, width)?;
        let result = self.process_chw(&mut chw, height, width)?;
        let turns = if self.params.manual_rotation_enabled {
            u32::from(self.params.manual_rotation_angle / 90) & 3
        } else {
            0
        };
        let overlays = normalize_face_overlays(&result.faces, width, height, turns);
        let (output_width, output_height) = output_dimensions(&self.params, width, height)?;
        let mut enhanced = if let Some(enhancer) = &mut self.enhancer {
            let output_elements = 3usize
                .checked_mul(output_width as usize)
                .and_then(|elements| elements.checked_mul(output_height as usize))
                .ok_or_else(|| anyhow::anyhow!("enhanced output allocation overflows"))?;
            let mut output = match self.enhanced_scratch.take() {
                Some(buffer) if buffer.len() == output_elements => buffer,
                _ => self.gpu.alloc_zeros(output_elements)?,
            };
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
        let mut output_gpu = match self.rgb_output_scratch.take() {
            Some(buffer) if buffer.len() == output_len => buffer,
            _ => self.gpu.alloc_zeros_u8(output_len)?,
        };
        self.gpu
            .chw_f32_to_hwc_u8(output_chw, &mut output_gpu, output_height, output_width)?;
        let data = self.gpu.download_u8(&output_gpu)?;
        self.rgb_input_scratch = Some(input);
        self.frame_scratch = Some(chw);
        self.enhanced_scratch = enhanced.take();
        self.rgb_output_scratch = Some(output_gpu);
        Ok(ProcessedRgb {
            data,
            width: output_width,
            height: output_height,
            faces_detected: result.faces_detected,
            faces_swapped: result.faces_swapped,
            overlays,
        })
    }
}

fn normalize_face_overlays(
    faces: &[crate::pipeline::face_detector::DetectedFace],
    width: u32,
    height: u32,
    turns: u32,
) -> Vec<FaceOverlay> {
    let (detected_width, detected_height) = if turns.is_multiple_of(2) {
        (width, height)
    } else {
        (height, width)
    };
    let normalize = |point: [f32; 2]| {
        let u = point[0] / detected_width.max(1) as f32;
        let v = point[1] / detected_height.max(1) as f32;
        match turns & 3 {
            1 => [v, 1.0 - u],
            2 => [1.0 - u, 1.0 - v],
            3 => [1.0 - v, u],
            _ => [u, v],
        }
    };
    faces
        .iter()
        .map(|face| {
            let corners = [
                normalize([face.bbox[0], face.bbox[1]]),
                normalize([face.bbox[2], face.bbox[3]]),
            ];
            let mut kps_5 = [[0.0; 2]; 5];
            for (output, point) in kps_5.iter_mut().zip(face.kps_5) {
                *output = normalize(point);
            }
            FaceOverlay {
                bbox: [
                    corners[0][0].min(corners[1][0]),
                    corners[0][1].min(corners[1][1]),
                    corners[0][0].max(corners[1][0]),
                    corners[0][1].max(corners[1][1]),
                ],
                kps_5,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{apply_face_likeness, normalize_face_overlays, rotate_analysis_image};
    use crate::pipeline::face_detector::DetectedFace;

    #[test]
    fn face_likeness_matches_crossswap_latent_subtraction() {
        let mut source = [0.5, -0.25, 1.0];
        apply_face_likeness(&mut source, &[0.2, 0.4, -0.5], 0.75);
        assert_eq!(source, [0.35, -0.55, 1.375]);
    }

    #[test]
    fn manual_analysis_rotation_preserves_pixels_and_rotates_dimensions() {
        let image = image::RgbImage::from_raw(2, 1, vec![1, 2, 3, 4, 5, 6]).unwrap();
        let rotated = rotate_analysis_image(image, true, 90);
        assert_eq!(rotated.dimensions(), (1, 2));
        assert_eq!(rotated.into_raw(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn diagnostic_geometry_is_normalized_back_to_display_orientation() {
        let face = DetectedFace {
            bbox: [10.0, 20.0, 30.0, 60.0],
            kps_5: [[10.0, 20.0]; 5],
            score: 1.0,
        };
        let overlays = normalize_face_overlays(&[face], 100, 200, 0);
        assert_eq!(overlays[0].bbox, [0.1, 0.1, 0.3, 0.3]);
        assert_eq!(overlays[0].kps_5[0], [0.1, 0.1]);
    }
}
