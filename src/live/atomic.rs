use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use cudarc::driver::{CudaSlice, CudaStream};

use super::{LiveEngine, ProcessedRgb, ResolvedFaceAssignment, ResolvedIdentity, blake3_file};
use crate::config::parameters::FaceSwapParams;
use crate::engine::{
    ActivationError, ActivationOutcome, BuildCancellation, BuildRequestOutcome, BuildSnapshot,
    EngineGeneration, EngineSpec, FaceAssignmentSpec, FrameOutcome, OwnedEngineSupervisor,
    ShadowBuild, ShadowBuildQueue, SupervisorPhase, SupervisorSnapshot,
};
use crate::gpu::ops::GpuOps;
use crate::pipeline::frame_processor::FrameResult;

#[derive(Debug, Clone)]
pub struct FaceAssignmentPaths {
    pub source_path: PathBuf,
    pub target_path: Option<PathBuf>,
    pub similarity_threshold: f32,
    /// Target-local controls; `None` inherits the generation defaults.
    pub params: Option<FaceSwapParams>,
    /// Content-addressed artifacts selected by this assignment.
    pub models: std::collections::BTreeMap<crate::engine::ModelRole, crate::engine::ModelArtifact>,
}

#[derive(Debug, Clone)]
pub enum FaceIdentityInput {
    Image(PathBuf),
    Embedding(Vec<f32>),
}

#[derive(Debug, Clone)]
pub struct FaceAssignmentInputs {
    pub source: FaceIdentityInput,
    pub target: Option<FaceIdentityInput>,
    pub similarity_threshold: f32,
    pub params: Option<FaceSwapParams>,
    pub models: std::collections::BTreeMap<crate::engine::ModelRole, crate::engine::ModelArtifact>,
}

impl From<&FaceAssignmentPaths> for FaceAssignmentInputs {
    fn from(paths: &FaceAssignmentPaths) -> Self {
        Self {
            source: FaceIdentityInput::Image(paths.source_path.clone()),
            target: paths.target_path.clone().map(FaceIdentityInput::Image),
            similarity_threshold: paths.similarity_threshold,
            params: paths.params.clone(),
            models: paths.models.clone(),
        }
    }
}

pub fn embedding_blake3(embedding: &[f32]) -> anyhow::Result<String> {
    anyhow::ensure!(
        embedding.len() == crate::math::constants::EMBEDDING_SIZE,
        "ArcFace embedding must contain {} values, got {}",
        crate::math::constants::EMBEDDING_SIZE,
        embedding.len()
    );
    anyhow::ensure!(
        embedding.iter().all(|value| value.is_finite()),
        "ArcFace embedding contains a non-finite value"
    );
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"noperson-arcface-embedding-v1\0");
    for value in embedding {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    Ok(hasher.finalize().to_hex().to_string())
}

#[derive(Default)]
struct IdentityCatalog {
    identities: RwLock<HashMap<String, ResolvedIdentity>>,
}

impl IdentityCatalog {
    fn register_path(&self, path: &Path) -> anyhow::Result<String> {
        let digest = blake3_file(path)?;
        self.identities
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(digest.clone(), ResolvedIdentity::Image(path.to_path_buf()));
        Ok(digest)
    }

    fn register(&self, identity: &FaceIdentityInput) -> anyhow::Result<String> {
        match identity {
            FaceIdentityInput::Image(path) => self.register_path(path),
            FaceIdentityInput::Embedding(embedding) => {
                let digest = embedding_blake3(embedding)?;
                self.identities
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(
                        digest.clone(),
                        ResolvedIdentity::Embedding(embedding.clone()),
                    );
                Ok(digest)
            }
        }
    }

    fn resolve(&self, digest: &str) -> Option<ResolvedIdentity> {
        self.identities
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(digest)
            .cloned()
    }
}

/// Concrete whole-engine builder used by the latest-request-wins queue.
pub struct LiveShadowBuilder {
    gpu: Arc<GpuOps>,
    models_dir: PathBuf,
    stream: Arc<CudaStream>,
    identities: Arc<IdentityCatalog>,
}

impl LiveShadowBuilder {
    pub fn new(gpu: Arc<GpuOps>, models_dir: PathBuf, stream: Arc<CudaStream>) -> Self {
        Self {
            gpu,
            models_dir,
            stream,
            identities: Arc::new(IdentityCatalog::default()),
        }
    }

    pub fn register_identity(&self, path: &Path) -> anyhow::Result<String> {
        self.identities.register_path(path)
    }
}

impl ShadowBuild<LiveEngine> for LiveShadowBuilder {
    type Error = anyhow::Error;

    fn build(
        &mut self,
        spec: &EngineSpec,
        cancellation: &BuildCancellation,
    ) -> Result<LiveEngine, Self::Error> {
        anyhow::ensure!(!cancellation.is_cancelled(), "shadow build cancelled");
        let identity = self
            .identities
            .resolve(&spec.identity_blake3)
            .ok_or_else(|| anyhow::anyhow!("identity content is not registered"))?;
        let assignments = spec
            .assignments
            .iter()
            .map(|assignment| {
                Ok(ResolvedFaceAssignment {
                    source: self
                        .identities
                        .resolve(&assignment.source_identity_blake3)
                        .ok_or_else(|| {
                            anyhow::anyhow!("source identity content is not registered")
                        })?,
                    target: assignment
                        .target_identity_blake3
                        .as_deref()
                        .map(|digest| {
                            self.identities.resolve(digest).ok_or_else(|| {
                                anyhow::anyhow!("target identity content is not registered")
                            })
                        })
                        .transpose()?,
                    similarity_threshold: assignment.similarity_threshold,
                    params: assignment.params.clone(),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let engine = LiveEngine::new_from_spec_assignments_cancellable(
            Arc::clone(&self.gpu),
            &self.models_dir,
            &identity,
            &assignments,
            spec,
            &self.stream,
            cancellation,
        )?;
        anyhow::ensure!(!cancellation.is_cancelled(), "shadow build cancelled");
        Ok(engine)
    }
}

/// Real live backend with frame-boundary generation activation.
pub struct AtomicLiveEngine {
    supervisor: OwnedEngineSupervisor<LiveEngine>,
    builds: ShadowBuildQueue<LiveEngine>,
    identities: Arc<IdentityCatalog>,
}

impl AtomicLiveEngine {
    fn register_assignments(
        identities: &IdentityCatalog,
        spec: &mut EngineSpec,
        assignments: &[FaceAssignmentInputs],
    ) -> anyhow::Result<()> {
        spec.assignments = assignments
            .iter()
            .map(|assignment| {
                Ok(FaceAssignmentSpec {
                    source_identity_blake3: identities.register(&assignment.source)?,
                    target_identity_blake3: assignment
                        .target
                        .as_ref()
                        .map(|identity| identities.register(identity))
                        .transpose()?,
                    similarity_threshold: assignment.similarity_threshold,
                    params: assignment.params.clone(),
                    models: assignment.models.clone(),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        if let Some(first) = spec.assignments.first() {
            spec.identity_blake3 = first.source_identity_blake3.clone();
        }
        Ok(())
    }

    pub fn bootstrap(
        mut builder: LiveShadowBuilder,
        identity_path: &Path,
        mut initial_spec: EngineSpec,
        probation_frames: u32,
    ) -> anyhow::Result<Self> {
        initial_spec.identity_blake3 = builder.register_identity(identity_path)?;
        let initial_engine = builder.build(&initial_spec, &BuildCancellation::default())?;
        let initial = EngineGeneration::new(initial_spec, initial_engine)?;
        let identities = Arc::clone(&builder.identities);
        let builds = ShadowBuildQueue::spawn(builder)?;
        Ok(Self {
            supervisor: OwnedEngineSupervisor::new(initial, probation_frames),
            builds,
            identities,
        })
    }

    pub fn bootstrap_assignments(
        builder: LiveShadowBuilder,
        assignments: &[FaceAssignmentPaths],
        initial_spec: EngineSpec,
        probation_frames: u32,
    ) -> anyhow::Result<Self> {
        let assignments = assignments
            .iter()
            .map(FaceAssignmentInputs::from)
            .collect::<Vec<_>>();
        Self::bootstrap_inputs(builder, &assignments, initial_spec, probation_frames)
    }

    pub fn bootstrap_inputs(
        mut builder: LiveShadowBuilder,
        assignments: &[FaceAssignmentInputs],
        mut initial_spec: EngineSpec,
        probation_frames: u32,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !assignments.is_empty(),
            "at least one face assignment is required"
        );
        Self::register_assignments(&builder.identities, &mut initial_spec, assignments)?;
        let initial_engine = builder.build(&initial_spec, &BuildCancellation::default())?;
        let initial = EngineGeneration::new(initial_spec, initial_engine)?;
        let identities = Arc::clone(&builder.identities);
        let builds = ShadowBuildQueue::spawn(builder)?;
        Ok(Self {
            supervisor: OwnedEngineSupervisor::new(initial, probation_frames),
            builds,
            identities,
        })
    }

    pub fn request(
        &self,
        mut spec: EngineSpec,
        identity_path: &Path,
    ) -> anyhow::Result<BuildRequestOutcome> {
        spec.identity_blake3 = self.identities.register_path(identity_path)?;
        let generation = spec.generation_digest()?;
        if self.supervisor.snapshot().active_generation == generation {
            self.builds.cancel_pending();
            return Ok(BuildRequestOutcome::Coalesced { generation });
        }
        Ok(self.builds.request(spec)?)
    }

    pub fn request_assignments(
        &self,
        spec: EngineSpec,
        assignments: &[FaceAssignmentPaths],
    ) -> anyhow::Result<BuildRequestOutcome> {
        let assignments = assignments
            .iter()
            .map(FaceAssignmentInputs::from)
            .collect::<Vec<_>>();
        self.request_inputs(spec, &assignments)
    }

    pub fn request_inputs(
        &self,
        mut spec: EngineSpec,
        assignments: &[FaceAssignmentInputs],
    ) -> anyhow::Result<BuildRequestOutcome> {
        anyhow::ensure!(
            !assignments.is_empty(),
            "at least one face assignment is required"
        );
        Self::register_assignments(&self.identities, &mut spec, assignments)?;
        let generation = spec.generation_digest()?;
        if self.supervisor.snapshot().active_generation == generation {
            self.builds.cancel_pending();
            return Ok(BuildRequestOutcome::Coalesced { generation });
        }
        Ok(self.builds.request(spec)?)
    }

    /// Activate a ready shadow only at the caller's next frame boundary.
    pub fn poll_activation(&mut self) -> Result<Option<ActivationOutcome>, ActivationError> {
        if self.supervisor.snapshot().phase != SupervisorPhase::Stable {
            return Ok(None);
        }
        let Some(candidate) = self.builds.take_ready() else {
            return Ok(None);
        };
        self.supervisor.activate(candidate).map(Some)
    }

    pub fn process_chw(
        &mut self,
        frame: &mut CudaSlice<f32>,
        height: u32,
        width: u32,
    ) -> anyhow::Result<FrameResult> {
        self.poll_activation()?;
        let result = {
            let (_, engine) = self.supervisor.active_mut();
            engine.process_chw(frame, height, width)
        };
        self.supervisor.record_frame(if result.is_ok() {
            FrameOutcome::Success
        } else {
            FrameOutcome::Failure
        });
        result
    }

    pub fn process_chw_to_pitched_nv12(
        &mut self,
        frame: &mut CudaSlice<f32>,
        height: u32,
        width: u32,
        output_device_ptr: u64,
        output_pitch: u32,
        matrix: crate::io::native_video::ColorMatrix,
        range: crate::io::native_video::ColorRange,
        pixel_format: crate::io::native_video::PixelFormat,
    ) -> anyhow::Result<FrameResult> {
        self.poll_activation()?;
        let result = {
            let (_, engine) = self.supervisor.active_mut();
            engine.process_chw_to_pitched_nv12(
                frame,
                height,
                width,
                output_device_ptr,
                output_pitch,
                matrix,
                range,
                pixel_format,
            )
        };
        self.supervisor.record_frame(if result.is_ok() {
            FrameOutcome::Success
        } else {
            FrameOutcome::Failure
        });
        result
    }

    pub fn process_rgb(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> anyhow::Result<ProcessedRgb> {
        self.poll_activation()?;
        let result = {
            let (_, engine) = self.supervisor.active_mut();
            engine.process_rgb(data, width, height)
        };
        self.supervisor.record_frame(if result.is_ok() {
            FrameOutcome::Success
        } else {
            FrameOutcome::Failure
        });
        result
    }

    pub fn supervisor_snapshot(&self) -> SupervisorSnapshot {
        self.supervisor.snapshot()
    }

    pub fn build_snapshot(&self) -> BuildSnapshot {
        self.builds.snapshot()
    }

    pub fn wait_for_build(&self, timeout: Duration) -> BuildSnapshot {
        self.builds.wait_until_settled(timeout)
    }

    pub fn cancel_pending_build(&self) {
        self.builds.cancel_pending();
    }
}
