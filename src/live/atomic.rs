use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use cudarc::driver::{CudaSlice, CudaStream};

use super::{LiveEngine, ProcessedRgb, ResolvedFaceAssignment, sha256_file};
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
}

#[derive(Default)]
struct IdentityCatalog {
    paths: RwLock<HashMap<String, PathBuf>>,
}

impl IdentityCatalog {
    fn register(&self, path: &Path) -> anyhow::Result<String> {
        let digest = sha256_file(path)?;
        self.paths
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(digest.clone(), path.to_path_buf());
        Ok(digest)
    }

    fn resolve(&self, digest: &str) -> Option<PathBuf> {
        self.paths
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
        self.identities.register(path)
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
        let identity_path = self
            .identities
            .resolve(&spec.identity_sha256)
            .ok_or_else(|| anyhow::anyhow!("identity content is not registered"))?;
        let assignments = spec
            .assignments
            .iter()
            .map(|assignment| {
                Ok(ResolvedFaceAssignment {
                    source_path: self
                        .identities
                        .resolve(&assignment.source_identity_sha256)
                        .ok_or_else(|| {
                            anyhow::anyhow!("source identity content is not registered")
                        })?,
                    target_path: assignment
                        .target_identity_sha256
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
            &identity_path,
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
        assignments: &[FaceAssignmentPaths],
    ) -> anyhow::Result<()> {
        spec.assignments = assignments
            .iter()
            .map(|assignment| {
                Ok(FaceAssignmentSpec {
                    source_identity_sha256: identities.register(&assignment.source_path)?,
                    target_identity_sha256: assignment
                        .target_path
                        .as_deref()
                        .map(|path| identities.register(path))
                        .transpose()?,
                    similarity_threshold: assignment.similarity_threshold,
                    params: assignment.params.clone(),
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        if let Some(first) = spec.assignments.first() {
            spec.identity_sha256 = first.source_identity_sha256.clone();
        }
        Ok(())
    }

    pub fn bootstrap(
        mut builder: LiveShadowBuilder,
        identity_path: &Path,
        mut initial_spec: EngineSpec,
        probation_frames: u32,
    ) -> anyhow::Result<Self> {
        initial_spec.identity_sha256 = builder.register_identity(identity_path)?;
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
        mut builder: LiveShadowBuilder,
        assignments: &[FaceAssignmentPaths],
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
        spec.identity_sha256 = self.identities.register(identity_path)?;
        let generation = spec.generation_digest()?;
        if self.supervisor.snapshot().active_generation == generation {
            self.builds.cancel_pending();
            return Ok(BuildRequestOutcome::Coalesced { generation });
        }
        Ok(self.builds.request(spec)?)
    }

    pub fn request_assignments(
        &self,
        mut spec: EngineSpec,
        assignments: &[FaceAssignmentPaths],
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
}
