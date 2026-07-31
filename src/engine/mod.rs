//! Immutable engine generations and atomic activation primitives.

mod builder;
mod spec;
mod supervisor;

pub use builder::{
    BuildCancellation, BuildPhase, BuildRequestOutcome, BuildSnapshot, ShadowBuild,
    ShadowBuildQueue,
};
pub use spec::{EngineSpec, EngineSpecError, FaceAssignmentSpec, ModelArtifact, ModelRole};
pub use supervisor::{
    ActivationError, ActivationOutcome, EngineFrame, EngineGeneration, EngineSupervisor,
    FrameOutcome, OwnedEngineSupervisor, ProbationUpdate, SupervisorPhase, SupervisorSnapshot,
};
