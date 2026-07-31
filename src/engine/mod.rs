//! Immutable engine generations and atomic activation primitives.

mod spec;
mod supervisor;

pub use spec::{EngineSpec, EngineSpecError, ModelArtifact, ModelRole};
pub use supervisor::{
    ActivationError, ActivationOutcome, EngineFrame, EngineGeneration, EngineSupervisor,
    FrameOutcome, ProbationUpdate, SupervisorPhase, SupervisorSnapshot,
};
