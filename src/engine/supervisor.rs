use std::sync::{Arc, Mutex, MutexGuard};

use arc_swap::ArcSwap;
use thiserror::Error;

use super::{EngineSpec, EngineSpecError};

/// One fully built immutable backend and its content-addressed specification.
pub struct EngineGeneration<E> {
    id: String,
    spec: EngineSpec,
    engine: E,
}

impl<E> EngineGeneration<E> {
    pub fn new(spec: EngineSpec, engine: E) -> Result<Self, EngineSpecError> {
        let id = spec.generation_digest()?;
        Ok(Self { id, spec, engine })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn spec(&self) -> &EngineSpec {
        &self.spec
    }

    pub fn engine(&self) -> &E {
        &self.engine
    }
}

/// A single frame's immutable view of an engine generation.
///
/// Keeping this lease alive guarantees that retirement cannot destroy the
/// generation while the frame is still executing.
#[derive(Clone)]
pub struct EngineFrame<E> {
    generation: Arc<EngineGeneration<E>>,
}

impl<E> EngineFrame<E> {
    pub fn id(&self) -> &str {
        self.generation.id()
    }

    pub fn spec(&self) -> &EngineSpec {
        self.generation.spec()
    }

    pub fn engine(&self) -> &E {
        self.generation.engine()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameOutcome {
    Success,
    Failure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationOutcome {
    Activated,
    AlreadyActive,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ActivationError {
    #[error("generation {candidate_generation} is still in probation")]
    ProbationInProgress { candidate_generation: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbationUpdate {
    Ignored,
    Pending {
        remaining_frames: u32,
    },
    Promoted {
        generation: String,
    },
    RolledBack {
        rejected_generation: String,
        restored_generation: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupervisorPhase {
    Stable,
    Probation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupervisorSnapshot {
    pub phase: SupervisorPhase,
    pub active_generation: String,
    pub rollback_generation: Option<String>,
    pub probation_remaining_frames: Option<u32>,
}

enum ControlPhase<E> {
    Stable,
    Probation {
        candidate_generation: String,
        rollback: Arc<EngineGeneration<E>>,
        remaining_frames: u32,
    },
}

struct Control<E> {
    phase: ControlPhase<E>,
}

/// Owns the one lock-free active-generation pointer and its small control plane.
///
/// The frame hot path only clones an `Arc` from `ArcSwap`; build, activation,
/// probation, and rollback bookkeeping stay off that path.
pub struct EngineSupervisor<E> {
    active: ArcSwap<EngineGeneration<E>>,
    control: Mutex<Control<E>>,
    probation_frames: u32,
}

impl<E> EngineSupervisor<E> {
    pub fn new(initial: EngineGeneration<E>, probation_frames: u32) -> Self {
        Self {
            active: ArcSwap::from_pointee(initial),
            control: Mutex::new(Control {
                phase: ControlPhase::Stable,
            }),
            probation_frames,
        }
    }

    /// Capture exactly one generation for an entire frame.
    pub fn begin_frame(&self) -> EngineFrame<E> {
        EngineFrame {
            generation: self.active.load_full(),
        }
    }

    /// Atomically make a fully built candidate visible to future frames.
    pub fn activate(
        &self,
        candidate: EngineGeneration<E>,
    ) -> Result<ActivationOutcome, ActivationError> {
        let mut control = self.lock_control();
        let active = self.active.load_full();
        if active.id() == candidate.id() {
            return Ok(ActivationOutcome::AlreadyActive);
        }

        if let ControlPhase::Probation {
            candidate_generation,
            ..
        } = &control.phase
        {
            return Err(ActivationError::ProbationInProgress {
                candidate_generation: candidate_generation.clone(),
            });
        }

        let candidate_generation = candidate.id().to_owned();
        if self.probation_frames == 0 {
            self.active.store(Arc::new(candidate));
        } else {
            let rollback = self.active.swap(Arc::new(candidate));
            control.phase = ControlPhase::Probation {
                candidate_generation,
                rollback,
                remaining_frames: self.probation_frames,
            };
        }
        Ok(ActivationOutcome::Activated)
    }

    /// Advance probation using the generation that actually rendered a frame.
    pub fn record_frame(&self, generation: &str, outcome: FrameOutcome) -> ProbationUpdate {
        let mut control = self.lock_control();
        let phase = std::mem::replace(&mut control.phase, ControlPhase::Stable);
        let ControlPhase::Probation {
            candidate_generation,
            rollback,
            remaining_frames,
        } = phase
        else {
            return ProbationUpdate::Ignored;
        };

        if candidate_generation != generation {
            control.phase = ControlPhase::Probation {
                candidate_generation,
                rollback,
                remaining_frames,
            };
            return ProbationUpdate::Ignored;
        }

        match outcome {
            FrameOutcome::Failure => {
                let rejected_generation = candidate_generation;
                let restored_generation = rollback.id().to_owned();
                self.active.store(rollback);
                ProbationUpdate::RolledBack {
                    rejected_generation,
                    restored_generation,
                }
            }
            FrameOutcome::Success if remaining_frames <= 1 => ProbationUpdate::Promoted {
                generation: candidate_generation,
            },
            FrameOutcome::Success => {
                let remaining_frames = remaining_frames - 1;
                control.phase = ControlPhase::Probation {
                    candidate_generation,
                    rollback,
                    remaining_frames,
                };
                ProbationUpdate::Pending { remaining_frames }
            }
        }
    }

    pub fn snapshot(&self) -> SupervisorSnapshot {
        let control = self.lock_control();
        let active_generation = self.active.load_full().id().to_owned();
        match &control.phase {
            ControlPhase::Stable => SupervisorSnapshot {
                phase: SupervisorPhase::Stable,
                active_generation,
                rollback_generation: None,
                probation_remaining_frames: None,
            },
            ControlPhase::Probation {
                rollback,
                remaining_frames,
                ..
            } => SupervisorSnapshot {
                phase: SupervisorPhase::Probation,
                active_generation,
                rollback_generation: Some(rollback.id().to_owned()),
                probation_remaining_frames: Some(*remaining_frames),
            },
        }
    }

    fn lock_control(&self) -> MutexGuard<'_, Control<E>> {
        self.control
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}
