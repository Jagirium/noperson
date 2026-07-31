use std::fmt::Display;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::{EngineGeneration, EngineSpec, EngineSpecError};

/// Cooperative cancellation observed between expensive model-build stages.
#[derive(Clone, Default)]
pub struct BuildCancellation {
    cancelled: Arc<AtomicBool>,
}

impl BuildCancellation {
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

/// Builds one complete shadow backend without touching the active generation.
pub trait ShadowBuild<E>: Send + 'static {
    type Error: Display + Send + 'static;

    fn build(
        &mut self,
        spec: &EngineSpec,
        cancellation: &BuildCancellation,
    ) -> Result<E, Self::Error>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildRequestOutcome {
    Queued { generation: String },
    Coalesced { generation: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildPhase {
    Idle,
    Queued,
    Building,
    Ready,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildSnapshot {
    pub phase: BuildPhase,
    pub queued_generation: Option<String>,
    pub building_generation: Option<String>,
    pub ready_generation: Option<String>,
    pub last_failure: Option<String>,
}

struct PendingBuild {
    generation: String,
    spec: EngineSpec,
}

struct ActiveBuild {
    generation: String,
    cancellation: BuildCancellation,
}

struct BuildState<E> {
    desired: Option<PendingBuild>,
    building: Option<ActiveBuild>,
    ready: Option<EngineGeneration<E>>,
    last_failure: Option<String>,
    shutdown: bool,
}

struct Shared<E> {
    state: Mutex<BuildState<E>>,
    changed: Condvar,
}

/// Single background builder with latest-request-wins coalescing.
pub struct ShadowBuildQueue<E> {
    shared: Arc<Shared<E>>,
    worker: Option<JoinHandle<()>>,
}

impl<E: Send + 'static> ShadowBuildQueue<E> {
    pub fn spawn<B>(mut builder: B) -> std::io::Result<Self>
    where
        B: ShadowBuild<E>,
    {
        let shared = Arc::new(Shared {
            state: Mutex::new(BuildState {
                desired: None,
                building: None,
                ready: None,
                last_failure: None,
                shutdown: false,
            }),
            changed: Condvar::new(),
        });
        let worker_shared = Arc::clone(&shared);
        let worker = std::thread::Builder::new()
            .name("engine-shadow-builder".to_owned())
            .spawn(move || build_loop(&worker_shared, &mut builder))?;
        Ok(Self {
            shared,
            worker: Some(worker),
        })
    }

    pub fn request(&self, spec: EngineSpec) -> Result<BuildRequestOutcome, EngineSpecError> {
        let generation = spec.generation_digest()?;
        let mut state = self.lock_state();
        let duplicate = state
            .desired
            .as_ref()
            .is_some_and(|request| request.generation == generation)
            || state
                .building
                .as_ref()
                .is_some_and(|build| build.generation == generation)
            || state
                .ready
                .as_ref()
                .is_some_and(|ready| ready.id() == generation);
        if duplicate {
            return Ok(BuildRequestOutcome::Coalesced { generation });
        }

        if let Some(building) = &state.building {
            building.cancellation.cancel();
        }
        state.desired = Some(PendingBuild {
            generation: generation.clone(),
            spec,
        });
        state.ready = None;
        state.last_failure = None;
        self.shared.changed.notify_all();
        Ok(BuildRequestOutcome::Queued { generation })
    }

    pub fn take_ready(&self) -> Option<EngineGeneration<E>> {
        let mut state = self.lock_state();
        let ready = state.ready.take();
        if ready.is_some() {
            self.shared.changed.notify_all();
        }
        ready
    }

    /// Cancel any shadow work when the desired state returns to the active spec.
    pub fn cancel_pending(&self) {
        let mut state = self.lock_state();
        if let Some(building) = &state.building {
            building.cancellation.cancel();
        }
        state.desired = None;
        state.ready = None;
        state.last_failure = None;
        self.shared.changed.notify_all();
    }

    pub fn snapshot(&self) -> BuildSnapshot {
        snapshot_from(&self.lock_state())
    }

    /// Wait for Ready/Failed, bounded so callers never depend on an untracked worker.
    pub fn wait_until_settled(&self, timeout: Duration) -> BuildSnapshot {
        let deadline = Instant::now() + timeout;
        let mut state = self.lock_state();
        loop {
            let snapshot = snapshot_from(&state);
            if matches!(snapshot.phase, BuildPhase::Ready | BuildPhase::Failed) {
                return snapshot;
            }

            let now = Instant::now();
            if now >= deadline {
                return snapshot;
            }
            let remaining = deadline.saturating_duration_since(now);
            let (next_state, wait) = self
                .shared
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next_state;
            if wait.timed_out() {
                return snapshot_from(&state);
            }
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, BuildState<E>> {
        self.shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl<E> Drop for ShadowBuildQueue<E> {
    fn drop(&mut self) {
        {
            let mut state = self
                .shared
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.shutdown = true;
            state.desired = None;
            if let Some(building) = &state.building {
                building.cancellation.cancel();
            }
            self.shared.changed.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn build_loop<B, E>(shared: &Shared<E>, builder: &mut B)
where
    B: ShadowBuild<E>,
{
    loop {
        let (request, cancellation) = {
            let mut state = shared
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            while state.desired.is_none() && !state.shutdown {
                state = shared
                    .changed
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            if state.shutdown {
                return;
            }

            let request = state.desired.take().expect("desired build checked above");
            let cancellation = BuildCancellation::default();
            state.building = Some(ActiveBuild {
                generation: request.generation.clone(),
                cancellation: cancellation.clone(),
            });
            shared.changed.notify_all();
            (request, cancellation)
        };

        let result = builder
            .build(&request.spec, &cancellation)
            .map_err(|error| error.to_string())
            .and_then(|engine| {
                EngineGeneration::new(request.spec, engine).map_err(|error| error.to_string())
            });

        let mut state = shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.shutdown {
            return;
        }
        state.building = None;

        if cancellation.is_cancelled() || state.desired.is_some() {
            shared.changed.notify_all();
            continue;
        }

        match result {
            Ok(generation) => {
                state.ready = Some(generation);
                state.last_failure = None;
            }
            Err(error) => {
                state.ready = None;
                state.last_failure = Some(error);
            }
        }
        shared.changed.notify_all();
    }
}

fn snapshot_from<E>(state: &BuildState<E>) -> BuildSnapshot {
    let phase = if state.ready.is_some() {
        BuildPhase::Ready
    } else if state.building.is_some() {
        BuildPhase::Building
    } else if state.desired.is_some() {
        BuildPhase::Queued
    } else if state.last_failure.is_some() {
        BuildPhase::Failed
    } else {
        BuildPhase::Idle
    };
    BuildSnapshot {
        phase,
        queued_generation: state
            .desired
            .as_ref()
            .map(|request| request.generation.clone()),
        building_generation: state
            .building
            .as_ref()
            .map(|build| build.generation.clone()),
        ready_generation: state.ready.as_ref().map(|ready| ready.id().to_owned()),
        last_failure: state.last_failure.clone(),
    }
}
