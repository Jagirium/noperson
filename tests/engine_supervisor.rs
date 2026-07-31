use std::collections::BTreeMap;

use noperson::config::parameters::FaceSwapParams;
use noperson::config::settings::{DetectorModel, ExecutionProvider};
use noperson::engine::{
    ActivationError, ActivationOutcome, EngineGeneration, EngineSpec, EngineSupervisor,
    FrameOutcome, ModelArtifact, ModelRole, OwnedEngineSupervisor, ProbationUpdate,
    SupervisorPhase,
};

fn spec(seed: char) -> EngineSpec {
    let digest: String = std::iter::repeat_n(seed, 64).collect();
    let models = [
        ModelRole::Detector,
        ModelRole::Recognizer,
        ModelRole::Swapper,
        ModelRole::Emap,
    ]
    .into_iter()
    .map(|role| {
        (
            role,
            ModelArtifact {
                logical_name: role.as_str().to_owned(),
                filename: format!("{}.bin", role.as_str()),
                sha256: digest.clone(),
            },
        )
    })
    .collect::<BTreeMap<_, _>>();
    EngineSpec {
        provider: ExecutionProvider::Cuda,
        device_id: 0,
        detector: DetectorModel::YoloFace8n,
        identity_sha256: digest,
        assignments: Vec::new(),
        models,
        params: FaceSwapParams::default(),
    }
}

fn generation(seed: char, payload: &'static str) -> EngineGeneration<&'static str> {
    EngineGeneration::new(spec(seed), payload).expect("valid generation")
}

#[test]
fn activation_is_atomic_at_the_frame_generation_boundary() {
    let supervisor = EngineSupervisor::new(generation('a', "old"), 2);
    let old_frame = supervisor.begin_frame();
    let old_id = old_frame.id().to_owned();

    assert_eq!(
        supervisor.activate(generation('b', "new")).unwrap(),
        ActivationOutcome::Activated
    );
    let new_frame = supervisor.begin_frame();

    assert_eq!(*old_frame.engine(), "old");
    assert_eq!(*new_frame.engine(), "new");
    assert_ne!(old_frame.id(), new_frame.id());
    assert_eq!(old_frame.id(), old_id);
}

#[test]
fn successful_probation_promotes_candidate_and_releases_rollback_slot() {
    let supervisor = EngineSupervisor::new(generation('a', "old"), 2);
    supervisor.activate(generation('b', "new")).unwrap();
    let candidate_id = supervisor.begin_frame().id().to_owned();

    assert_eq!(
        supervisor.record_frame(&candidate_id, FrameOutcome::Success),
        ProbationUpdate::Pending {
            remaining_frames: 1
        }
    );
    assert_eq!(
        supervisor.record_frame(&candidate_id, FrameOutcome::Success),
        ProbationUpdate::Promoted {
            generation: candidate_id.clone()
        }
    );

    let snapshot = supervisor.snapshot();
    assert_eq!(snapshot.phase, SupervisorPhase::Stable);
    assert_eq!(snapshot.active_generation, candidate_id);
    assert_eq!(snapshot.rollback_generation, None);
}

#[test]
fn failed_probation_rolls_back_without_mutating_in_flight_frames() {
    let supervisor = EngineSupervisor::new(generation('a', "old"), 3);
    let stable_id = supervisor.begin_frame().id().to_owned();
    supervisor.activate(generation('b', "new")).unwrap();
    let failed_frame = supervisor.begin_frame();
    let candidate_id = failed_frame.id().to_owned();

    assert_eq!(
        supervisor.record_frame(&candidate_id, FrameOutcome::Failure),
        ProbationUpdate::RolledBack {
            rejected_generation: candidate_id,
            restored_generation: stable_id.clone(),
        }
    );

    assert_eq!(*failed_frame.engine(), "new");
    let next_frame = supervisor.begin_frame();
    assert_eq!(*next_frame.engine(), "old");
    assert_eq!(next_frame.id(), stable_id);
}

#[test]
fn second_activation_waits_until_probation_finishes() {
    let supervisor = EngineSupervisor::new(generation('a', "old"), 2);
    supervisor.activate(generation('b', "candidate")).unwrap();

    assert!(matches!(
        supervisor.activate(generation('c', "too-soon")),
        Err(ActivationError::ProbationInProgress { .. })
    ));
}

#[test]
fn activating_current_generation_is_idempotent() {
    let supervisor = EngineSupervisor::new(generation('a', "old"), 2);
    let same = generation('a', "replacement payload");

    assert_eq!(
        supervisor.activate(same).unwrap(),
        ActivationOutcome::AlreadyActive
    );
    assert_eq!(*supervisor.begin_frame().engine(), "old");
}

#[test]
fn single_owner_supervisor_exposes_the_active_engine_without_a_lock() {
    let mut supervisor = OwnedEngineSupervisor::new(generation('a', "old"), 2);
    let old_id = supervisor.active_id().to_owned();
    assert_eq!(*supervisor.active_mut().1, "old");

    supervisor.activate(generation('b', "new")).unwrap();
    let candidate_id = supervisor.active_id().to_owned();
    assert_ne!(candidate_id, old_id);
    assert_eq!(*supervisor.active_mut().1, "new");

    assert!(matches!(
        supervisor.record_frame(FrameOutcome::Failure),
        ProbationUpdate::RolledBack { .. }
    ));
    assert_eq!(supervisor.active_id(), old_id);
    assert_eq!(*supervisor.active_mut().1, "old");
}
