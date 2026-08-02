use noperson::extra_gui::{EditorJobPhase, EditorJobState};

#[test]
fn editor_jobs_follow_one_explicit_state_machine() {
    let mut state = EditorJobState::default();
    assert_eq!(state.phase(), EditorJobPhase::Idle);
    state.begin(EditorJobPhase::Analyzing).unwrap();
    assert!(state.begin(EditorJobPhase::Recording).is_err());
    state.cancel().unwrap();
    assert_eq!(state.phase(), EditorJobPhase::Cancelling);
    state.settle();
    state.begin(EditorJobPhase::Building).unwrap();
    state.fail();
    assert_eq!(state.phase(), EditorJobPhase::Failed);
    state.begin(EditorJobPhase::Previewing).unwrap();
}

#[test]
fn idle_and_failed_jobs_are_not_reported_as_cancellable() {
    let mut state = EditorJobState::default();
    assert!(state.cancel().is_err());
    state.fail();
    assert!(state.cancel().is_err());
}
