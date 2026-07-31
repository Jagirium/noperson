use std::collections::BTreeMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::Duration;

use noperson::config::parameters::FaceSwapParams;
use noperson::config::settings::{DetectorModel, ExecutionProvider};
use noperson::engine::{
    BuildCancellation, BuildPhase, BuildRequestOutcome, EngineGeneration, EngineSpec,
    EngineSupervisor, ModelArtifact, ModelRole, ShadowBuild, ShadowBuildQueue,
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
        models,
        params: FaceSwapParams::default(),
    }
}

struct ScriptedBuilder {
    started: Sender<char>,
    release_cancelled_a: Receiver<()>,
}

impl ShadowBuild<String> for ScriptedBuilder {
    type Error = String;

    fn build(
        &mut self,
        spec: &EngineSpec,
        cancellation: &BuildCancellation,
    ) -> Result<String, Self::Error> {
        let seed = spec.identity_sha256.chars().next().unwrap();
        self.started.send(seed).unwrap();
        match seed {
            'a' => {
                while !cancellation.is_cancelled() {
                    std::thread::yield_now();
                }
                self.release_cancelled_a.recv().unwrap();
                Err("cancelled stale build".to_owned())
            }
            'f' => Err("simulated allocation failure".to_owned()),
            _ => Ok(seed.to_string()),
        }
    }
}

#[test]
fn rapid_requests_cancel_the_build_and_coalesce_to_the_latest_spec() {
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let queue = ShadowBuildQueue::spawn(ScriptedBuilder {
        started: started_tx,
        release_cancelled_a: release_rx,
    })
    .unwrap();

    let a = spec('a');
    let c = spec('c');
    let c_id = c.generation_digest().unwrap();
    assert!(matches!(
        queue.request(a).unwrap(),
        BuildRequestOutcome::Queued { .. }
    ));
    assert_eq!(
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        'a'
    );

    queue.request(spec('b')).unwrap();
    queue.request(c).unwrap();
    release_tx.send(()).unwrap();

    let snapshot = queue.wait_until_settled(Duration::from_secs(1));
    assert_eq!(snapshot.phase, BuildPhase::Ready);
    assert_eq!(snapshot.ready_generation.as_deref(), Some(c_id.as_str()));
    assert_eq!(
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        'c'
    );
    assert!(
        started_rx.try_recv().is_err(),
        "intermediate build must not start"
    );

    let ready = queue.take_ready().expect("latest generation ready");
    assert_eq!(ready.id(), c_id);
    assert_eq!(ready.engine(), "c");
}

#[test]
fn duplicate_requests_are_coalesced_without_a_second_build() {
    let (started_tx, started_rx) = mpsc::channel();
    let (_release_tx, release_rx) = mpsc::channel();
    let queue = ShadowBuildQueue::spawn(ScriptedBuilder {
        started: started_tx,
        release_cancelled_a: release_rx,
    })
    .unwrap();
    let candidate = spec('d');

    queue.request(candidate.clone()).unwrap();
    assert!(matches!(
        queue.request(candidate).unwrap(),
        BuildRequestOutcome::Coalesced { .. }
    ));
    assert_eq!(
        queue.wait_until_settled(Duration::from_secs(1)).phase,
        BuildPhase::Ready
    );
    assert_eq!(
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        'd'
    );
    assert!(started_rx.try_recv().is_err());
}

#[test]
fn build_failure_never_changes_the_active_generation() {
    let active = EngineGeneration::new(spec('e'), "active").unwrap();
    let supervisor = EngineSupervisor::new(active, 2);
    let active_id = supervisor.begin_frame().id().to_owned();
    let (started_tx, _started_rx) = mpsc::channel();
    let (_release_tx, release_rx) = mpsc::channel();
    let queue = ShadowBuildQueue::spawn(ScriptedBuilder {
        started: started_tx,
        release_cancelled_a: release_rx,
    })
    .unwrap();

    queue.request(spec('f')).unwrap();
    let snapshot = queue.wait_until_settled(Duration::from_secs(1));

    assert_eq!(snapshot.phase, BuildPhase::Failed);
    assert_eq!(
        snapshot.last_failure.as_deref(),
        Some("simulated allocation failure")
    );
    assert!(queue.take_ready().is_none());
    assert_eq!(supervisor.begin_frame().id(), active_id);
}

#[test]
fn returning_to_the_active_spec_cancels_every_pending_candidate() {
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let queue = ShadowBuildQueue::spawn(ScriptedBuilder {
        started: started_tx,
        release_cancelled_a: release_rx,
    })
    .unwrap();

    queue.request(spec('a')).unwrap();
    assert_eq!(
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        'a'
    );
    queue.cancel_pending();
    release_tx.send(()).unwrap();

    let deadline = std::time::Instant::now() + Duration::from_secs(1);
    while queue.snapshot().phase != BuildPhase::Idle && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert_eq!(queue.snapshot().phase, BuildPhase::Idle);
    assert!(queue.take_ready().is_none());
}
