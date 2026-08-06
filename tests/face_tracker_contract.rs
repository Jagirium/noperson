use noperson::pipeline::face_detector::DetectedFace;
use noperson::pipeline::face_tracker::{
    TemporalFaceTracker, TrackerFrame, TrackerPolicy, detect_or_track,
};

fn face(x: f32, y: f32, size: f32) -> DetectedFace {
    DetectedFace {
        bbox: [x, y, x + size, y + size],
        kps_5: [
            [x + 0.30 * size, y + 0.35 * size],
            [x + 0.70 * size, y + 0.35 * size],
            [x + 0.50 * size, y + 0.55 * size],
            [x + 0.35 * size, y + 0.75 * size],
            [x + 0.65 * size, y + 0.75 * size],
        ],
        score: 0.95,
    }
}

fn expect_detect(frame: TrackerFrame) {
    assert!(matches!(frame, TrackerFrame::Detect));
}

#[test]
fn stable_motion_grows_the_interval_and_predicts_the_skipped_frame() {
    let mut tracker = TemporalFaceTracker::new(TrackerPolicy::realtime_adaptive());

    expect_detect(tracker.begin_frame(1920, 1080));
    tracker.update(vec![face(100.0, 100.0, 100.0)]);
    expect_detect(tracker.begin_frame(1920, 1080));
    tracker.update(vec![face(102.0, 100.0, 100.0)]);
    assert_eq!(tracker.detection_interval(), 2);

    let TrackerFrame::Tracked(predicted) = tracker.begin_frame(1920, 1080) else {
        panic!("stable tracks should skip one detector frame");
    };
    assert_eq!(predicted.len(), 1);
    assert!((predicted[0].bbox[0] - 104.0).abs() < 0.01);
    expect_detect(tracker.begin_frame(1920, 1080));
}

#[test]
fn unstable_association_immediately_returns_to_every_frame_detection() {
    let mut tracker = TemporalFaceTracker::new(TrackerPolicy::realtime_adaptive());
    expect_detect(tracker.begin_frame(640, 480));
    tracker.update(vec![face(10.0, 10.0, 80.0)]);
    expect_detect(tracker.begin_frame(640, 480));
    tracker.update(vec![face(11.0, 10.0, 80.0)]);
    assert_eq!(tracker.detection_interval(), 2);

    let TrackerFrame::Tracked(_) = tracker.begin_frame(640, 480) else {
        panic!("the stable track should get its scheduled prediction");
    };
    expect_detect(tracker.begin_frame(640, 480));
    tracker.update(vec![face(200.0, 150.0, 80.0)]);
    assert_eq!(tracker.detection_interval(), 1);
    expect_detect(tracker.begin_frame(640, 480));
}

#[test]
fn missing_detections_are_recovered_for_two_detector_observations() {
    let mut tracker = TemporalFaceTracker::new(TrackerPolicy::offline_recovery());
    expect_detect(tracker.begin_frame(640, 480));
    tracker.update(vec![face(30.0, 40.0, 60.0)]);

    expect_detect(tracker.begin_frame(640, 480));
    assert_eq!(tracker.update(Vec::new()).len(), 1);
    expect_detect(tracker.begin_frame(640, 480));
    assert_eq!(tracker.update(Vec::new()).len(), 1);
    expect_detect(tracker.begin_frame(640, 480));
    assert!(tracker.update(Vec::new()).is_empty());
}

#[test]
fn track_order_survives_reversed_detector_order() {
    let mut tracker = TemporalFaceTracker::new(TrackerPolicy::offline_recovery());
    expect_detect(tracker.begin_frame(640, 480));
    tracker.update(vec![face(20.0, 20.0, 50.0), face(300.0, 20.0, 50.0)]);

    expect_detect(tracker.begin_frame(640, 480));
    let tracked = tracker.update(vec![face(298.0, 20.0, 50.0), face(22.0, 20.0, 50.0)]);
    assert_eq!(tracked.len(), 2);
    assert!((tracked[0].bbox[0] - 22.0).abs() < 0.01);
    assert!((tracked[1].bbox[0] - 298.0).abs() < 0.01);
}

#[test]
fn geometry_change_discards_stale_tracks_and_forces_detection() {
    let mut tracker = TemporalFaceTracker::new(TrackerPolicy::realtime_adaptive());
    expect_detect(tracker.begin_frame(640, 480));
    tracker.update(vec![face(20.0, 20.0, 50.0)]);

    expect_detect(tracker.begin_frame(1280, 720));
    assert_eq!(tracker.track_count(), 0);
    assert_eq!(tracker.detection_interval(), 1);
}

#[test]
fn adaptive_route_does_not_invoke_the_detector_on_predicted_frames() {
    let mut tracker = TemporalFaceTracker::new(TrackerPolicy::realtime_adaptive());
    let mut detector_calls = 0;
    let mut detect = || -> anyhow::Result<Vec<DetectedFace>> {
        detector_calls += 1;
        Ok(vec![face(100.0 + detector_calls as f32, 100.0, 100.0)])
    };
    detect_or_track(&mut tracker, 1920, 1080, &mut detect).unwrap();
    detect_or_track(&mut tracker, 1920, 1080, &mut detect).unwrap();
    assert_eq!(tracker.detection_interval(), 2);
    detect_or_track(&mut tracker, 1920, 1080, &mut detect).unwrap();

    assert_eq!(detector_calls, 2);
}

#[test]
fn offline_recovery_route_still_invokes_the_detector_on_every_frame() {
    let mut tracker = TemporalFaceTracker::new(TrackerPolicy::offline_recovery());
    let mut detector_calls = 0;
    for _ in 0..4 {
        detect_or_track(&mut tracker, 640, 480, || {
            detector_calls += 1;
            Ok::<_, std::convert::Infallible>(vec![face(50.0, 50.0, 80.0)])
        })
        .unwrap();
    }
    assert_eq!(detector_calls, 4);
}
