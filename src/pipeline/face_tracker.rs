//! Deterministic temporal face tracking for detector scheduling and recovery.

use crate::pipeline::face_detector::DetectedFace;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackerPolicy {
    adaptive: bool,
    max_interval: u32,
    max_missed_observations: u32,
    association_iou: f32,
    stable_iou: f32,
}

impl TrackerPolicy {
    pub const fn realtime_adaptive() -> Self {
        Self {
            adaptive: true,
            max_interval: 4,
            max_missed_observations: 2,
            association_iou: 0.30,
            stable_iou: 0.75,
        }
    }

    pub const fn offline_recovery() -> Self {
        Self {
            adaptive: false,
            max_interval: 1,
            max_missed_observations: 2,
            association_iou: 0.30,
            stable_iou: 0.75,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TrackerFrame {
    Detect,
    Tracked(Vec<DetectedFace>),
}

pub fn detect_or_track<E>(
    tracker: &mut TemporalFaceTracker,
    width: u32,
    height: u32,
    detect: impl FnOnce() -> Result<Vec<DetectedFace>, E>,
) -> Result<Vec<DetectedFace>, E> {
    match tracker.begin_frame(width, height) {
        TrackerFrame::Detect => detect().map(|detections| tracker.update(detections)),
        TrackerFrame::Tracked(faces) => Ok(faces),
    }
}

#[derive(Debug, Clone)]
struct FaceTrack {
    id: u64,
    face: DetectedFace,
    observed: DetectedFace,
    bbox_velocity: [f32; 4],
    keypoint_velocity: [[f32; 2]; 5],
    observations: u32,
    missed_observations: u32,
}

impl FaceTrack {
    fn new(id: u64, face: DetectedFace) -> Self {
        Self {
            id,
            observed: face.clone(),
            face,
            bbox_velocity: [0.0; 4],
            keypoint_velocity: [[0.0; 2]; 5],
            observations: 1,
            missed_observations: 0,
        }
    }

    fn advance(&mut self, width: u32, height: u32) {
        for (value, velocity) in self.face.bbox.iter_mut().zip(self.bbox_velocity) {
            *value += velocity;
        }
        for (point, velocity) in self.face.kps_5.iter_mut().zip(self.keypoint_velocity) {
            point[0] += velocity[0];
            point[1] += velocity[1];
        }
        clamp_face(&mut self.face, width, height);
    }

    fn observe(&mut self, detection: DetectedFace, elapsed_frames: u32) {
        let elapsed = elapsed_frames.max(1) as f32;
        let measured_bbox = std::array::from_fn(|index| {
            (detection.bbox[index] - self.observed.bbox[index]) / elapsed
        });
        let measured_keypoints = std::array::from_fn(|point| {
            std::array::from_fn(|axis| {
                (detection.kps_5[point][axis] - self.observed.kps_5[point][axis]) / elapsed
            })
        });
        if self.observations == 1 {
            self.bbox_velocity = measured_bbox;
            self.keypoint_velocity = measured_keypoints;
        } else {
            for (velocity, measured) in self.bbox_velocity.iter_mut().zip(measured_bbox) {
                *velocity = 0.65 * *velocity + 0.35 * measured;
            }
            for (velocity, measured) in self.keypoint_velocity.iter_mut().zip(measured_keypoints) {
                velocity[0] = 0.65 * velocity[0] + 0.35 * measured[0];
                velocity[1] = 0.65 * velocity[1] + 0.35 * measured[1];
            }
        }
        self.observed = detection.clone();
        self.face = detection;
        self.observations = self.observations.saturating_add(1);
        self.missed_observations = 0;
    }
}

/// One tracker belongs to one immutable engine generation. Geometry changes
/// and generation swaps therefore cannot leak stale face locations.
#[derive(Debug, Clone)]
pub struct TemporalFaceTracker {
    policy: TrackerPolicy,
    geometry: Option<(u32, u32)>,
    tracks: Vec<FaceTrack>,
    next_id: u64,
    detection_interval: u32,
    frames_since_detection: u32,
}

impl TemporalFaceTracker {
    pub const fn new(policy: TrackerPolicy) -> Self {
        Self {
            policy,
            geometry: None,
            tracks: Vec::new(),
            next_id: 0,
            detection_interval: 1,
            frames_since_detection: 0,
        }
    }

    pub fn set_policy(&mut self, policy: TrackerPolicy) {
        self.policy = policy;
        self.reset();
    }

    pub fn reset(&mut self) {
        self.geometry = None;
        self.tracks.clear();
        self.detection_interval = 1;
        self.frames_since_detection = 0;
    }

    pub fn begin_frame(&mut self, width: u32, height: u32) -> TrackerFrame {
        if self.geometry != Some((width, height)) {
            self.reset();
            self.geometry = Some((width, height));
            return TrackerFrame::Detect;
        }
        if self.tracks.is_empty() {
            self.frames_since_detection = 0;
            return TrackerFrame::Detect;
        }
        for track in &mut self.tracks {
            track.advance(width, height);
        }
        self.frames_since_detection = self.frames_since_detection.saturating_add(1);
        if !self.policy.adaptive || self.frames_since_detection >= self.detection_interval {
            TrackerFrame::Detect
        } else {
            TrackerFrame::Tracked(self.faces())
        }
    }

    pub fn update(&mut self, detections: Vec<DetectedFace>) -> Vec<DetectedFace> {
        let elapsed_frames = self.frames_since_detection.max(1);
        let existing_count = self.tracks.len();
        let mut candidates = Vec::new();
        for (track_index, track) in self.tracks.iter().enumerate() {
            for (detection_index, detection) in detections.iter().enumerate() {
                let overlap = intersection_over_union(track.face.bbox, detection.bbox);
                if overlap >= self.policy.association_iou {
                    candidates.push((overlap, track_index, detection_index));
                }
            }
        }
        candidates.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });

        let mut matched_tracks = vec![false; existing_count];
        let mut matched_detections = vec![false; detections.len()];
        let mut matched_iou = Vec::new();
        for (overlap, track_index, detection_index) in candidates {
            if matched_tracks[track_index] || matched_detections[detection_index] {
                continue;
            }
            matched_tracks[track_index] = true;
            matched_detections[detection_index] = true;
            matched_iou.push(overlap);
            self.tracks[track_index].observe(detections[detection_index].clone(), elapsed_frames);
        }

        for (index, track) in self.tracks.iter_mut().take(existing_count).enumerate() {
            if !matched_tracks[index] {
                track.missed_observations = track.missed_observations.saturating_add(1);
                track.face.score *= 0.90;
            }
        }
        self.tracks
            .retain(|track| track.missed_observations <= self.policy.max_missed_observations);
        for (index, detection) in detections.into_iter().enumerate() {
            if matched_detections[index] {
                continue;
            }
            let id = self.next_id;
            self.next_id = self.next_id.saturating_add(1);
            self.tracks.push(FaceTrack::new(id, detection));
        }
        self.tracks.sort_by_key(|track| track.id);

        let stable = existing_count > 0
            && matched_iou.len() == existing_count
            && matched_detections.iter().all(|matched| *matched)
            && matched_iou
                .iter()
                .all(|overlap| *overlap >= self.policy.stable_iou);
        self.detection_interval = if self.policy.adaptive && stable {
            self.detection_interval
                .saturating_add(1)
                .min(self.policy.max_interval)
        } else {
            1
        };
        self.frames_since_detection = 0;
        self.faces()
    }

    pub const fn detection_interval(&self) -> u32 {
        self.detection_interval
    }

    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    fn faces(&self) -> Vec<DetectedFace> {
        self.tracks.iter().map(|track| track.face.clone()).collect()
    }
}

fn intersection_over_union(left: [f32; 4], right: [f32; 4]) -> f32 {
    let intersection_width = (left[2].min(right[2]) - left[0].max(right[0])).max(0.0);
    let intersection_height = (left[3].min(right[3]) - left[1].max(right[1])).max(0.0);
    let intersection = intersection_width * intersection_height;
    let left_area = ((left[2] - left[0]).max(0.0)) * ((left[3] - left[1]).max(0.0));
    let right_area = ((right[2] - right[0]).max(0.0)) * ((right[3] - right[1]).max(0.0));
    let union = left_area + right_area - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

fn clamp_face(face: &mut DetectedFace, width: u32, height: u32) {
    let max_x = width.saturating_sub(1) as f32;
    let max_y = height.saturating_sub(1) as f32;
    let dx = if face.bbox[0] < 0.0 {
        -face.bbox[0]
    } else if face.bbox[2] > max_x {
        max_x - face.bbox[2]
    } else {
        0.0
    };
    let dy = if face.bbox[1] < 0.0 {
        -face.bbox[1]
    } else if face.bbox[3] > max_y {
        max_y - face.bbox[3]
    } else {
        0.0
    };
    face.bbox[0] += dx;
    face.bbox[2] += dx;
    face.bbox[1] += dy;
    face.bbox[3] += dy;
    for point in &mut face.kps_5 {
        point[0] = (point[0] + dx).clamp(0.0, max_x);
        point[1] = (point[1] + dy).clamp(0.0, max_y);
    }
}
