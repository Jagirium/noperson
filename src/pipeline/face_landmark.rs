//! Face landmark detectors: 5/68/3d68/98/106/203/478 points.
//!
//! Port of crosswap/app/processors/face_landmark_detectors.py
//! Each model has unique input size, normalization, and output decoding.

use crate::backend::{Buffer, ComputeOps, DevicePtrMut};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use thiserror::Error;

use crate::config::parameters::LandmarkMode;
use crate::math::affine;
use crate::models::manager::ModelManager;
use crate::pipeline::ort_binding::{create_cuda_tensor_f32, run_bound_values};
use crate::pipeline::workspace::GpuWorkspace;

/// Landmark model types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LandmarkModel {
    /// res50.onnx — 5 points, input 512×512, norm: subtract BGR mean [104,117,123]
    Points5,
    /// 2dfan4.onnx — 68 points, input 256×256, norm: /255
    Points68,
    /// 1k3d68.onnx — 68 3D points, input 192×192, identity normalization
    Points3d68,
    /// peppapig — 98 points, input 256×256, norm: /255
    Points98,
    /// 2d106det.onnx — 106 points, input 192×192, identity normalization
    Points106,
    /// landmark.onnx — 203 points, input 224×224, norm: /255
    Points203,
    /// face_landmarks_detector — 478 MediaPipe points, input 256×256, norm: /255
    Points478,
}

impl From<LandmarkMode> for LandmarkModel {
    fn from(value: LandmarkMode) -> Self {
        match value {
            LandmarkMode::Points5 => Self::Points5,
            LandmarkMode::Points68 => Self::Points68,
            LandmarkMode::Points3d68 => Self::Points3d68,
            LandmarkMode::Points98 => Self::Points98,
            LandmarkMode::Points106 => Self::Points106,
            LandmarkMode::Points203 => Self::Points203,
            LandmarkMode::Points478 => Self::Points478,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LandmarkError {
    #[error("unsupported landmark mode {0}")]
    UnsupportedMode(String),
    #[error("landmark model expected {expected} points, got {actual}")]
    PointCount { expected: usize, actual: usize },
    #[error("landmark bounding box is empty")]
    InvalidBoundingBox,
}

#[derive(Debug, Clone)]
pub struct LandmarkResult {
    pub five: [[f32; 2]; 5],
    pub points: Vec<[f32; 2]>,
    pub scores: Vec<f32>,
}

impl LandmarkResult {
    /// CrossSwap keeps a scored refiner only when it beats the face detector;
    /// refiners without confidence output are always accepted.
    pub fn is_preferred_to(&self, detector_score: f32) -> bool {
        self.scores.is_empty()
            || self.scores.iter().copied().sum::<f32>() / (self.scores.len() as f32)
                > detector_score
    }
}

impl LandmarkModel {
    pub fn from_mode(mode: &str) -> Result<Self, LandmarkError> {
        match mode {
            "5" => Ok(Self::Points5),
            "68" => Ok(Self::Points68),
            "3d68" => Ok(Self::Points3d68),
            "98" => Ok(Self::Points98),
            "106" => Ok(Self::Points106),
            "203" => Ok(Self::Points203),
            "478" => Ok(Self::Points478),
            _ => Err(LandmarkError::UnsupportedMode(mode.to_owned())),
        }
    }

    pub fn model_name(&self) -> &'static str {
        match self {
            Self::Points5 => "FaceLandmark5",
            Self::Points68 => "FaceLandmark68",
            Self::Points3d68 => "FaceLandmark3d68",
            Self::Points98 => "FaceLandmark98",
            Self::Points106 => "FaceLandmark106",
            Self::Points203 => "FaceLandmark203",
            Self::Points478 => "FaceLandmark478",
        }
    }

    pub fn input_size(&self) -> u32 {
        match self {
            Self::Points5 => 512,
            Self::Points68 | Self::Points98 | Self::Points478 => 256,
            Self::Points3d68 | Self::Points106 => 192,
            Self::Points203 => 224,
        }
    }

    pub fn num_points(&self) -> usize {
        match self {
            Self::Points5 => 5,
            Self::Points68 | Self::Points3d68 => 68,
            Self::Points98 => 98,
            Self::Points106 => 106,
            Self::Points203 => 203,
            Self::Points478 => 478,
        }
    }

    pub fn onnx_filename(&self) -> &'static str {
        match self {
            Self::Points5 => "res50.onnx",
            Self::Points68 => "2dfan4.onnx",
            Self::Points3d68 => "1k3d68.onnx",
            Self::Points98 => "peppapig_teacher_Nx3x256x256.onnx",
            Self::Points106 => "2d106det.onnx",
            Self::Points203 => "landmark.onnx",
            Self::Points478 => "face_landmarks_detector_Nx3x256x256.onnx",
        }
    }

    pub fn input_name(&self) -> &'static str {
        match self {
            Self::Points3d68 | Self::Points106 => "data",
            Self::Points478 => "input_12",
            _ => "input",
        }
    }

    pub fn output_specs(&self) -> &'static [(&'static str, usize)] {
        match self {
            Self::Points5 => &[("conf", 21_504), ("landmarks", 107_520)],
            Self::Points68 => &[("landmarks_xyscore", 204), ("heatmaps", 278_528)],
            Self::Points3d68 => &[("fc1", 3_309)],
            Self::Points98 => &[("landmarks_xyscore", 294)],
            Self::Points106 => &[("fc1", 212)],
            Self::Points203 => &[("output", 214), ("853", 262), ("856", 406)],
            Self::Points478 => &[("Identity", 1_434), ("Identity_1", 1), ("Identity_2", 1)],
        }
    }

    pub fn bbox_affine(&self, bbox: [f32; 4]) -> Result<[[f64; 3]; 2], LandmarkError> {
        let bbox = bbox.map(f64::from);
        let width = bbox[2] - bbox[0];
        let height = bbox[3] - bbox[1];
        if width <= 0.0 || height <= 0.0 {
            return Err(LandmarkError::InvalidBoundingBox);
        }
        if *self == Self::Points68 {
            let scale = 195.0 / width.max(height);
            let tx = (256.0 - (bbox[2] + bbox[0]) * scale) * 0.5;
            let ty = (256.0 - (bbox[3] + bbox[1]) * scale) * 0.5;
            return Ok([[scale, 0.0, tx], [0.0, scale, ty]]);
        }
        if *self == Self::Points98 {
            if width <= 20.0 || height <= 20.0 {
                return Err(LandmarkError::InvalidBoundingBox);
            }
            let face_width = 1.4 * width;
            let center_x = ((bbox[0] + bbox[2]) * 0.5).floor();
            let center_y = ((bbox[1] + bbox[3]) * 0.5).floor();
            let half = (face_width * 0.5).floor();
            let left = center_x - half;
            let top = center_y - half;
            let crop_size = half * 2.0;
            let scale = 256.0 / crop_size;
            return Ok([[scale, 0.0, -left * scale], [0.0, scale, -top * scale]]);
        }
        let size = self.input_size() as f64;
        let scale = size / (width.max(height) * 1.5);
        let center_x = (bbox[2] + bbox[0]) * 0.5;
        let center_y = (bbox[3] + bbox[1]) * 0.5;
        Ok([
            [scale, 0.0, size * 0.5 - center_x * scale],
            [0.0, scale, size * 0.5 - center_y * scale],
        ])
    }

    /// Apply model-specific normalization to pixel values.
    pub fn normalize(&self, pixel: f32, channel: u32) -> f32 {
        match self {
            Self::Points5 => {
                // CrossSwap subtracts this vector after CHW→HWC, without a channel swap.
                let mean = match channel {
                    0 => 104.0,
                    1 => 117.0,
                    2 => 123.0,
                    _ => 0.0,
                };
                pixel - mean
            }
            Self::Points68 | Self::Points98 | Self::Points203 | Self::Points478 => pixel / 255.0,
            Self::Points3d68 | Self::Points106 => pixel,
        }
    }

    /// Reduce a model's dense output to CrossSwap's canonical ArcFace-5 order.
    pub fn to_five(&self, points: &[[f32; 2]]) -> Result<[[f32; 2]; 5], LandmarkError> {
        let expected = self.num_points();
        if points.len() != expected {
            return Err(LandmarkError::PointCount {
                expected,
                actual: points.len(),
            });
        }

        let result = match self {
            Self::Points5 => [points[0], points[1], points[2], points[3], points[4]],
            Self::Points68 | Self::Points3d68 => [
                mean_pair(points[36], points[39]),
                mean_pair(points[42], points[45]),
                points[30],
                points[48],
                points[54],
            ],
            Self::Points98 => [points[96], points[97], points[54], points[76], points[82]],
            Self::Points106 => [points[38], points[88], points[86], points[52], points[61]],
            Self::Points203 => [
                points[197],
                points[198],
                points[201],
                points[48],
                points[66],
            ],
            Self::Points478 => [points[468], points[473], points[4], points[61], points[291]],
        };
        Ok(result)
    }

    /// Reduce dense confidence scores with the same indices and eye averaging
    /// used by CrossSwap before applying the landmark score threshold.
    pub fn to_five_scores(&self, scores: &[f32]) -> Result<Vec<f32>, LandmarkError> {
        if scores.is_empty() {
            return Ok(Vec::new());
        }
        if *self == Self::Points5 {
            return Ok(scores.to_vec());
        }
        let expected = self.num_points();
        if scores.len() != expected {
            return Err(LandmarkError::PointCount {
                expected,
                actual: scores.len(),
            });
        }
        Ok(match self {
            Self::Points68 | Self::Points3d68 => vec![
                (scores[36] + scores[39]) * 0.5,
                (scores[42] + scores[45]) * 0.5,
                scores[30],
                scores[48],
                scores[54],
            ],
            Self::Points98 => vec![scores[96], scores[97], scores[54], scores[76], scores[82]],
            Self::Points106 | Self::Points203 | Self::Points478 => Vec::new(),
            Self::Points5 => unreachable!("handled above"),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn detect_gpu(
        self,
        manager: &mut ModelManager,
        gpu: &ComputeOps,
        workspace: &mut GpuWorkspace,
        frame: &Buffer<f32>,
        frame_height: u32,
        frame_width: u32,
        bbox: [f32; 4],
        detector_points: &[[f32; 2]; 5],
        from_points: bool,
        score_threshold: f32,
    ) -> anyhow::Result<Option<LandmarkResult>> {
        let size = self.input_size();
        let crop_affine = if from_points {
            landmark_points_affine(self, detector_points, size)
        } else {
            self.bbox_affine(bbox)?
        };
        gpu.warp_affine_npp(
            frame,
            &mut workspace.landmark_input,
            frame_height,
            frame_width,
            size,
            size,
            &crop_affine,
        )?;
        let normalization = match self {
            Self::Points5 => 0,
            Self::Points68 | Self::Points98 | Self::Points203 | Self::Points478 => 1,
            Self::Points3d68 | Self::Points106 => 2,
        };
        gpu.landmark_normalize(&mut workspace.landmark_input, size * size, normalization)?;
        run_landmark_session(self, manager, gpu, workspace)?;
        let inverse = affine::invert_2x3(&crop_affine);
        let result = decode_landmarks(self, workspace, &inverse)?;
        if !result.scores.is_empty()
            && result.scores.iter().copied().sum::<f32>() / (result.scores.len() as f32)
                < score_threshold
        {
            return Ok(None);
        }
        Ok(Some(result))
    }

    /// Run CrossSwap's restorer `Reference` five-point detector against the
    /// current 512x512 swapped crop held by the persistent workspace.
    pub fn detect_restorer_reference_gpu(
        self,
        manager: &mut ModelManager,
        gpu: &ComputeOps,
        workspace: &mut GpuWorkspace,
        score_threshold: f32,
    ) -> anyhow::Result<Option<LandmarkResult>> {
        debug_assert_eq!(self, Self::Points5);
        let crop_affine = self.bbox_affine([0.0, 0.0, 512.0, 512.0])?;
        gpu.warp_affine_npp(
            &workspace.face_512,
            &mut workspace.landmark_input,
            512,
            512,
            512,
            512,
            &crop_affine,
        )?;
        gpu.landmark_normalize(&mut workspace.landmark_input, 512 * 512, 0)?;
        run_landmark_session(self, manager, gpu, workspace)?;
        let inverse = affine::invert_2x3(&crop_affine);
        let result = decode_landmarks(self, workspace, &inverse)?;
        if !result.scores.is_empty()
            && result.scores.iter().copied().sum::<f32>() / (result.scores.len() as f32)
                < score_threshold
        {
            return Ok(None);
        }
        Ok(Some(result))
    }
}

fn landmark_points_affine(
    model: LandmarkModel,
    points: &[[f32; 2]; 5],
    size: u32,
) -> [[f64; 3]; 2] {
    if model == LandmarkModel::Points478 {
        const MAP: [[[f32; 2]; 5]; 5] = [
            [
                [51.642, 50.115],
                [57.617, 49.99],
                [35.74, 69.007],
                [51.157, 89.05],
                [57.025, 89.702],
            ],
            [
                [45.031, 50.118],
                [65.568, 50.872],
                [39.677, 68.111],
                [45.177, 86.19],
                [64.246, 86.758],
            ],
            [
                [39.73, 51.138],
                [72.27, 51.138],
                [56.0, 68.493],
                [42.463, 87.01],
                [69.537, 87.01],
            ],
            [
                [46.845, 50.872],
                [67.382, 50.118],
                [72.737, 68.111],
                [48.167, 86.758],
                [67.236, 86.19],
            ],
            [
                [54.796, 49.99],
                [60.771, 50.115],
                [76.673, 69.007],
                [55.388, 89.702],
                [61.257, 89.05],
            ],
        ];
        let factor = size as f32 / 112.0;
        return MAP
            .iter()
            .map(|template| {
                let scaled = template.map(|point| [point[0] * factor, point[1] * factor]);
                let matrix = affine::estimate_face_affine(points, &scaled);
                let error = points
                    .iter()
                    .zip(scaled)
                    .map(|(point, target)| {
                        let x = matrix[0][0] * point[0] as f64
                            + matrix[0][1] * point[1] as f64
                            + matrix[0][2];
                        let y = matrix[1][0] * point[0] as f64
                            + matrix[1][1] * point[1] as f64
                            + matrix[1][2];
                        (x - target[0] as f64).hypot(y - target[1] as f64)
                    })
                    .sum::<f64>();
                (error, matrix)
            })
            .min_by(|left, right| left.0.total_cmp(&right.0))
            .expect("landmark template map is non-empty")
            .1;
    }
    let factor = size as f32 / 128.0;
    let mut template = crate::math::constants::ARCFACE_DST;
    for point in &mut template {
        point[0] = point[0] * factor + 8.0 * factor;
        point[1] *= factor;
    }
    affine::estimate_face_affine(points, &template)
}

fn output_shapes(model: LandmarkModel) -> &'static [&'static [i64]] {
    match model {
        LandmarkModel::Points5 => &[&[1, 10_752, 2], &[1, 10_752, 10]],
        LandmarkModel::Points68 => &[&[1, 68, 3], &[1, 68, 64, 64]],
        LandmarkModel::Points3d68 => &[&[1, 3_309]],
        LandmarkModel::Points98 => &[&[1, 98, 3]],
        LandmarkModel::Points106 => &[&[1, 212]],
        LandmarkModel::Points203 => &[&[1, 214], &[1, 262], &[1, 406]],
        LandmarkModel::Points478 => &[&[1, 1, 1, 1_434], &[1, 1, 1, 1], &[1, 1]],
    }
}

fn run_landmark_session(
    model: LandmarkModel,
    manager: &mut ModelManager,
    gpu: &ComputeOps,
    workspace: &mut GpuWorkspace,
) -> anyhow::Result<()> {
    let size = model.input_size() as i64;
    {
        let memory = MemoryInfo::new(
            AllocationDevice::CUDA,
            manager.device_id(),
            AllocatorType::Device,
            MemoryType::Default,
        )?;
        let (input_ptr, _input_guard) = workspace.landmark_input.device_ptr_mut(&gpu.stream);
        let (a_ptr, _a_guard) = workspace.landmark_output_a.device_ptr_mut(&gpu.stream);
        let (b_ptr, _b_guard) = workspace.landmark_output_b.device_ptr_mut(&gpu.stream);
        let (c_ptr, _c_guard) = workspace.landmark_output_c.device_ptr_mut(&gpu.stream);
        let input = unsafe { create_cuda_tensor_f32(&memory, input_ptr, &[1, 3, size, size])? };
        let pointers = [a_ptr, b_ptr, c_ptr];
        let shapes = output_shapes(model);
        let outputs = pointers
            .iter()
            .zip(shapes)
            .map(|(pointer, shape)| unsafe { create_cuda_tensor_f32(&memory, *pointer, shape) })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let bound_outputs = model
            .output_specs()
            .iter()
            .zip(shapes)
            .zip(outputs.iter())
            .map(|(((name, _), _), output)| (*name, output))
            .collect::<Vec<_>>();
        run_bound_values(
            manager,
            &gpu.stream,
            model.model_name(),
            &[(model.input_name(), &input)],
            &bound_outputs,
        )?;
    }

    for ((index, (_, length)), buffer) in model.output_specs().iter().enumerate().zip([
        &workspace.landmark_output_a,
        &workspace.landmark_output_b,
        &workspace.landmark_output_c,
    ]) {
        let host = match index {
            0 => &mut workspace.host_landmark_a,
            1 => &mut workspace.host_landmark_b,
            _ => &mut workspace.host_landmark_c,
        };
        let view = buffer.slice(..*length);
        gpu.stream.memcpy_dtoh(&view, &mut host[..*length])?;
    }
    Ok(())
}

fn decode_landmarks(
    model: LandmarkModel,
    workspace: &GpuWorkspace,
    inverse: &[[f64; 3]; 2],
) -> anyhow::Result<LandmarkResult> {
    decode_landmark_outputs(
        model,
        &workspace.host_landmark_a,
        &workspace.host_landmark_b,
        &workspace.host_landmark_c,
        inverse,
    )
}

fn decode_landmark_outputs(
    model: LandmarkModel,
    output_a: &[f32],
    output_b: &[f32],
    output_c: &[f32],
    inverse: &[[f64; 3]; 2],
) -> anyhow::Result<LandmarkResult> {
    let (mut points, scores) = match model {
        LandmarkModel::Points5 => decode_points5(&output_a[..21_504], &output_b[..107_520])?,
        LandmarkModel::Points68 => {
            let points = output_a[..204]
                .chunks_exact(3)
                .map(|point| [point[0] * 4.0, point[1] * 4.0])
                .collect();
            let scores = output_b[..278_528]
                .chunks_exact(64 * 64)
                .map(|heatmap| heatmap.iter().copied().fold(f32::NEG_INFINITY, f32::max))
                .collect();
            (points, scores)
        }
        LandmarkModel::Points3d68 => {
            let raw = &output_a[..3_309];
            let start = raw.len() - 68 * 3;
            (
                raw[start..]
                    .chunks_exact(3)
                    .map(|point| [(point[0] + 1.0) * 96.0, (point[1] + 1.0) * 96.0])
                    .collect(),
                Vec::new(),
            )
        }
        LandmarkModel::Points98 => (
            output_a[..294]
                .chunks_exact(3)
                .map(|point| [point[0] * 256.0, point[1] * 256.0])
                .collect(),
            output_a[..294]
                .chunks_exact(3)
                .map(|point| point[2])
                .collect(),
        ),
        LandmarkModel::Points106 => (
            output_a[..212]
                .chunks_exact(2)
                .map(|point| [(point[0] + 1.0) * 96.0, (point[1] + 1.0) * 96.0])
                .collect(),
            Vec::new(),
        ),
        LandmarkModel::Points203 => (
            output_c[..406]
                .chunks_exact(2)
                .map(|point| [point[0] * 224.0, point[1] * 224.0])
                .collect(),
            Vec::new(),
        ),
        LandmarkModel::Points478 => (
            output_a[..1_434]
                .chunks_exact(3)
                .map(|point| [point[0], point[1]])
                .collect(),
            Vec::new(),
        ),
    };
    for point in &mut points {
        *point = transform_point(*point, inverse);
    }
    let five = model.to_five(&points)?;
    let scores = model.to_five_scores(&scores)?;
    Ok(LandmarkResult {
        five,
        points,
        scores,
    })
}

fn decode_points5(conf: &[f32], landmarks: &[f32]) -> anyhow::Result<(Vec<[f32; 2]>, Vec<f32>)> {
    let (best, score) = conf
        .chunks_exact(2)
        .enumerate()
        .map(|(index, value)| (index, value[1]))
        .filter(|(_, score)| *score > 0.1)
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .ok_or_else(|| anyhow::anyhow!("landmark-5 confidence is below threshold"))?;
    let prior = landmark5_prior(best)
        .ok_or_else(|| anyhow::anyhow!("landmark-5 prior index {best} is out of range"))?;
    let raw = &landmarks[best * 10..best * 10 + 10];
    let points = raw
        .chunks_exact(2)
        .map(|point| {
            [
                (prior[0] + point[0] * 0.1 * prior[2]) * 512.0,
                (prior[1] + point[1] * 0.1 * prior[3]) * 512.0,
            ]
        })
        .collect();
    Ok((points, vec![score]))
}

fn landmark5_prior(index: usize) -> Option<[f32; 4]> {
    let (local, width, step, sizes) = if index < 8_192 {
        (index, 64usize, 8.0f32, [16.0f32, 32.0])
    } else if index < 10_240 {
        (index - 8_192, 32, 16.0, [64.0, 128.0])
    } else if index < 10_752 {
        (index - 10_240, 16, 32.0, [256.0, 512.0])
    } else {
        return None;
    };
    let anchor = local % 2;
    let cell = local / 2;
    let x = cell % width;
    let y = cell / width;
    let size = sizes[anchor];
    Some([
        (x as f32 + 0.5) * step / 512.0,
        (y as f32 + 0.5) * step / 512.0,
        size / 512.0,
        size / 512.0,
    ])
}

fn transform_point(point: [f32; 2], matrix: &[[f64; 3]; 2]) -> [f32; 2] {
    [
        (matrix[0][0] * point[0] as f64 + matrix[0][1] * point[1] as f64 + matrix[0][2]) as f32,
        (matrix[1][0] * point[0] as f64 + matrix[1][1] * point[1] as f64 + matrix[1][2]) as f32,
    ]
}

fn mean_pair(left: [f32; 2], right: [f32; 2]) -> [f32; 2] {
    [(left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5]
}

#[cfg(test)]
mod tests {
    use super::{LandmarkModel, decode_landmark_outputs, decode_points5, landmark5_prior};

    const IDENTITY: [[f64; 3]; 2] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

    #[test]
    fn landmark5_prior_boundaries_match_crossswap_anchor_order() {
        assert_eq!(
            landmark5_prior(0),
            Some([0.0078125, 0.0078125, 0.03125, 0.03125])
        );
        assert_eq!(
            landmark5_prior(8_192),
            Some([0.015625, 0.015625, 0.125, 0.125])
        );
        assert_eq!(landmark5_prior(10_751), Some([0.96875, 0.96875, 1.0, 1.0]));
        assert_eq!(landmark5_prior(10_752), None);
    }

    #[test]
    fn landmark5_decoder_selects_highest_anchor_without_allocating_priors() {
        let mut conf = vec![0.0; 21_504];
        conf[8_192 * 2 + 1] = 0.9;
        let landmarks = vec![0.0; 107_520];
        let (points, scores) = decode_points5(&conf, &landmarks).unwrap();
        assert_eq!(points, vec![[8.0, 8.0]; 5]);
        assert_eq!(scores, vec![0.9]);
    }

    #[test]
    fn dense_decoders_apply_model_scales_before_inverse_affine() {
        let raw_106 = vec![0.0; 212];
        let decoded =
            decode_landmark_outputs(LandmarkModel::Points106, &raw_106, &[], &[], &IDENTITY)
                .unwrap();
        assert_eq!(decoded.points, vec![[96.0, 96.0]; 106]);

        let mut raw_203 = vec![0.0; 406];
        for point in raw_203.chunks_exact_mut(2) {
            point[0] = 0.5;
            point[1] = 0.25;
        }
        let decoded =
            decode_landmark_outputs(LandmarkModel::Points203, &[], &[], &raw_203, &IDENTITY)
                .unwrap();
        assert_eq!(decoded.points, vec![[112.0, 56.0]; 203]);

        let mut raw_98 = vec![0.0; 294];
        for point in raw_98.chunks_exact_mut(3) {
            point.copy_from_slice(&[0.25, 0.75, 0.9]);
        }
        let decoded =
            decode_landmark_outputs(LandmarkModel::Points98, &raw_98, &[], &[], &IDENTITY).unwrap();
        assert_eq!(decoded.points, vec![[64.0, 192.0]; 98]);
        assert_eq!(decoded.scores, vec![0.9; 5]);
    }
}
