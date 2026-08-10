//! Face detection: YoloFace8n, RetinaFace, SCRFD.
//!
//! Port of crosswap/app/processors/face_detectors.py
//! All three detectors share the same DetectedFace output format.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use std::cell::RefCell;

use crate::config::settings::DetectorModel;
use crate::gpu::ops::GpuOps;
use crate::models::manager::ModelManager;
use crate::pipeline::ort_binding::{create_cuda_tensor_f32, run_bound_values};
use crate::pipeline::workspace::GpuWorkspace;

/// Detected face with bounding box and 5-point landmarks.
#[derive(Debug, Clone)]
pub struct DetectedFace {
    pub bbox: [f32; 4],       // [x1, y1, x2, y2] in original frame coords
    pub kps_5: [[f32; 2]; 5], // left_eye, right_eye, nose, left_mouth, right_mouth
    pub score: f32,
}

pub trait FaceDetectorBackend {
    fn detect_gpu(
        &self,
        mgr: &mut ModelManager,
        gpu: &GpuOps,
        frame_chw: &CudaSlice<f32>,
        ws: &mut GpuWorkspace,
        frame_h: u32,
        frame_w: u32,
    ) -> anyhow::Result<(Vec<DetectedFace>, f32)>;

    fn detect_gpu_auto_rotation(
        &self,
        mgr: &mut ModelManager,
        gpu: &GpuOps,
        frame_chw: &CudaSlice<f32>,
        ws: &mut GpuWorkspace,
        frame_h: u32,
        frame_w: u32,
    ) -> anyhow::Result<(Vec<DetectedFace>, f32)> {
        self.detect_gpu(mgr, gpu, frame_chw, ws, frame_h, frame_w)
    }
}

pub enum FaceDetector {
    Yolo(YoloFaceDetector),
    Anchor(AnchorDetector),
}

impl FaceDetector {
    pub fn from_model(model: DetectorModel, score_threshold: f32) -> Self {
        match model {
            DetectorModel::YoloFace8n => Self::Yolo(YoloFaceDetector::new(score_threshold)),
            DetectorModel::RetinaFace => Self::Anchor(AnchorDetector::new(
                AnchorDetectorKind::RetinaFace,
                score_threshold,
            )),
            DetectorModel::Scrfd2_5g => Self::Anchor(AnchorDetector::new(
                AnchorDetectorKind::Scrfd,
                score_threshold,
            )),
        }
    }

    pub fn configured(model: DetectorModel, score_threshold: f32, max_faces: usize) -> Self {
        match model {
            DetectorModel::YoloFace8n => {
                Self::Yolo(YoloFaceDetector::new(score_threshold).with_max_faces(max_faces))
            }
            DetectorModel::RetinaFace => Self::Anchor(
                AnchorDetector::new(AnchorDetectorKind::RetinaFace, score_threshold)
                    .with_max_faces(max_faces),
            ),
            DetectorModel::Scrfd2_5g => Self::Anchor(
                AnchorDetector::new(AnchorDetectorKind::Scrfd, score_threshold)
                    .with_max_faces(max_faces),
            ),
        }
    }

    pub fn model(&self) -> DetectorModel {
        match self {
            Self::Yolo(_) => DetectorModel::YoloFace8n,
            Self::Anchor(detector) => match detector.kind {
                AnchorDetectorKind::RetinaFace => DetectorModel::RetinaFace,
                AnchorDetectorKind::Scrfd => DetectorModel::Scrfd2_5g,
            },
        }
    }
}

impl FaceDetectorBackend for FaceDetector {
    fn detect_gpu(
        &self,
        mgr: &mut ModelManager,
        gpu: &GpuOps,
        frame_chw: &CudaSlice<f32>,
        ws: &mut GpuWorkspace,
        frame_h: u32,
        frame_w: u32,
    ) -> anyhow::Result<(Vec<DetectedFace>, f32)> {
        match self {
            Self::Yolo(detector) => detector.detect_gpu(mgr, gpu, frame_chw, ws, frame_h, frame_w),
            Self::Anchor(detector) => {
                detector.detect_gpu(mgr, gpu, frame_chw, ws, frame_h, frame_w)
            }
        }
    }

    fn detect_gpu_auto_rotation(
        &self,
        mgr: &mut ModelManager,
        gpu: &GpuOps,
        frame_chw: &CudaSlice<f32>,
        ws: &mut GpuWorkspace,
        frame_h: u32,
        frame_w: u32,
    ) -> anyhow::Result<(Vec<DetectedFace>, f32)> {
        let (mut combined, scale) = self.detect_gpu(mgr, gpu, frame_chw, ws, frame_h, frame_w)?;
        let elements = 3 * frame_h as usize * frame_w as usize;
        let mut rotated = match self {
            Self::Yolo(detector) => detector.rotation_scratch.borrow_mut().take(),
            Self::Anchor(detector) => detector.rotation_scratch.borrow_mut().take(),
        }
        .filter(|buffer| buffer.len() == elements)
        .map_or_else(|| gpu.alloc_zeros(elements), Ok)?;
        for turns in 1..4 {
            let (rotated_h, rotated_w) = if turns % 2 == 0 {
                (frame_h, frame_w)
            } else {
                (frame_w, frame_h)
            };
            gpu.rotate_quadrants(frame_chw, &mut rotated, frame_h, frame_w, turns)?;
            let (faces, _) = self.detect_gpu(mgr, gpu, &rotated, ws, rotated_h, rotated_w)?;
            combined.extend(
                faces
                    .into_iter()
                    .map(|face| unrotate_detection(face, frame_h, frame_w, turns)),
            );
        }
        combined.sort_by(|left, right| right.score.total_cmp(&left.score));
        nms(&mut combined, 0.4);
        let max_faces = match self {
            Self::Yolo(detector) => detector.max_faces,
            Self::Anchor(detector) => detector.max_faces,
        };
        select_center_faces(&mut combined, frame_h, frame_w, max_faces);
        match self {
            Self::Yolo(detector) => *detector.rotation_scratch.borrow_mut() = Some(rotated),
            Self::Anchor(detector) => *detector.rotation_scratch.borrow_mut() = Some(rotated),
        }
        Ok((combined, scale))
    }
}

fn unrotate_point(point: [f32; 2], height: u32, width: u32, turns: u32) -> [f32; 2] {
    let [x, y] = point;
    match turns & 3 {
        1 => [width as f32 - 1.0 - y, x],
        2 => [width as f32 - 1.0 - x, height as f32 - 1.0 - y],
        3 => [y, height as f32 - 1.0 - x],
        _ => point,
    }
}

fn unrotate_detection(mut face: DetectedFace, height: u32, width: u32, turns: u32) -> DetectedFace {
    face.kps_5 = face
        .kps_5
        .map(|point| unrotate_point(point, height, width, turns));
    let [x1, y1, x2, y2] = face.bbox;
    let corners = [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]
        .map(|point| unrotate_point(point, height, width, turns));
    face.bbox = [
        corners
            .iter()
            .map(|point| point[0])
            .fold(f32::MAX, f32::min),
        corners
            .iter()
            .map(|point| point[1])
            .fold(f32::MAX, f32::min),
        corners
            .iter()
            .map(|point| point[0])
            .fold(f32::MIN, f32::max),
        corners
            .iter()
            .map(|point| point[1])
            .fold(f32::MIN, f32::max),
    ];
    face
}

// ═══════════════════════════════════════════════════════════════════════
// YoloFace8n
// ═══════════════════════════════════════════════════════════════════════

pub struct YoloFaceDetector {
    input_size: u32,
    score_threshold: f32,
    nms_iou_threshold: f32,
    max_faces: usize,
    rotation_scratch: RefCell<Option<CudaSlice<f32>>>,
}

impl YoloFaceDetector {
    pub fn new(score_threshold: f32) -> Self {
        Self {
            input_size: 640,
            score_threshold,
            nms_iou_threshold: 0.4,
            max_faces: 0,
            rotation_scratch: RefCell::new(None),
        }
    }

    pub fn with_max_faces(mut self, max_faces: usize) -> Self {
        self.max_faces = max_faces;
        self
    }

    /// Preprocess frame [3, H, W] in [0, 255] → [3, 640, 640] in [0, 1].
    pub fn preprocess(&self, img: &[f32], h: u32, w: u32) -> (Vec<f32>, f32) {
        letterbox_resize_normalize(img, h, w, self.input_size, 1.0 / 255.0, 0.0)
    }

    pub fn detect(
        &self,
        mgr: &mut ModelManager,
        input: &[f32],
        scale: f32,
    ) -> anyhow::Result<Vec<DetectedFace>> {
        let is = self.input_size as usize;
        let session = mgr
            .get_mut("YoloFace8n")
            .ok_or_else(|| anyhow::anyhow!("YoloFace8n not loaded"))?;
        let tensor = ort::value::Tensor::from_array(([1, 3, is, is], input.to_vec()))?;
        let outputs = session.run(ort::inputs!["images" => tensor])?;
        let (shape, data) = outputs["output0"].try_extract_tensor::<f32>()?;
        self.decode_yolo(data, shape, scale)
    }

    /// GPU-native detect via IoBinding: zero GPU→CPU roundtrip for input.
    ///
    /// Old path: letterbox on GPU → download to CPU → `Tensor::from_array` → ort
    /// re-uploads to GPU → inference → output back to CPU. Two full transfers.
    ///
    /// New path: letterbox on GPU → bind device pointer directly to ort input →
    /// bind pre-allocated GPU output buffer → `run_binding` → download only the
    /// small output for CPU decoding. One transfer (output only, 168000 f32).
    pub fn detect_gpu(
        &self,
        mgr: &mut ModelManager,
        gpu: &GpuOps,
        frame_chw: &CudaSlice<f32>,
        ws: &mut GpuWorkspace,
        frame_h: u32,
        frame_w: u32,
    ) -> anyhow::Result<(Vec<DetectedFace>, f32)> {
        let target = self.input_size;
        let (new_w, new_h, det_scale) = compute_letterbox_dims(frame_h, frame_w, target);

        // 1. GPU: resize frame_chw → detect_input with /255 normalization.
        gpu.letterbox_resize(
            frame_chw,
            &mut ws.detect_input,
            frame_h,
            frame_w,
            target,
            new_h,
            new_w,
            1.0 / 255.0,
            0.0,
        )?;

        // 2. IoBinding inference — input and output both live on GPU.
        let is = target as i64;
        let input_shape = [1i64, 3, is, is];
        let output_shape = [1i64, 20, 8400];

        let cuda_mem_info = MemoryInfo::new(
            AllocationDevice::CUDA,
            mgr.device_id(),
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| anyhow::anyhow!("MemoryInfo: {e}"))?;

        // IoBinding scope — constrain all mutable borrows to this block so
        // the next `download_into` can re-borrow `ws.detect_output` immutably.
        {
            let (input_dev, _g1) = ws.detect_input.device_ptr(&gpu.stream);
            let (output_dev, _g2) = ws.detect_output.device_ptr_mut(&gpu.stream);

            let input_value =
                unsafe { create_cuda_tensor_f32(&cuda_mem_info, input_dev, &input_shape)? };
            let output_value =
                unsafe { create_cuda_tensor_f32(&cuda_mem_info, output_dev, &output_shape)? };

            run_bound_values(
                mgr,
                &gpu.stream,
                "YoloFace8n",
                &[("images", &input_value)],
                &[("output0", &output_value)],
            )?;

            drop(input_value);
            drop(output_value);
        }

        // 3. Deterministically compact thresholded candidates on-device. Most
        // frames now transfer tens of floats instead of all 168000 outputs.
        gpu.compact_yolo_faces(
            &ws.detect_output,
            &mut ws.detect_candidates,
            &mut ws.detect_candidate_count,
            self.score_threshold,
            det_scale,
        )?;
        let mut count = [0u32; 1];
        gpu.stream
            .memcpy_dtoh(&ws.detect_candidate_count, &mut count)?;
        let count = count[0] as usize;
        anyhow::ensure!(
            count <= 8400,
            "YOLO candidate count exceeds output capacity"
        );
        let elements = count * 15;
        if elements > 0 {
            let candidates = ws.detect_candidates.slice(..elements);
            gpu.stream
                .memcpy_dtoh(&candidates, &mut ws.host_detect_candidates[..elements])?;
        }
        let mut faces = self.decode_compact_yolo(&ws.host_detect_candidates, count);
        select_center_faces(&mut faces, frame_h, frame_w, self.max_faces);
        Ok((faces, det_scale))
    }

    fn decode_compact_yolo(&self, data: &[f32], count: usize) -> Vec<DetectedFace> {
        decode_compact_candidates(data, count, self.nms_iou_threshold)
    }

    fn decode_yolo(
        &self,
        data: &[f32],
        shape: &[i64],
        scale: f32,
    ) -> anyhow::Result<Vec<DetectedFace>> {
        let n = shape[2] as usize;
        let mut dets = Vec::new();
        for i in 0..n {
            let score = data[4 * n + i];
            if score <= self.score_threshold {
                continue;
            }
            let cx = data[i];
            let cy = data[n + i];
            let w = data[2 * n + i];
            let h = data[3 * n + i];
            let mut kps = [[0.0f32; 2]; 5];
            for k in 0..5 {
                kps[k][0] = data[(5 + k * 3) * n + i] / scale;
                kps[k][1] = data[(5 + k * 3 + 1) * n + i] / scale;
            }
            dets.push(DetectedFace {
                bbox: [
                    (cx - w / 2.0) / scale,
                    (cy - h / 2.0) / scale,
                    (cx + w / 2.0) / scale,
                    (cy + h / 2.0) / scale,
                ],
                kps_5: kps,
                score,
            });
        }
        dets.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        nms(&mut dets, self.nms_iou_threshold);
        Ok(dets)
    }
}

impl FaceDetectorBackend for YoloFaceDetector {
    fn detect_gpu(
        &self,
        mgr: &mut ModelManager,
        gpu: &GpuOps,
        frame_chw: &CudaSlice<f32>,
        ws: &mut GpuWorkspace,
        frame_h: u32,
        frame_w: u32,
    ) -> anyhow::Result<(Vec<DetectedFace>, f32)> {
        YoloFaceDetector::detect_gpu(self, mgr, gpu, frame_chw, ws, frame_h, frame_w)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RetinaFace / SCRFD — anchor-based 3-stride detectors
// ═══════════════════════════════════════════════════════════════════════

pub struct AnchorDetector {
    input_size: u32,
    score_threshold: f32,
    nms_iou_threshold: f32,
    kind: AnchorDetectorKind,
    max_faces: usize,
    rotation_scratch: RefCell<Option<CudaSlice<f32>>>,
}

#[derive(Clone, Copy)]
pub enum AnchorDetectorKind {
    RetinaFace,
    Scrfd,
}

impl AnchorDetectorKind {
    pub const fn input_name(self) -> &'static str {
        "input.1"
    }

    pub const fn output_names(self) -> [&'static str; 9] {
        match self {
            Self::RetinaFace => [
                "448", "471", "494", "451", "474", "497", "454", "477", "500",
            ],
            Self::Scrfd => [
                "446", "466", "486", "449", "469", "489", "452", "472", "492",
            ],
        }
    }

    pub const fn output_lengths() -> [usize; 9] {
        [
            12_800, 3_200, 800, 51_200, 12_800, 3_200, 128_000, 32_000, 8_000,
        ]
    }

    pub const fn packed_output_len() -> usize {
        252_000
    }
}

impl AnchorDetector {
    pub fn new(kind: AnchorDetectorKind, score_threshold: f32) -> Self {
        Self {
            input_size: 512,
            score_threshold,
            nms_iou_threshold: 0.4,
            kind,
            max_faces: 0,
            rotation_scratch: RefCell::new(None),
        }
    }

    pub fn with_max_faces(mut self, max_faces: usize) -> Self {
        self.max_faces = max_faces;
        self
    }

    /// Preprocess frame [3, H, W] in [0, 255] → [3, 512, 512].
    /// RetinaFace: (pixel - 127.5) / 128. SCRFD: no normalization (raw pixels from resize).
    pub fn preprocess(&self, img: &[f32], h: u32, w: u32) -> (Vec<f32>, f32) {
        match self.kind {
            AnchorDetectorKind::RetinaFace => {
                letterbox_resize_normalize(img, h, w, self.input_size, 1.0 / 128.0, -127.5 / 128.0)
            }
            AnchorDetectorKind::Scrfd => {
                letterbox_resize_normalize(img, h, w, self.input_size, 1.0, 0.0)
            }
        }
    }

    pub fn detect(
        &self,
        mgr: &mut ModelManager,
        input: &[f32],
        scale: f32,
    ) -> anyhow::Result<Vec<DetectedFace>> {
        let is = self.input_size as usize;
        let model_name = match self.kind {
            AnchorDetectorKind::RetinaFace => "RetinaFace",
            AnchorDetectorKind::Scrfd => "SCRFD2.5g",
        };
        let session = mgr
            .get_mut(model_name)
            .ok_or_else(|| anyhow::anyhow!("{model_name} not loaded"))?;

        let tensor = ort::value::Tensor::from_array(([1, 3, is, is], input.to_vec()))?;

        // Get names before mutable borrow
        let input_name = match self.kind {
            AnchorDetectorKind::RetinaFace => "input.1".to_string(),
            AnchorDetectorKind::Scrfd => session
                .inputs()
                .first()
                .map(|i| i.name().to_string())
                .unwrap_or_else(|| "input.1".to_string()),
        };
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        let outputs = session.run(ort::inputs![&input_name => tensor])?;
        let mut all_outputs: Vec<Vec<f32>> = Vec::new();
        for name in &output_names {
            let (_, data) = outputs[name.as_str()].try_extract_tensor::<f32>()?;
            all_outputs.push(data.to_vec());
        }

        let output_refs: Vec<&[f32]> = all_outputs.iter().map(Vec::as_slice).collect();
        self.decode_anchor(&output_refs, is as u32, scale)
    }

    pub fn detect_gpu(
        &self,
        mgr: &mut ModelManager,
        gpu: &GpuOps,
        frame_chw: &CudaSlice<f32>,
        ws: &mut GpuWorkspace,
        frame_h: u32,
        frame_w: u32,
    ) -> anyhow::Result<(Vec<DetectedFace>, f32)> {
        let target = self.input_size;
        let (new_w, new_h, det_scale) = compute_letterbox_dims(frame_h, frame_w, target);
        let (mul, add) = match self.kind {
            AnchorDetectorKind::RetinaFace => (1.0 / 128.0, -127.5 / 128.0),
            AnchorDetectorKind::Scrfd => (1.0, 0.0),
        };
        gpu.letterbox_resize(
            frame_chw,
            &mut ws.detect_input,
            frame_h,
            frame_w,
            target,
            new_h,
            new_w,
            mul,
            add,
        )?;

        let memory = MemoryInfo::new(
            AllocationDevice::CUDA,
            mgr.device_id(),
            AllocatorType::Device,
            MemoryType::Default,
        )?;
        {
            let (input_ptr, _input_guard) = ws.detect_input.device_ptr(&gpu.stream);
            let (output_ptr, _output_guard) = ws.anchor_output.device_ptr_mut(&gpu.stream);
            let input = unsafe {
                create_cuda_tensor_f32(&memory, input_ptr, &[1, 3, target as i64, target as i64])?
            };
            let lengths = AnchorDetectorKind::output_lengths();
            let columns = [1usize, 1, 1, 4, 4, 4, 10, 10, 10];
            let mut offset = 0usize;
            let mut outputs = Vec::with_capacity(9);
            for (length, columns) in lengths.into_iter().zip(columns) {
                let pointer = output_ptr + (offset * core::mem::size_of::<f32>()) as u64;
                let rows = length / columns;
                outputs.push(unsafe {
                    create_cuda_tensor_f32(&memory, pointer, &[rows as i64, columns as i64])?
                });
                offset += length;
            }
            debug_assert_eq!(offset, AnchorDetectorKind::packed_output_len());

            let model_name = match self.kind {
                AnchorDetectorKind::RetinaFace => "RetinaFace",
                AnchorDetectorKind::Scrfd => "SCRFD2.5g",
            };
            let output_names = self.kind.output_names();
            let bound_outputs = output_names
                .into_iter()
                .zip(outputs.iter())
                .collect::<Vec<_>>();
            run_bound_values(
                mgr,
                &gpu.stream,
                model_name,
                &[(self.kind.input_name(), &input)],
                &bound_outputs,
            )?;
        }

        gpu.compact_anchor_faces(
            &ws.anchor_output,
            &mut ws.detect_candidates,
            &mut ws.detect_candidate_count,
            self.score_threshold,
            det_scale,
        )?;
        let mut count = [0u32; 1];
        gpu.stream
            .memcpy_dtoh(&ws.detect_candidate_count, &mut count)?;
        let count = count[0] as usize;
        anyhow::ensure!(
            count <= 16_800,
            "anchor candidate count exceeds output capacity"
        );
        let elements = count * 15;
        if elements > 0 {
            let candidates = ws.detect_candidates.slice(..elements);
            gpu.stream
                .memcpy_dtoh(&candidates, &mut ws.host_detect_candidates[..elements])?;
        }
        let mut faces =
            decode_compact_candidates(&ws.host_detect_candidates, count, self.nms_iou_threshold);
        select_center_faces(&mut faces, frame_h, frame_w, self.max_faces);
        Ok((faces, det_scale))
    }

    /// Decode 3-stride anchor-based outputs.
    /// Output layout: [scores_s8, scores_s16, scores_s32, bbox_s8, bbox_s16, bbox_s32, kps_s8, kps_s16, kps_s32]
    fn decode_anchor(
        &self,
        outputs: &[&[f32]],
        input_size: u32,
        det_scale: f32,
    ) -> anyhow::Result<Vec<DetectedFace>> {
        let strides = [8u32, 16, 32];
        let fmc = 3; // feature map count
        let mut dets = Vec::new();

        for (idx, &stride) in strides.iter().enumerate() {
            let feat_h = input_size / stride;
            let feat_w = input_size / stride;
            let n_anchors_per_loc = 2u32;
            let n_anchors = (feat_h * feat_w * n_anchors_per_loc) as usize;

            let scores = &outputs[idx]; // [1, N, 1] flattened
            let bboxes = &outputs[idx + fmc]; // [1, N, 4] flattened
            let kpss = &outputs[idx + fmc * 2]; // [1, N, 10] flattened

            // Build anchor grid: each (x, y) in pixel space, 2 anchors per location
            let mut anchors = Vec::with_capacity(n_anchors);
            for gy in 0..feat_h {
                for gx in 0..feat_w {
                    let cx = gx as f32 * stride as f32;
                    let cy = gy as f32 * stride as f32;
                    anchors.push((cx, cy));
                    anchors.push((cx, cy)); // 2 anchors per location
                }
            }

            for i in 0..n_anchors {
                let score = scores[i]; // score is already sigmoid in model output
                if score < self.score_threshold {
                    continue;
                }

                let (ax, ay) = anchors[i];
                let s = stride as f32;

                // BBox decode: anchor ± offset*stride
                let dl = bboxes[i * 4] * s;
                let dt = bboxes[i * 4 + 1] * s;
                let dr = bboxes[i * 4 + 2] * s;
                let db = bboxes[i * 4 + 3] * s;

                let x1 = (ax - dl) / det_scale;
                let y1 = (ay - dt) / det_scale;
                let x2 = (ax + dr) / det_scale;
                let y2 = (ay + db) / det_scale;

                // KPS decode: anchor + offset*stride
                let mut kps = [[0.0f32; 2]; 5];
                for k in 0..5 {
                    kps[k][0] = (ax + kpss[i * 10 + k * 2] * s) / det_scale;
                    kps[k][1] = (ay + kpss[i * 10 + k * 2 + 1] * s) / det_scale;
                }

                dets.push(DetectedFace {
                    bbox: [x1, y1, x2, y2],
                    kps_5: kps,
                    score,
                });
            }
        }

        dets.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        nms(&mut dets, self.nms_iou_threshold);
        Ok(dets)
    }
}

impl FaceDetectorBackend for AnchorDetector {
    fn detect_gpu(
        &self,
        mgr: &mut ModelManager,
        gpu: &GpuOps,
        frame_chw: &CudaSlice<f32>,
        ws: &mut GpuWorkspace,
        frame_h: u32,
        frame_w: u32,
    ) -> anyhow::Result<(Vec<DetectedFace>, f32)> {
        AnchorDetector::detect_gpu(self, mgr, gpu, frame_chw, ws, frame_h, frame_w)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Shared utilities
// ═══════════════════════════════════════════════════════════════════════

/// Compute letterbox target dimensions and det_scale for a given source resolution.
/// Mirrors letterbox_resize_normalize but returns only the sizes so both CPU and GPU
/// paths can share the math.
pub(crate) fn compute_letterbox_dims(h: u32, w: u32, target: u32) -> (u32, u32, f32) {
    let im_ratio = h as f32 / w as f32;
    let (new_w, new_h) = if im_ratio > 1.0 {
        (((target as f32) / im_ratio) as u32, target)
    } else {
        (target, ((target as f32) * im_ratio) as u32)
    };
    let det_scale = new_h as f32 / h as f32;
    (new_w, new_h, det_scale)
}

/// Letterbox resize + normalize: img [3,H,W] → [3,S,S], padded right+bottom.
/// `mul` and `add` applied as: output = pixel * mul + add
/// YoloFace: mul=1/255, add=0 → [0,1]
/// RetinaFace: mul=1/128, add=-127.5/128 → (pixel-127.5)/128
/// SCRFD: mul=1, add=0 → raw pixels
#[allow(dead_code)]
fn letterbox_resize_normalize(
    img: &[f32],
    h: u32,
    w: u32,
    target: u32,
    mul: f32,
    add: f32,
) -> (Vec<f32>, f32) {
    let im_ratio = h as f32 / w as f32;
    let (new_w, new_h) = if im_ratio > 1.0 {
        (((target as f32) / im_ratio) as u32, target)
    } else {
        (target, ((target as f32) * im_ratio) as u32)
    };
    let det_scale = new_h as f32 / h as f32;

    let mut output = vec![add; (3 * target * target) as usize]; // fill with `add` (0 for padding)

    for c in 0..3u32 {
        for y in 0..new_h {
            let sy = y as f32 * (h as f32 / new_h as f32);
            let y0 = (sy as u32).min(h - 1);
            let y1 = (y0 + 1).min(h - 1);
            let fy = sy - y0 as f32;

            for x in 0..new_w {
                let sx = x as f32 * (w as f32 / new_w as f32);
                let x0 = (sx as u32).min(w - 1);
                let x1 = (x0 + 1).min(w - 1);
                let fx = sx - x0 as f32;

                let idx = |yy: u32, xx: u32| (c * h * w + yy * w + xx) as usize;
                let v = img[idx(y0, x0)] * (1.0 - fx) * (1.0 - fy)
                    + img[idx(y0, x1)] * fx * (1.0 - fy)
                    + img[idx(y1, x0)] * (1.0 - fx) * fy
                    + img[idx(y1, x1)] * fx * fy;

                output[(c * target * target + y * target + x) as usize] = v * mul + add;
            }
        }
    }

    (output, det_scale)
}

/// Greedy NMS in-place. Removes low-IoU detections.
fn nms(dets: &mut Vec<DetectedFace>, iou_thresh: f32) {
    let mut keep = vec![true; dets.len()];
    for i in 0..dets.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..dets.len() {
            if !keep[j] {
                continue;
            }
            if iou(&dets[i].bbox, &dets[j].bbox) > iou_thresh {
                keep[j] = false;
            }
        }
    }
    let mut i = 0;
    dets.retain(|_| {
        let k = keep[i];
        i += 1;
        k
    });
}

fn decode_compact_candidates(
    data: &[f32],
    count: usize,
    nms_iou_threshold: f32,
) -> Vec<DetectedFace> {
    let mut dets = Vec::with_capacity(count);
    for candidate in data[..count * 15].chunks_exact(15) {
        let kps_5 =
            std::array::from_fn(|index| [candidate[4 + index * 2], candidate[5 + index * 2]]);
        dets.push(DetectedFace {
            bbox: [candidate[0], candidate[1], candidate[2], candidate[3]],
            kps_5,
            score: candidate[14],
        });
    }
    dets.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    nms(&mut dets, nms_iou_threshold);
    dets
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    let inter = (ix2 - ix1 + 1.0).max(0.0) * (iy2 - iy1 + 1.0).max(0.0);
    let area_a = (a[2] - a[0] + 1.0) * (a[3] - a[1] + 1.0);
    let area_b = (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0);
    inter / (area_a + area_b - inter)
}

fn select_center_faces(
    faces: &mut Vec<DetectedFace>,
    image_height: u32,
    image_width: u32,
    max_faces: usize,
) {
    if max_faces == 0 || faces.len() <= 1 {
        return;
    }
    let center_x = (image_width / 2) as f32;
    let center_y = (image_height / 2) as f32;
    let priority = |face: &DetectedFace| {
        let area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]);
        let offset_x = (face.bbox[0] + face.bbox[2]) * 0.5 - center_x;
        let offset_y = (face.bbox[1] + face.bbox[3]) * 0.5 - center_y;
        area - (offset_x * offset_x + offset_y * offset_y) * 2.0
    };
    faces.sort_by(|left, right| priority(right).total_cmp(&priority(left)));
    faces.truncate(max_faces);
}

#[cfg(test)]
mod tests {
    use super::{
        AnchorDetector, AnchorDetectorKind, DetectedFace, YoloFaceDetector, select_center_faces,
        unrotate_detection, unrotate_point,
    };

    #[test]
    fn compact_yolo_candidates_preserve_geometry_score_sort_and_nms() {
        let detector = YoloFaceDetector::new(0.5);
        let mut candidates = vec![0.0f32; 3 * 15];
        candidates[0..4].copy_from_slice(&[0.0, 0.0, 10.0, 10.0]);
        candidates[4..14].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        candidates[14] = 0.8;
        candidates[15..19].copy_from_slice(&[1.0, 1.0, 11.0, 11.0]);
        candidates[29] = 0.7;
        candidates[30..34].copy_from_slice(&[100.0, 100.0, 120.0, 120.0]);
        candidates[44] = 0.9;

        let faces = detector.decode_compact_yolo(&candidates, 3);
        assert_eq!(faces.len(), 2);
        assert_eq!(faces[0].score, 0.9);
        assert_eq!(faces[1].bbox, [0.0, 0.0, 10.0, 10.0]);
        assert_eq!(faces[1].kps_5[0], [1.0, 2.0]);
    }

    #[test]
    fn rotated_coordinates_map_back_to_the_original_frame() {
        assert_eq!(unrotate_point([1.0, 1.0], 3, 4, 1), [2.0, 1.0]);
        assert_eq!(unrotate_point([1.0, 1.0], 3, 4, 2), [2.0, 1.0]);
        assert_eq!(unrotate_point([1.0, 2.0], 3, 4, 3), [2.0, 1.0]);

        let rotated = DetectedFace {
            bbox: [0.0, 1.0, 1.0, 3.0],
            kps_5: [[1.0, 1.0]; 5],
            score: 0.9,
        };
        let original = unrotate_detection(rotated, 3, 4, 1);
        assert_eq!(original.bbox, [0.0, 0.0, 2.0, 1.0]);
        assert_eq!(original.kps_5, [[2.0, 1.0]; 5]);
    }

    #[test]
    fn anchor_decoder_matches_crossswap_stride_geometry_and_inclusive_threshold() {
        let lengths = AnchorDetectorKind::output_lengths();
        let mut owned: Vec<Vec<f32>> = lengths.into_iter().map(|len| vec![0.0; len]).collect();
        owned[0][0] = 0.5;
        owned[3][..4].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        owned[6][..10].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let outputs: Vec<&[f32]> = owned.iter().map(Vec::as_slice).collect();

        let detector = AnchorDetector::new(AnchorDetectorKind::RetinaFace, 0.5);
        let faces = detector.decode_anchor(&outputs, 512, 2.0).unwrap();
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].bbox, [-4.0, -8.0, 12.0, 16.0]);
        assert_eq!(
            faces[0].kps_5,
            [
                [4.0, 8.0],
                [12.0, 16.0],
                [20.0, 24.0],
                [28.0, 32.0],
                [36.0, 40.0]
            ]
        );
        assert_eq!(faces[0].score, 0.5);
    }

    #[test]
    fn max_faces_uses_crossswap_area_minus_center_distance_priority() {
        let mut faces = vec![
            DetectedFace {
                bbox: [0.0, 0.0, 40.0, 40.0],
                kps_5: [[0.0; 2]; 5],
                score: 0.99,
            },
            DetectedFace {
                bbox: [40.0, 40.0, 60.0, 60.0],
                kps_5: [[0.0; 2]; 5],
                score: 0.5,
            },
        ];
        select_center_faces(&mut faces, 100, 100, 1);
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].bbox, [40.0, 40.0, 60.0, 60.0]);
    }
}
