//! Face detection: YoloFace8n, RetinaFace, SCRFD.
//!
//! Port of crosswap/app/processors/face_detectors.py
//! All three detectors share the same DetectedFace output format.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};

use crate::gpu::ops::GpuOps;
use crate::models::manager::ModelManager;
use crate::pipeline::ort_binding::{bind_input_raw, bind_output_raw, create_cuda_tensor_f32};
use crate::pipeline::workspace::GpuWorkspace;

/// Detected face with bounding box and 5-point landmarks.
#[derive(Debug, Clone)]
pub struct DetectedFace {
    pub bbox: [f32; 4],       // [x1, y1, x2, y2] in original frame coords
    pub kps_5: [[f32; 2]; 5], // left_eye, right_eye, nose, left_mouth, right_mouth
    pub score: f32,
}

// ═══════════════════════════════════════════════════════════════════════
// YoloFace8n
// ═══════════════════════════════════════════════════════════════════════

pub struct YoloFaceDetector {
    input_size: u32,
    score_threshold: f32,
    nms_iou_threshold: f32,
}

impl YoloFaceDetector {
    pub fn new(score_threshold: f32) -> Self {
        Self {
            input_size: 640,
            score_threshold,
            nms_iou_threshold: 0.4,
        }
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
            0,
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

            let (session, binding) = mgr.session_and_binding("YoloFace8n")?;
            unsafe {
                bind_input_raw(binding, "images", &input_value)?;
                bind_output_raw(binding, "output0", &output_value)?;
            }
            binding.synchronize_inputs()?;
            let _ = session.run_binding(binding)?;
            binding.synchronize_outputs()?;
            binding.clear();

            drop(input_value);
            drop(output_value);
        }

        // 3. Download only the output (168000 f32 = 672 KB) for CPU decode.
        gpu.download_into(&ws.detect_output, &mut ws.host_det_output)?;
        let data = ws.host_det_output.as_slice();
        let shape = [1i64, 20, 8400];
        let faces = self.decode_yolo(data, &shape, det_scale)?;
        Ok((faces, det_scale))
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

// ═══════════════════════════════════════════════════════════════════════
// RetinaFace / SCRFD — anchor-based 3-stride detectors
// ═══════════════════════════════════════════════════════════════════════

pub struct AnchorDetector {
    input_size: u32,
    score_threshold: f32,
    nms_iou_threshold: f32,
    kind: AnchorDetectorKind,
}

#[derive(Clone, Copy)]
pub enum AnchorDetectorKind {
    RetinaFace,
    Scrfd,
}

impl AnchorDetector {
    pub fn new(kind: AnchorDetectorKind, score_threshold: f32) -> Self {
        Self {
            input_size: 512,
            score_threshold,
            nms_iou_threshold: 0.4,
            kind,
        }
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

        self.decode_anchor(&all_outputs, is as u32, scale)
    }

    /// Decode 3-stride anchor-based outputs.
    /// Output layout: [scores_s8, scores_s16, scores_s32, bbox_s8, bbox_s16, bbox_s32, kps_s8, kps_s16, kps_s32]
    fn decode_anchor(
        &self,
        outputs: &[Vec<f32>],
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
                if score <= self.score_threshold {
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
