//! ArcFace face recognition + Inswapper latent computation.
//!
//! Port of crosswap/app/processors/face_swappers.py (recognize + calc_inswapper_latent)

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DriverError};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};

use crate::config::parameters::SimilarityType;
use crate::gpu::ops::GpuOps;
use crate::math::affine;
use crate::math::constants::{ARCFACE_DST, ARCFACE_MAP_TEMPLATES};
use crate::models::manager::ModelManager;
use crate::pipeline::ort_binding::{bind_input_raw, bind_output_raw, create_cuda_tensor_f32};
use crate::pipeline::workspace::GpuWorkspace;

/// Face recognition and latent computation.
pub struct FaceRecognizer;

impl FaceRecognizer {
    /// Extract 512-dim ArcFace embedding from a face (CPU fallback path).
    pub fn recognize(
        manager: &mut ModelManager,
        frame_chw: &[f32],
        frame_h: u32,
        frame_w: u32,
        kps_5: &[[f32; 2]; 5],
    ) -> anyhow::Result<Vec<f32>> {
        let affine_mat = affine::estimate_face_affine(kps_5, &ARCFACE_DST);
        let mut face_112 = vec![0.0f32; 3 * 112 * 112];
        warp_affine_chw(
            frame_chw,
            frame_h,
            frame_w,
            &mut face_112,
            112,
            112,
            &affine_mat,
        );
        for v in face_112.iter_mut() {
            *v = (*v - 127.5) / 127.5;
        }

        let session = manager
            .get_mut("Inswapper128ArcFace")
            .ok_or_else(|| anyhow::anyhow!("Inswapper128ArcFace not loaded"))?;
        let input_tensor = ort::value::Tensor::from_array(([1usize, 3, 112, 112], face_112))?;
        let outputs = session.run(ort::inputs![input_tensor])?;
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(data[..512].to_vec())
    }

    /// GPU-native recognize via IoBinding: zero GPU→CPU roundtrip for single face.
    ///
    /// Synchronizes after `run_binding` to prevent races on `ws.arcface_embedding`.
    pub fn recognize_gpu(
        manager: &mut ModelManager,
        gpu: &GpuOps,
        frame_chw_gpu: &CudaSlice<f32>,
        frame_h: u32,
        frame_w: u32,
        kps_5: &[[f32; 2]; 5],
        ws: &mut GpuWorkspace,
    ) -> anyhow::Result<[f32; 512]> {
        Self::recognize_gpu_with_similarity(
            manager,
            gpu,
            frame_chw_gpu,
            frame_h,
            frame_w,
            kps_5,
            ws,
            SimilarityType::Opal,
        )
    }

    pub fn recognize_gpu_with_similarity(
        manager: &mut ModelManager,
        gpu: &GpuOps,
        frame_chw_gpu: &CudaSlice<f32>,
        frame_h: u32,
        frame_w: u32,
        kps_5: &[[f32; 2]; 5],
        ws: &mut GpuWorkspace,
        similarity_type: SimilarityType,
    ) -> anyhow::Result<[f32; 512]> {
        match similarity_type {
            SimilarityType::Pearl => {
                let mut template = ARCFACE_DST;
                for point in &mut template {
                    point[0] += 8.0;
                }
                let affine_mat = affine::estimate_face_affine(kps_5, &template);
                gpu.warp_affine_npp(
                    frame_chw_gpu,
                    &mut ws.face_128,
                    frame_h,
                    frame_w,
                    128,
                    128,
                    &affine_mat,
                )?;
                gpu.resize_npp(&ws.face_128, &mut ws.face_112, 128, 128, 112, 112)?;
            }
            SimilarityType::Opal | SimilarityType::Optimal => {
                let template = recognition_template(kps_5, similarity_type);
                let affine_mat = affine::estimate_face_affine(kps_5, &template);
                gpu.warp_affine_npp(
                    frame_chw_gpu,
                    &mut ws.face_112,
                    frame_h,
                    frame_w,
                    112,
                    112,
                    &affine_mat,
                )?;
            }
        }
        gpu.affine_scale(&mut ws.face_112, 1.0 / 127.5, -1.0)?;

        let input_shape = [1i64, 3, 112, 112];
        let output_shape = [1i64, 512];
        let cuda_mem_info = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| anyhow::anyhow!("MemoryInfo: {e}"))?;

        {
            let (input_dev, _g1) = ws.face_112.device_ptr(&gpu.stream);
            let (output_dev, _g2) = ws.arcface_embedding.device_ptr_mut(&gpu.stream);
            let input_value =
                unsafe { create_cuda_tensor_f32(&cuda_mem_info, input_dev, &input_shape)? };
            let output_value =
                unsafe { create_cuda_tensor_f32(&cuda_mem_info, output_dev, &output_shape)? };
            let (session, binding) = manager.session_and_binding("Inswapper128ArcFace")?;
            let input_name = session
                .inputs()
                .first()
                .map(|i| i.name().to_string())
                .unwrap_or_else(|| "input".to_string());
            let output_name = session
                .outputs()
                .first()
                .map(|o| o.name().to_string())
                .unwrap_or_else(|| "683".to_string());
            unsafe {
                bind_input_raw(binding, &input_name, &input_value)?;
                bind_output_raw(binding, &output_name, &output_value)?;
            }
            binding.synchronize_inputs()?;
            let _ = session.run_binding(binding)?;
            binding.synchronize_outputs()?;
            binding.clear();
            drop(input_value);
            drop(output_value);
        }

        gpu.download_into(&ws.arcface_embedding, &mut ws.host_embedding)?;
        Ok(ws.host_embedding)
    }

    /// Compute Inswapper latent from ArcFace embedding (CPU).
    ///
    /// latent = L2norm(L2norm(embedding) @ emap)
    pub fn calc_latent(embedding: &[f32], emap: &[f32]) -> Vec<f32> {
        assert_eq!(embedding.len(), 512);
        assert_eq!(emap.len(), 512 * 512);

        let norm = embedding
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(1e-10);
        let n_e: Vec<f32> = embedding.iter().map(|x| x / norm).collect();

        let mut latent = vec![0.0f32; 512];
        for j in 0..512 {
            let mut sum = 0.0f32;
            for i in 0..512 {
                sum += n_e[i] * emap[i * 512 + j];
            }
            latent[j] = sum;
        }

        let norm2 = latent.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        for v in latent.iter_mut() {
            *v /= norm2;
        }
        latent
    }

    /// GPU-native latent computation: L2norm(L2norm(embedding) @ emap) on CUDA.
    ///
    /// `embedding_gpu` [512], `emap_gpu` [512*512], `output_gpu` [512] — all on GPU.
    /// Uses matmul_512 + l2_normalize kernels. Result stays on GPU.
    pub fn calc_latent_gpu(
        gpu: &GpuOps,
        embedding_gpu: &mut CudaSlice<f32>,
        emap_gpu: &CudaSlice<f32>,
        output_gpu: &mut CudaSlice<f32>,
    ) -> Result<(), DriverError> {
        gpu.calc_latent_512(embedding_gpu, emap_gpu, output_gpu)?;
        Ok(())
    }

    /// Cosine similarity between two 512-dim embeddings.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), 512);
        assert_eq!(b.len(), 512);
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let denom = norm_a * norm_b;
        if denom > 1e-8 { dot / denom } else { 0.0 }
    }
}

fn recognition_template(kps_5: &[[f32; 2]; 5], similarity_type: SimilarityType) -> [[f32; 2]; 5] {
    if similarity_type != SimilarityType::Optimal {
        return ARCFACE_DST;
    }
    ARCFACE_MAP_TEMPLATES
        .iter()
        .copied()
        .min_by(|left, right| template_error(kps_5, left).total_cmp(&template_error(kps_5, right)))
        .unwrap_or(ARCFACE_DST)
}

fn template_error(kps_5: &[[f32; 2]; 5], template: &[[f32; 2]; 5]) -> f64 {
    let matrix = affine::estimate_face_affine(kps_5, template);
    kps_5
        .iter()
        .zip(template)
        .map(|(point, target)| {
            let x = matrix[0][0] * f64::from(point[0])
                + matrix[0][1] * f64::from(point[1])
                + matrix[0][2];
            let y = matrix[1][0] * f64::from(point[0])
                + matrix[1][1] * f64::from(point[1])
                + matrix[1][2];
            (x - f64::from(target[0])).hypot(y - f64::from(target[1]))
        })
        .sum()
}

#[cfg(test)]
mod template_tests {
    use super::recognition_template;
    use crate::config::parameters::SimilarityType;
    use crate::math::constants::{ARCFACE_DST, ARCFACE_MAP_TEMPLATES};

    #[test]
    fn optimal_similarity_selects_the_matching_pose_template() {
        assert_eq!(
            recognition_template(&ARCFACE_MAP_TEMPLATES[4], SimilarityType::Optimal),
            ARCFACE_MAP_TEMPLATES[4]
        );
        assert_eq!(
            recognition_template(&ARCFACE_DST, SimilarityType::Opal),
            ARCFACE_DST
        );
    }
}

/// CPU bilinear affine warp for CHW float32 image.
pub fn warp_affine_chw(
    src: &[f32],
    src_h: u32,
    src_w: u32,
    dst: &mut [f32],
    dst_h: u32,
    dst_w: u32,
    affine: &[[f64; 3]; 2],
) {
    let inv = affine::invert_2x3(affine);
    for c in 0..3u32 {
        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let sx = (inv[0][0] * dx as f64 + inv[0][1] * dy as f64 + inv[0][2]) as f32;
                let sy = (inv[1][0] * dx as f64 + inv[1][1] * dy as f64 + inv[1][2]) as f32;
                if sx < 0.0 || sy < 0.0 || sx >= (src_w - 1) as f32 || sy >= (src_h - 1) as f32 {
                    dst[(c * dst_h * dst_w + dy * dst_w + dx) as usize] = 0.0;
                    continue;
                }
                let x0 = sx as u32;
                let y0 = sy as u32;
                let x1 = (x0 + 1).min(src_w - 1);
                let y1 = (y0 + 1).min(src_h - 1);
                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;
                let idx = |yy: u32, xx: u32| (c * src_h * src_w + yy * src_w + xx) as usize;
                let v = src[idx(y0, x0)] * (1.0 - fx) * (1.0 - fy)
                    + src[idx(y0, x1)] * fx * (1.0 - fy)
                    + src[idx(y1, x0)] * (1.0 - fx) * fy
                    + src[idx(y1, x1)] * fx * fy;
                dst[(c * dst_h * dst_w + dy * dst_w + dx) as usize] = v;
            }
        }
    }
}
