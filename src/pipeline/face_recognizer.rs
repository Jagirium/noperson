//! ArcFace face recognition + Inswapper latent computation.
//!
//! Port of crosswap/app/processors/face_swappers.py (recognize + calc_inswapper_latent)

use crate::backend::{Buffer, ComputeError, ComputeOps, DevicePtr, DevicePtrMut};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};

use crate::config::parameters::SimilarityType;
use crate::math::affine;
use crate::math::constants::{ARCFACE_DST, ARCFACE_MAP_TEMPLATES};
use crate::models::manager::ModelManager;
use crate::pipeline::ort_binding::{create_cuda_tensor_f32, run_bound_values};
use crate::pipeline::workspace::GpuWorkspace;

/// Face recognition and latent computation.
pub struct FaceRecognizer;

impl FaceRecognizer {
    /// GPU-native recognize via IoBinding: zero GPU→CPU roundtrip for single face.
    ///
    /// Provider-aware stream ordering prevents races on `ws.arcface_embedding`.
    pub fn recognize_gpu(
        manager: &mut ModelManager,
        gpu: &ComputeOps,
        frame_chw_gpu: &Buffer<f32>,
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

    /// Run ArcFace into the workspace's device-resident embedding buffer.
    ///
    /// Unlike the compatibility APIs, this performs no device-to-host copy.
    pub fn recognize_gpu_into(
        manager: &mut ModelManager,
        gpu: &ComputeOps,
        frame_chw_gpu: &Buffer<f32>,
        frame_h: u32,
        frame_w: u32,
        kps_5: &[[f32; 2]; 5],
        ws: &mut GpuWorkspace,
    ) -> anyhow::Result<()> {
        Self::recognize_gpu_into_with_similarity(
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
        gpu: &ComputeOps,
        frame_chw_gpu: &Buffer<f32>,
        frame_h: u32,
        frame_w: u32,
        kps_5: &[[f32; 2]; 5],
        ws: &mut GpuWorkspace,
        similarity_type: SimilarityType,
    ) -> anyhow::Result<[f32; 512]> {
        Self::recognize_gpu_into_with_similarity(
            manager,
            gpu,
            frame_chw_gpu,
            frame_h,
            frame_w,
            kps_5,
            ws,
            similarity_type,
        )?;
        gpu.download_into(&ws.arcface_embedding, &mut ws.host_embedding)?;
        Ok(ws.host_embedding)
    }

    /// Similarity-mode-aware ArcFace inference that leaves output on-device.
    pub fn recognize_gpu_into_with_similarity(
        manager: &mut ModelManager,
        gpu: &ComputeOps,
        frame_chw_gpu: &Buffer<f32>,
        frame_h: u32,
        frame_w: u32,
        kps_5: &[[f32; 2]; 5],
        ws: &mut GpuWorkspace,
        similarity_type: SimilarityType,
    ) -> anyhow::Result<()> {
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
            manager.device_id(),
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| anyhow::anyhow!("MemoryInfo: {e}"))?;

        let (input_name, output_name) = {
            let session = manager
                .get("Inswapper128ArcFace")
                .ok_or_else(|| anyhow::anyhow!("Inswapper128ArcFace not loaded"))?;
            let input_name = session
                .inputs()
                .first()
                .map(|input| input.name().to_string())
                .unwrap_or_else(|| "input".to_string());
            let output_name = session
                .outputs()
                .first()
                .map(|output| output.name().to_string())
                .unwrap_or_else(|| "683".to_string());
            (input_name, output_name)
        };

        {
            let (input_dev, _g1) = ws.face_112.device_ptr(&gpu.stream);
            let (output_dev, _g2) = ws.arcface_embedding.device_ptr_mut(&gpu.stream);
            let input_value =
                unsafe { create_cuda_tensor_f32(&cuda_mem_info, input_dev, &input_shape)? };
            let output_value =
                unsafe { create_cuda_tensor_f32(&cuda_mem_info, output_dev, &output_shape)? };
            run_bound_values(
                manager,
                &gpu.stream,
                "Inswapper128ArcFace",
                &[(input_name.as_str(), &input_value)],
                &[(output_name.as_str(), &output_value)],
            )?;
        }

        Ok(())
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
        gpu: &ComputeOps,
        embedding_gpu: &mut Buffer<f32>,
        emap_gpu: &Buffer<f32>,
        output_gpu: &mut Buffer<f32>,
    ) -> Result<(), ComputeError> {
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
