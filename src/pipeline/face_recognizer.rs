//! ArcFace face recognition + Inswapper latent computation.
//!
//! Port of crosswap/app/processors/face_swappers.py (recognize + calc_inswapper_latent)

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DriverError};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};

use crate::gpu::ops::GpuOps;
use crate::math::affine;
use crate::math::constants::ARCFACE_DST;
use crate::models::manager::ModelManager;
use crate::pipeline::ort_binding::{bind_input_raw, bind_output_raw, create_cuda_tensor_f32};
use crate::pipeline::workspace::{GpuWorkspace, MAX_FACES};

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
    ) -> anyhow::Result<Vec<f32>> {
        let affine_mat = affine::estimate_face_affine(kps_5, &ARCFACE_DST);
        gpu.warp_affine_npp(
            frame_chw_gpu,
            &mut ws.face_112,
            frame_h,
            frame_w,
            112,
            112,
            &affine_mat,
        )?;
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
        Ok(ws.host_embedding.clone())
    }

    /// Batched GPU recognize: extract embeddings for multiple faces in 1 ort call.
    ///
    /// Warps each face to 112×112, stacks into [N, 3, 112, 112], runs 1 ort inference,
    /// downloads [N, 512] embeddings.
    ///
    /// `faces_kps` = slice of 5-point landmark sets, one per face.
    /// Returns Vec of embeddings (len = faces_kps.len(), each 512-dim).
    pub fn recognize_batch_gpu(
        manager: &mut ModelManager,
        gpu: &GpuOps,
        frame_chw_gpu: &CudaSlice<f32>,
        frame_h: u32,
        frame_w: u32,
        faces_kps: &[[[f32; 2]; 5]],
        ws: &mut GpuWorkspace,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let n = faces_kps.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        anyhow::ensure!(
            n <= MAX_FACES,
            "batch size {n} exceeds MAX_FACES {MAX_FACES}"
        );

        let face_size = 3 * 112 * 112;
        let total_input = n * face_size;

        // 1. Download frame to CPU once — reuse for all face warps.
        //    TODO: replace with GPU warp_affine_npp into batch buffer slices.
        let frame_cpu = gpu.download(frame_chw_gpu)?;
        let frame_slice = frame_cpu.as_slice();

        // 2. Warp each face to 112×112 and normalize, store into host_face_112_batch
        for (i, kps) in faces_kps.iter().enumerate() {
            let affine_mat = affine::estimate_face_affine(kps, &ARCFACE_DST);
            let offset = i * face_size;
            let slot = &mut ws.host_face_112_batch[offset..offset + face_size];
            warp_affine_chw(frame_slice, frame_h, frame_w, slot, 112, 112, &affine_mat);
            for v in slot.iter_mut() {
                *v = (*v - 127.5) / 127.5;
            }
        }

        // 3. Upload batched faces to GPU
        gpu.upload_into(
            &ws.host_face_112_batch[..total_input],
            &mut ws.face_112_batch,
        )?;

        // 4. IoBinding inference — input [N, 3, 112, 112], output [N, 512]
        let input_shape = [n as i64, 3, 112, 112];
        let output_shape = [n as i64, 512];
        let cuda_mem_info = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| anyhow::anyhow!("MemoryInfo: {e}"))?;

        {
            let (input_dev, _g1) = ws.face_112_batch.device_ptr(&gpu.stream);
            let (output_dev, _g2) = ws.arcface_embeddings_batch.device_ptr_mut(&gpu.stream);
            let input_value =
                unsafe { create_cuda_tensor_f32(&cuda_mem_info, input_dev, &input_shape)? };
            let output_value =
                unsafe { create_cuda_tensor_f32(&cuda_mem_info, output_dev, &output_shape)? };
            let (session, binding) = manager.session_and_binding("Inswapper128ArcFaceBatch")?;
            let input_name = session
                .inputs()
                .first()
                .map(|i| i.name().to_string())
                .unwrap_or_else(|| "input.1".to_string());
            let output_name = session
                .outputs()
                .first()
                .map(|o| o.name().to_string())
                .unwrap_or_else(|| "683".to_string());
            unsafe {
                bind_input_raw(binding, &input_name, &input_value)?;
                bind_output_raw(binding, &output_name, &output_value)?;
            }
            let _ = session.run_binding(binding)?;
            binding.clear();
            drop(input_value);
            drop(output_value);
        }

        // 5. Download [N, 512] embeddings
        gpu.download_into(&ws.arcface_embeddings_batch, &mut ws.host_embeddings_batch)?;

        // 6. Split into Vec<Vec<f32>>
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let offset = i * 512;
            result.push(ws.host_embeddings_batch[offset..offset + 512].to_vec());
        }
        Ok(result)
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
        // 1. L2 normalize embedding in-place (we don't need the raw embedding after)
        gpu.l2_normalize(embedding_gpu, 512)?;
        // 2. Matmul: output[i] = sum_j(n_e[j] * emap[i][j])
        gpu.matmul_512(embedding_gpu, emap_gpu, output_gpu, 512)?;
        // 3. L2 normalize the result in-place
        gpu.l2_normalize(output_gpu, 512)?;
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
