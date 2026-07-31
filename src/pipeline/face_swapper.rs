//! Inswapper128 face swapping with interlaced tile inference.
//!
//! Port of crosswap/app/processors/workers/frame_worker.py swap_core()
//!
//! The canonical model has a dynamic batch axis. Larger output sizes are split
//! into interlaced 128×128 tiles and evaluated together in one ORT call.
//!
//! ONNX inputs:  "target" [N, 3, 128, 128], "source" [N, 512]
//! ONNX outputs: "output" [N, 3, 128, 128]

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};

use crate::gpu::ops::GpuOps;
use crate::models::manager::ModelManager;
use crate::pipeline::ort_binding::{bind_input_raw, bind_output_raw, create_cuda_tensor_f32};
use crate::pipeline::workspace::GpuWorkspace;

/// Face swapper using the canonical dynamic-batch Inswapper128 model.
pub struct FaceSwapper;

impl FaceSwapper {
    /// Swap a face using Inswapper128 with interlaced tiling.
    ///
    /// `face_chw` — aligned face [3, face_size, face_size] in [0, 255] float32.
    ///   face_size = dim * 128 (dim=1 → 128, dim=2 → 256).
    /// `latent` — [512] Inswapper latent from FaceRecognizer::calc_latent().
    /// `dim` — tiling dimension (1 or 2).
    ///
    /// Returns swapped face [3, face_size, face_size] in [0, 255] float32.
    pub fn swap(
        manager: &mut ModelManager,
        face_chw: &[f32],
        latent: &[f32],
        dim: u32,
    ) -> anyhow::Result<Vec<f32>> {
        let tile_size = 128usize;
        let face_size = (dim as usize) * tile_size;
        let n_tiles = (dim * dim) as usize;

        assert_eq!(face_chw.len(), 3 * face_size * face_size);
        assert_eq!(latent.len(), 512);

        // 1. Extract interlaced tiles + normalize to [0, 1]
        let mut tiles = vec![0.0f32; n_tiles * 3 * tile_size * tile_size];
        interlace_extract(face_chw, &mut tiles, dim as usize, 3, tile_size, face_size);

        // Normalize: / 255.0
        for v in tiles.iter_mut() {
            *v /= 255.0;
        }

        // 2. Replicate the latent for the dynamic tile batch.
        let mut latent_batch = vec![0.0f32; n_tiles * 512];
        for tile in 0..n_tiles {
            latent_batch[tile * 512..(tile + 1) * 512].copy_from_slice(latent);
        }

        // 3. Run all tiles through the canonical dynamic-batch model.
        let session = manager
            .get_mut("Inswapper128")
            .ok_or_else(|| anyhow::anyhow!("Inswapper128 not loaded"))?;
        let target_tensor =
            ort::value::Tensor::from_array(([n_tiles, 3usize, tile_size, tile_size], tiles))?;
        let source_tensor = ort::value::Tensor::from_array(([n_tiles, 512usize], latent_batch))?;
        let outputs = session.run(ort::inputs![
            "target" => target_tensor,
            "source" => source_tensor
        ])?;
        let (shape, out_data) = outputs["output"].try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            shape.as_ref() == [n_tiles as i64, 3, tile_size as i64, tile_size as i64],
            "Unexpected output shape: {shape:?}"
        );
        let out_tiles: Vec<f32> = out_data
            .iter()
            .map(|value| (value * 255.0).clamp(0.0, 255.0))
            .collect();

        // 4. Scatter tiles back to face.
        let mut result = vec![0.0f32; 3 * face_size * face_size];
        interlace_scatter(
            &out_tiles,
            &mut result,
            dim as usize,
            3,
            tile_size,
            face_size,
        );

        Ok(result)
    }

    /// GPU-native swap with low-level IoBinding (same C API path as Python).
    ///
    /// Pipeline:
    ///   1. interlace_extract  face_256 → swap_batch_in
    ///   2. normalize /255      swap_batch_in
    ///   3. upload latent       host → swap_latent_gpu (replicated per tile)
    ///   4. SYNC our stream so the buffers are ready before ort begins
    ///   5. CreateTensorWithDataAsOrtValue × 3   (target/source/output)
    ///      — all three point to OUR CudaSlice device pointers, zero-copy
    ///   6. BindInput target / BindInput source / BindOutput output
    ///   7. RunWithBinding — ort writes inference output directly into
    ///      ws.swap_batch_out, no internal copies, no allocator overhead
    ///   8. clear IoBinding while OrtValue guards are still alive
    ///   9. denormalize + interlace_scatter → face_256
    pub fn swap_gpu(
        manager: &mut ModelManager,
        gpu: &GpuOps,
        ws: &mut GpuWorkspace,
        latent: &[f32],
        dim: u32,
    ) -> anyhow::Result<()> {
        Self::swap_gpu_cached(manager, gpu, ws, latent, dim, true)
    }

    /// Repeat a strength pass while retaining the invariant latent on-device.
    pub(crate) fn swap_gpu_cached(
        manager: &mut ModelManager,
        gpu: &GpuOps,
        ws: &mut GpuWorkspace,
        latent: &[f32],
        dim: u32,
        upload_latent: bool,
    ) -> anyhow::Result<()> {
        assert_eq!(latent.len(), 512);
        anyhow::ensure!((1..=4).contains(&dim), "unsupported swap dim: {dim}");
        let tile_size = 128u32;
        let n_tiles = (dim * dim) as usize;

        {
            let face_ptr: *const CudaSlice<f32> = &ws.face_256;
            let batch_ptr: *mut CudaSlice<f32> = &mut ws.swap_batch_in;
            unsafe {
                gpu.interlace_extract_normalized(&*face_ptr, &mut *batch_ptr, dim, tile_size)?;
            }
        }
        if upload_latent {
            let latent_values = n_tiles * 512;
            for tile in 0..n_tiles {
                ws.host_swap_tiles[tile * 512..(tile + 1) * 512].copy_from_slice(latent);
            }
            gpu.upload_into(
                &ws.host_swap_tiles[..latent_values],
                &mut ws.swap_latent_gpu,
            )?;
        }

        let cuda_mem_info = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|error| anyhow::anyhow!("MemoryInfo: {error}"))?;
        {
            let target_shape = [n_tiles as i64, 3, tile_size as i64, tile_size as i64];
            let source_shape = [n_tiles as i64, 512];
            let (target_dev, _target_guard) = ws.swap_batch_in.device_ptr(&gpu.stream);
            let (source_dev, _source_guard) = ws.swap_latent_gpu.device_ptr(&gpu.stream);
            let (output_dev, _output_guard) = ws.swap_batch_out.device_ptr_mut(&gpu.stream);
            let target_value =
                unsafe { create_cuda_tensor_f32(&cuda_mem_info, target_dev, &target_shape)? };
            let source_value =
                unsafe { create_cuda_tensor_f32(&cuda_mem_info, source_dev, &source_shape)? };
            let output_value =
                unsafe { create_cuda_tensor_f32(&cuda_mem_info, output_dev, &target_shape)? };
            let (session, binding) = manager.session_and_binding("Inswapper128")?;
            unsafe {
                bind_input_raw(binding, "target", &target_value)?;
                bind_input_raw(binding, "source", &source_value)?;
                bind_output_raw(binding, "output", &output_value)?;
            }
            binding.synchronize_inputs()?;
            let _ = session.run_binding(binding)?;
            binding.synchronize_outputs()?;
            binding.clear();
        }

        {
            let batch_ptr: *const CudaSlice<f32> = &ws.swap_batch_out;
            let face_ptr: *mut CudaSlice<f32> = &mut ws.face_256;
            unsafe {
                gpu.interlace_scatter_denormalized(&*batch_ptr, &mut *face_ptr, dim, tile_size)?;
            }
        }
        Ok(())
    }

    /// GPU swap with explicit model name (for batch-specific models).
    pub fn swap_gpu_named(
        manager: &mut ModelManager,
        gpu: &GpuOps,
        ws: &mut GpuWorkspace,
        latent: &[f32],
        dim: u32,
        model_name: &str,
    ) -> anyhow::Result<()> {
        assert_eq!(latent.len(), 512);
        anyhow::ensure!((1..=4).contains(&dim), "unsupported swap dim: {dim}");
        let tile_size = 128u32;
        let n_tiles = (dim * dim) as usize;

        // 1. Interlace extract: face_256 → swap_batch_in
        {
            let face_ptr: *const CudaSlice<f32> = &ws.face_256;
            let batch_ptr: *mut CudaSlice<f32> = &mut ws.swap_batch_in;
            unsafe {
                gpu.interlace_extract_normalized(&*face_ptr, &mut *batch_ptr, dim, tile_size)?;
            }
        }

        // 2. Normalize tiles / 255.0 in place

        // 3. Replicate latent for each tile + upload to swap_latent_gpu
        let latent_replica_len = n_tiles * 512;
        {
            let slot = &mut ws.host_swap_tiles[..latent_replica_len];
            for t in 0..n_tiles {
                slot[t * 512..(t + 1) * 512].copy_from_slice(latent);
            }
            gpu.upload_into(
                &ws.host_swap_tiles[..latent_replica_len],
                &mut ws.swap_latent_gpu,
            )?;
        }

        // 4. No pre-inference sync needed — ort shares our compute stream
        //    (set via set_compute_stream on ModelManager), so all prior kernels
        //    are already ordered in the same stream before run_binding.

        // 5-7. IoBinding inference scope. All mutable borrows on ws.swap_batch_out
        // are confined here so denormalize/scatter below can re-borrow it.
        {
            let cuda_mem_info = MemoryInfo::new(
                AllocationDevice::CUDA,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )
            .map_err(|e| anyhow::anyhow!("MemoryInfo: {e}"))?;

            let target_shape = [n_tiles as i64, 3, tile_size as i64, tile_size as i64];
            let source_shape = [n_tiles as i64, 512];
            let (target_dev, _g1) = ws.swap_batch_in.device_ptr(&gpu.stream);
            let (source_dev, _g2) = ws.swap_latent_gpu.device_ptr(&gpu.stream);
            let (output_dev, _g3) = ws.swap_batch_out.device_ptr_mut(&gpu.stream);
            let target_value =
                unsafe { create_cuda_tensor_f32(&cuda_mem_info, target_dev, &target_shape)? };
            let source_value =
                unsafe { create_cuda_tensor_f32(&cuda_mem_info, source_dev, &source_shape)? };
            let output_value =
                unsafe { create_cuda_tensor_f32(&cuda_mem_info, output_dev, &target_shape)? };
            let (session, binding) = manager.session_and_binding(model_name)?;
            unsafe {
                bind_input_raw(binding, "target", &target_value)?;
                bind_input_raw(binding, "source", &source_value)?;
                bind_output_raw(binding, "output", &output_value)?;
            }
            // TensorRT may execute on an internal stream even when the CUDA EP
            // shares ours. Explicit IoBinding fences are required; otherwise
            // the first frame can read the crop late and scatter a partially
            // written output, producing the characteristic double-face ghost.
            binding.synchronize_inputs()?;
            let _ = session.run_binding(binding)?;
            binding.synchronize_outputs()?;
            binding.clear();
        }

        // 8. Denormalize + scatter back into face_256
        {
            let batch_ptr: *const CudaSlice<f32> = &ws.swap_batch_out;
            let face_ptr: *mut CudaSlice<f32> = &mut ws.face_256;
            unsafe {
                gpu.interlace_scatter_denormalized(&*batch_ptr, &mut *face_ptr, dim, tile_size)?;
            }
        }
        Ok(())
    }
}

/// Extract interlaced tiles from a face.
///
/// face[c, j*dim+tj, i*dim+ti] → tiles[tj*dim+ti, c, j, i]
fn interlace_extract(
    face: &[f32],      // [C, face_size, face_size]
    tiles: &mut [f32], // [n_tiles, C, T, T]
    dim: usize,
    channels: usize,
    tile_size: usize,
    face_size: usize,
) {
    for tj in 0..dim {
        for ti in 0..dim {
            let tile_idx = tj * dim + ti;
            for c in 0..channels {
                for ty in 0..tile_size {
                    for tx in 0..tile_size {
                        let fy = ty * dim + tj;
                        let fx = tx * dim + ti;
                        let face_idx = c * face_size * face_size + fy * face_size + fx;
                        let tile_off = tile_idx * channels * tile_size * tile_size
                            + c * tile_size * tile_size
                            + ty * tile_size
                            + tx;
                        tiles[tile_off] = face[face_idx];
                    }
                }
            }
        }
    }
}

/// Scatter tiles back into a face (inverse of extract).
///
/// tiles[tj*dim+ti, c, j, i] → face[c, j*dim+tj, i*dim+ti]
fn interlace_scatter(
    tiles: &[f32],    // [n_tiles, C, T, T]
    face: &mut [f32], // [C, face_size, face_size]
    dim: usize,
    channels: usize,
    tile_size: usize,
    face_size: usize,
) {
    for tj in 0..dim {
        for ti in 0..dim {
            let tile_idx = tj * dim + ti;
            for c in 0..channels {
                for ty in 0..tile_size {
                    for tx in 0..tile_size {
                        let fy = ty * dim + tj;
                        let fx = tx * dim + ti;
                        let face_idx = c * face_size * face_size + fy * face_size + fx;
                        let tile_off = tile_idx * channels * tile_size * tile_size
                            + c * tile_size * tile_size
                            + ty * tile_size
                            + tx;
                        face[face_idx] = tiles[tile_off];
                    }
                }
            }
        }
    }
}
