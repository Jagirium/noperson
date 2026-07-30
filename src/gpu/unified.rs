//! Unified CUDA stream for cudarc + ort.
//!
//! cudarc kernels and ort inference share one CUDA stream.
//! Inputs are zero-copy (ort reads directly from cudarc buffers).
//! Outputs use a same-device dtod copy from the ort allocation to our buffer.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::{Shape, TensorRefMut};

/// Create an ort session sharing the cudarc CUDA stream.
pub fn create_session_on_stream(
    stream: &Arc<CudaStream>,
    model_path: &str,
) -> anyhow::Result<Session> {
    let raw_stream = stream.cu_stream();
    let session = Session::builder()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_execution_providers([unsafe {
            ort::ep::CUDA::default()
                .with_compute_stream(raw_stream as *mut ())
                .build()
        }])
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_intra_threads(1)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_inter_threads(1)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .commit_from_file(model_path)?;
    Ok(session)
}

/// Wrap a cudarc CudaSlice as an ort TensorRefMut (zero-copy input).
///
/// # Safety
///
/// The slice must remain allocated on the CUDA device and must not be mutated
/// through another alias until the returned tensor and its ORT run are dropped.
pub unsafe fn input_tensor<'a>(
    slice: &'a mut CudaSlice<f32>,
    stream: &'a CudaStream,
    shape: impl Into<Shape>,
) -> anyhow::Result<TensorRefMut<'a, f32>> {
    let mem_info = MemoryInfo::new(
        AllocationDevice::CUDA,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )?;
    let (cu_ptr, _guard) = slice.device_ptr_mut(stream);
    let raw_ptr = cu_ptr as usize as *mut core::ffi::c_void;
    Ok(unsafe { TensorRefMut::from_raw(mem_info, raw_ptr, shape.into())? })
}

fn cuda_mem_info() -> anyhow::Result<MemoryInfo<'static>> {
    Ok(MemoryInfo::new(
        AllocationDevice::CUDA,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )?)
}

/// Copy ort output (GPU) to our cudarc buffer (GPU) on the same stream.
unsafe fn copy_ort_output_to_slice(
    outputs: &ort::session::SessionOutputs<'_>,
    output_name: &str,
    stream: &Arc<CudaStream>,
    d_output: &mut CudaSlice<f32>,
) -> anyhow::Result<()> {
    let ort_out = outputs[output_name]
        .downcast_ref::<ort::value::DynTensorValueType>()
        .map_err(|e| anyhow::anyhow!("output downcast failed: {e}"))?;
    let ort_ptr = ort_out.data_ptr() as u64;

    unsafe {
        let ort_slice: CudaSlice<f32> = stream.upgrade_device_ptr(ort_ptr, d_output.len());
        stream.memcpy_dtod(&ort_slice, d_output)?;
        std::mem::forget(ort_slice);
    }
    Ok(())
}

/// Run Inswapper128: d_tiles + d_latent → d_output.
///
/// Inputs: zero-copy from cudarc buffers via IoBinding.
/// Output: ort allocates on the same CUDA device and copies dtod to d_output.
pub fn run_inswapper(
    session: &mut Session,
    stream: &Arc<CudaStream>,
    d_tiles: &mut CudaSlice<f32>,  // [N, 3, 128, 128]
    d_latent: &mut CudaSlice<f32>, // [N, 512]
    d_output: &mut CudaSlice<f32>, // [N, 3, 128, 128]
    batch_size: usize,
) -> anyhow::Result<()> {
    let target = unsafe { input_tensor(d_tiles, stream, [batch_size, 3, 128, 128])? };
    let source = unsafe { input_tensor(d_latent, stream, [batch_size, 512])? };

    let mut binding = session.create_binding()?;
    binding.bind_input("target", &target)?;
    binding.bind_input("source", &source)?;
    binding.bind_output_to_device("output", &cuda_mem_info()?)?;

    let outputs = session.run_binding(&binding)?;
    unsafe { copy_ort_output_to_slice(&outputs, "output", stream, d_output)? };
    Ok(())
}

/// Run ArcFace: d_face_112 → d_embedding.
pub fn run_arcface(
    session: &mut Session,
    stream: &Arc<CudaStream>,
    d_face: &mut CudaSlice<f32>,      // [1, 3, 112, 112]
    d_embedding: &mut CudaSlice<f32>, // [1, 512]
) -> anyhow::Result<()> {
    let input = unsafe { input_tensor(d_face, stream, [1usize, 3, 112, 112])? };

    let mut binding = session.create_binding()?;
    // ArcFace input name is model-dependent; use first input
    let input_name = session
        .inputs()
        .first()
        .map(|i| i.name().to_string())
        .unwrap_or_else(|| "input".to_string());
    binding.bind_input(&input_name, &input)?;

    let output_name = session
        .outputs()
        .first()
        .map(|o| o.name().to_string())
        .unwrap_or_else(|| "output".to_string());
    binding.bind_output_to_device(&output_name, &cuda_mem_info()?)?;

    let outputs = session.run_binding(&binding)?;
    unsafe { copy_ort_output_to_slice(&outputs, &output_name, stream, d_embedding)? };
    Ok(())
}

/// Run GPEN restorer: d_face → d_output (both [1, 3, S, S]).
pub fn run_gpen(
    session: &mut Session,
    stream: &Arc<CudaStream>,
    d_face: &mut CudaSlice<f32>,   // [1, 3, S, S]
    d_output: &mut CudaSlice<f32>, // [1, 3, S, S]
    size: usize,
) -> anyhow::Result<()> {
    let input = unsafe { input_tensor(d_face, stream, [1usize, 3, size, size])? };

    let mut binding = session.create_binding()?;
    binding.bind_input("input", &input)?;
    binding.bind_output_to_device("output", &cuda_mem_info()?)?;

    let outputs = session.run_binding(&binding)?;
    unsafe { copy_ort_output_to_slice(&outputs, "output", stream, d_output)? };
    Ok(())
}

/// Run Occluder: d_face_256 → d_mask_256 ([1, 1, 256, 256]).
pub fn run_occluder(
    session: &mut Session,
    stream: &Arc<CudaStream>,
    d_face: &mut CudaSlice<f32>, // [1, 3, 256, 256]
    d_mask: &mut CudaSlice<f32>, // [1, 1, 256, 256]
) -> anyhow::Result<()> {
    let input = unsafe { input_tensor(d_face, stream, [1usize, 3, 256, 256])? };

    let mut binding = session.create_binding()?;
    binding.bind_input("img", &input)?;
    binding.bind_output_to_device("output", &cuda_mem_info()?)?;

    let outputs = session.run_binding(&binding)?;
    unsafe { copy_ort_output_to_slice(&outputs, "output", stream, d_mask)? };
    Ok(())
}

/// Run FaceParser: d_face_512 → d_parser_out ([1, 19, 512, 512]).
pub fn run_faceparser(
    session: &mut Session,
    stream: &Arc<CudaStream>,
    d_face: &mut CudaSlice<f32>,   // [1, 3, 512, 512]
    d_output: &mut CudaSlice<f32>, // [1, 19, 512, 512]
) -> anyhow::Result<()> {
    let input = unsafe { input_tensor(d_face, stream, [1usize, 3, 512, 512])? };

    let mut binding = session.create_binding()?;

    let input_name = session
        .inputs()
        .first()
        .map(|i| i.name().to_string())
        .unwrap_or_else(|| "input".to_string());
    binding.bind_input(&input_name, &input)?;

    let output_name = session
        .outputs()
        .first()
        .map(|o| o.name().to_string())
        .unwrap_or_else(|| "output".to_string());
    binding.bind_output_to_device(&output_name, &cuda_mem_info()?)?;

    let outputs = session.run_binding(&binding)?;
    unsafe { copy_ort_output_to_slice(&outputs, &output_name, stream, d_output)? };
    Ok(())
}

/// Run YoloFace8n detection: d_input_640 → d_output.
pub fn run_yoloface(
    session: &mut Session,
    stream: &Arc<CudaStream>,
    d_input: &mut CudaSlice<f32>,  // [1, 3, 640, 640]
    d_output: &mut CudaSlice<f32>, // [1, 20, 8400]
) -> anyhow::Result<()> {
    let input = unsafe { input_tensor(d_input, stream, [1usize, 3, 640, 640])? };

    let mut binding = session.create_binding()?;
    binding.bind_input("images", &input)?;
    binding.bind_output_to_device("output0", &cuda_mem_info()?)?;

    let outputs = session.run_binding(&binding)?;
    unsafe { copy_ort_output_to_slice(&outputs, "output0", stream, d_output)? };
    Ok(())
}

/// Run a frame enhancer: d_tiles_in → d_tiles_out (batched).
/// Input/output names are both "input"/"output".
/// Input: raw [0, 255] pixels, no normalization.
pub fn run_enhancer(
    session: &mut Session,
    stream: &Arc<CudaStream>,
    d_input: &mut CudaSlice<f32>,
    input_shape: &[usize],
    d_output: &mut CudaSlice<f32>,
) -> anyhow::Result<()> {
    let input = unsafe { input_tensor(d_input, stream, input_shape.to_vec())? };

    let mut binding = session.create_binding()?;
    binding.bind_input("input", &input)?;
    binding.bind_output_to_device("output", &cuda_mem_info()?)?;

    let outputs = session.run_binding(&binding)?;
    unsafe { copy_ort_output_to_slice(&outputs, "output", stream, d_output)? };
    Ok(())
}

/// Run XSeg mask: d_face_256 → d_mask_256.
/// Input: "in_face:0" [1, 3, 256, 256] /255. Output: "out_mask:0" [1, 1, 256, 256].
pub fn run_xseg(
    session: &mut Session,
    stream: &Arc<CudaStream>,
    d_face: &mut CudaSlice<f32>, // [1, 3, 256, 256]
    d_mask: &mut CudaSlice<f32>, // [1, 1, 256, 256]
) -> anyhow::Result<()> {
    let input = unsafe { input_tensor(d_face, stream, [1usize, 3, 256, 256])? };

    let mut binding = session.create_binding()?;
    binding.bind_input("in_face:0", &input)?;
    binding.bind_output_to_device("out_mask:0", &cuda_mem_info()?)?;

    let outputs = session.run_binding(&binding)?;
    unsafe { copy_ort_output_to_slice(&outputs, "out_mask:0", stream, d_mask)? };
    Ok(())
}
