//! Shared low-level IoBinding helpers for GPU-native ONNX inference.
//!
//! All hot-path inference calls (detect, recognize, swap) use this module
//! to bind CUDA device pointers directly into ort sessions, avoiding the
//! GPU → CPU → GPU roundtrip that `Tensor::from_array` incurs.

use std::ffi::CString;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use ort::AsPointer;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::IoBinding;

use crate::models::manager::ModelManager;

/// RAII wrapper around an OrtValue — releases the underlying tensor on drop.
pub struct OrtValueGuard(pub(crate) *mut ort_sys::OrtValue);

impl Drop for OrtValueGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                (ort::api().ReleaseValue)(self.0);
            }
        }
    }
}

/// Create an OrtValue tensor view over a raw CUDA device pointer (zero-copy).
/// Equivalent of Python `bind_input(buffer_ptr=tensor.data_ptr())`.
///
/// # Safety
/// Caller must ensure:
/// - `device_ptr` points to at least `product(shape) * sizeof(f32)` bytes
/// - The memory remains valid for the lifetime of the returned guard
/// - The memory is accessible on the CUDA device matched by `mem_info`
pub unsafe fn create_cuda_tensor_f32(
    mem_info: &MemoryInfo,
    device_ptr: u64,
    shape: &[i64],
) -> anyhow::Result<OrtValueGuard> {
    let mut elements: usize = 1;
    for &d in shape {
        elements *= d as usize;
    }
    let bytes = elements * core::mem::size_of::<f32>();

    let mut value: *mut ort_sys::OrtValue = core::ptr::null_mut();
    let status = unsafe {
        (ort::api().CreateTensorWithDataAsOrtValue)(
            mem_info.ptr(),
            device_ptr as *mut core::ffi::c_void,
            bytes,
            shape.as_ptr(),
            shape.len(),
            ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut value,
        )
    };
    anyhow::ensure!(
        status.0.is_null() && !value.is_null(),
        "CreateTensorWithDataAsOrtValue failed"
    );
    Ok(OrtValueGuard(value))
}

/// Bind an OrtValue to a session input via low-level C API.
///
/// # Safety
/// `binding` must be a valid IoBinding not currently being used by another thread.
pub unsafe fn bind_input_raw(
    binding: &mut IoBinding,
    name: &str,
    value: &OrtValueGuard,
) -> anyhow::Result<()> {
    let cname = CString::new(name)?;
    let status =
        unsafe { (ort::api().BindInput)(binding.ptr().cast_mut(), cname.as_ptr(), value.0) };
    anyhow::ensure!(status.0.is_null(), "BindInput {} failed", name);
    Ok(())
}

/// Bind an OrtValue to a session output via low-level C API.
///
/// # Safety
/// Same constraints as `bind_input_raw`. Output buffer writes directly into
/// the user-provided device buffer (zero-copy).
pub unsafe fn bind_output_raw(
    binding: &mut IoBinding,
    name: &str,
    value: &OrtValueGuard,
) -> anyhow::Result<()> {
    let cname = CString::new(name)?;
    let status =
        unsafe { (ort::api().BindOutput)(binding.ptr().cast_mut(), cname.as_ptr(), value.0) };
    anyhow::ensure!(status.0.is_null(), "BindOutput {} failed", name);
    Ok(())
}

/// Run a one-input/one-output model directly against caller-owned CUDA buffers.
/// The cached IoBinding avoids an ORT output allocation and the following D2D copy.
#[allow(clippy::too_many_arguments)]
pub fn run_bound_f32(
    manager: &mut ModelManager,
    stream: &CudaStream,
    session_name: &str,
    input_name: &str,
    input: &CudaSlice<f32>,
    input_shape: &[i64],
    output_name: &str,
    output: &mut CudaSlice<f32>,
    output_shape: &[i64],
) -> anyhow::Result<()> {
    let memory = MemoryInfo::new(
        AllocationDevice::CUDA,
        manager.device_id(),
        AllocatorType::Device,
        MemoryType::Default,
    )?;
    let (input_ptr, _input_guard) = input.device_ptr(stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(stream);
    let input_value = unsafe { create_cuda_tensor_f32(&memory, input_ptr, input_shape)? };
    let output_value = unsafe { create_cuda_tensor_f32(&memory, output_ptr, output_shape)? };
    let (session, binding) = manager.session_and_binding(session_name)?;
    unsafe {
        bind_input_raw(binding, input_name, &input_value)?;
        bind_output_raw(binding, output_name, &output_value)?;
    }
    binding.synchronize_inputs()?;
    let _ = session.run_binding(binding)?;
    binding.synchronize_outputs()?;
    binding.clear();
    Ok(())
}
