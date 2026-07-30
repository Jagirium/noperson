//! Shared low-level IoBinding helpers for GPU-native ONNX inference.
//!
//! All hot-path inference calls (detect, recognize, swap) use this module
//! to bind CUDA device pointers directly into ort sessions, avoiding the
//! GPU → CPU → GPU roundtrip that `Tensor::from_array` incurs.

use std::ffi::CString;

use ort::AsPointer;
use ort::memory::MemoryInfo;
use ort::session::IoBinding;

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
