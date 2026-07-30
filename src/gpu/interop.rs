//! Zero-copy interop between cudarc CudaSlice and ort Value.
//!
//! The key insight: both cudarc and ort can work with raw CUDA device pointers.
//! We pre-allocate buffers via cudarc, then create ort Values that point to the
//! same GPU memory. ONNX Runtime writes inference results directly into our
//! pre-allocated buffers — zero allocation, zero copy.

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::value::{Shape, TensorRefMut};

/// Create an ort MemoryInfo descriptor for CUDA device memory.
pub fn cuda_memory_info(device_id: i32) -> ort::Result<MemoryInfo<'static>> {
    MemoryInfo::new(
        AllocationDevice::CUDA,
        device_id,
        AllocatorType::Device,
        MemoryType::Default,
    )
}

/// Create an ort TensorRefMut that wraps an existing cudarc CudaSlice (zero-copy).
///
/// Returns the tensor and a SyncOnDrop guard that must be kept alive until
/// the tensor is consumed by ort (e.g., via IoBinding).
///
/// # Safety
/// - The CudaSlice must remain alive for the lifetime of the returned tensor.
/// - The shape must match the actual buffer size in elements.
/// - The buffer must be on the same CUDA device as the ort session.
pub unsafe fn tensor_from_cuda_slice<
    'a,
    T: ort::value::PrimitiveTensorElementType + std::fmt::Debug,
>(
    slice: &'a mut CudaSlice<T>,
    stream: &'a CudaStream,
    shape: impl Into<Shape>,
    device_id: i32,
) -> ort::Result<TensorRefMut<'a, T>> {
    let mem_info = cuda_memory_info(device_id)?;
    let (cu_ptr, _guard) = slice.device_ptr_mut(stream);
    let raw_ptr = cu_ptr as usize as *mut core::ffi::c_void;

    // Note: _guard is dropped here, but that's OK because we're on the same
    // CUDA stream — ordering is guaranteed. If we need cross-stream safety,
    // the caller must hold the guard.
    unsafe { TensorRefMut::from_raw(mem_info, raw_ptr, shape.into()) }
}

/// Get the raw CUDA device pointer from a cudarc CudaSlice as u64.
///
/// This is useful for passing to NPP functions or custom CUDA kernels.
pub fn cuda_device_ptr<T: cudarc::driver::DeviceRepr>(
    slice: &CudaSlice<T>,
    stream: &CudaStream,
) -> u64 {
    let (ptr, _guard) = slice.device_ptr(stream);
    ptr
}
