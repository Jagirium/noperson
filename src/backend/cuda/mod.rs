//! NVIDIA CUDA implementation selected by the `cuda` Cargo feature.

mod capabilities;
pub(crate) mod inference;
pub mod interop;
pub mod npp;
pub mod ops;

pub use capabilities::query_device_capabilities;
pub use cudarc::driver::{
    CudaContext as ComputeContext, CudaEvent as ComputeEvent, CudaSlice as Buffer,
    CudaStream as ComputeStream, DevicePtr, DevicePtrMut, DeviceRepr, DriverError as ComputeError,
    PinnedHostSlice as PinnedHostBuffer,
};
pub(crate) use cudarc::driver::{
    DeviceSlice, SyncOnDrop, result as driver_result, sys as driver_sys,
};
pub use ops::CudaOps;
pub use ops::CudaOps as ComputeOps;

#[derive(Debug, thiserror::Error)]
pub enum CudaInitializationError {
    #[error(transparent)]
    Driver(#[from] cudarc::driver::DriverError),
    #[error(transparent)]
    IncompatibleKernel(#[from] crate::backend::CompatibilityError),
}

pub use CudaInitializationError as ComputeInitializationError;
