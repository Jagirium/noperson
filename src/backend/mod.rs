//! Compile-time compute backend facade and vendor-neutral capability types.

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
compile_error!("select exactly one compute backend feature: `cuda` or `rocm`");

#[cfg(all(feature = "cuda", feature = "rocm"))]
compile_error!("compute backend features `cuda` and `rocm` are mutually exclusive");

#[cfg(all(feature = "tensorrt", not(feature = "cuda")))]
compile_error!("the `tensorrt` feature requires the `cuda` compute backend");

#[cfg(all(feature = "migraphx", not(feature = "rocm")))]
compile_error!("the `migraphx` feature requires the `rocm` compute backend");

#[cfg(feature = "rocm")]
compile_error!(
    "the ROCm feature contract is reserved, but the HIP compute facade is not implemented yet"
);

pub(crate) mod inference;
mod types;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::{
    Buffer, ComputeContext, ComputeError, ComputeEvent, ComputeInitializationError, ComputeOps,
    ComputeStream, DevicePtr, DevicePtrMut, DeviceRepr, PinnedHostBuffer,
    query_device_capabilities,
};

pub use types::{
    CompatibilityError, CompiledCapabilities, ComputeBackendKind, DeviceArchitecture,
    DeviceCapabilities, InferenceProvider, KernelArtifactDescriptor, KernelTarget, LaunchGeometry,
    ProviderUnavailableError, SubgroupRequirement, SubgroupWidth, Toolchain,
};
