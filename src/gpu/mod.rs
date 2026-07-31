pub mod interop;
pub mod kernels;
pub mod npp;
pub mod ops;

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream, DriverError};

/// Central GPU context — owns the CUDA device, stream, loaded PTX modules, and NPP handles.
pub struct GpuDevice {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub kernels: kernels::KernelCache,
}

impl GpuDevice {
    pub fn new(device_id: usize) -> Result<Self, DriverError> {
        let ctx = CudaContext::new(device_id)?;
        let stream = ctx.default_stream();

        let kernels = kernels::KernelCache::load(&ctx)?;

        Ok(Self {
            ctx,
            stream,
            kernels,
        })
    }
}
