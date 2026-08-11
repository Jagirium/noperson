//! CUDA Driver API capability discovery translated into backend-neutral types.

use cudarc::driver::{CudaContext, DriverError, sys};

use crate::backend::{ComputeBackendKind, DeviceArchitecture, DeviceCapabilities, SubgroupWidth};

fn invalid_attribute() -> DriverError {
    DriverError(sys::CUresult::CUDA_ERROR_UNKNOWN)
}

fn attribute_u32(
    context: &CudaContext,
    attribute: sys::CUdevice_attribute,
) -> Result<u32, DriverError> {
    u32::try_from(context.attribute(attribute)?).map_err(|_| invalid_attribute())
}

pub fn query_device_capabilities(context: &CudaContext) -> Result<DeviceCapabilities, DriverError> {
    let (major, minor) = context.compute_capability()?;
    let major = u16::try_from(major).map_err(|_| invalid_attribute())?;
    let minor = u16::try_from(minor).map_err(|_| invalid_attribute())?;
    let subgroup = SubgroupWidth::new(
        u16::try_from(attribute_u32(
            context,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        )?)
        .map_err(|_| invalid_attribute())?,
    )
    .ok_or_else(invalid_attribute)?;

    Ok(DeviceCapabilities {
        backend: ComputeBackendKind::Cuda,
        architecture: DeviceArchitecture::NvidiaSm(major * 10 + minor),
        native_subgroup_width: subgroup,
        supported_subgroup_widths: vec![subgroup],
        max_workgroup_size: attribute_u32(
            context,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        )?,
        max_workgroup_dimensions: [
            attribute_u32(
                context,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
            )?,
            attribute_u32(
                context,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
            )?,
            attribute_u32(
                context,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
            )?,
        ],
        shared_memory_per_workgroup: u64::from(attribute_u32(
            context,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
        )?),
        total_memory_bytes: u64::try_from(context.total_mem()?).map_err(|_| invalid_attribute())?,
    })
}
