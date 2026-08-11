use std::num::NonZeroU16;

use serde::{Deserialize, Serialize};

/// Compute API compiled into the current binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeBackendKind {
    Cuda,
    Rocm,
}

/// ONNX Runtime execution provider selected for model sessions.
///
/// Variants are never feature-gated so serialized workspaces remain readable
/// when opened by a binary built without their implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferenceProvider {
    Cuda,
    #[serde(rename = "TensorRT", alias = "TensorRt")]
    TensorRt,
    Rocm,
    MiGraphX,
}

impl InferenceProvider {
    pub const fn compute_backend(self) -> ComputeBackendKind {
        match self {
            Self::Cuda | Self::TensorRt => ComputeBackendKind::Cuda,
            Self::Rocm | Self::MiGraphX => ComputeBackendKind::Rocm,
        }
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Cuda => "Native CUDA",
            Self::TensorRt => "TensorRT",
            Self::Rocm => "ROCm",
            Self::MiGraphX => "MIGraphX",
        }
    }
}

/// Backends and providers present in this exact Cargo build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompiledCapabilities {
    pub compute_backend: ComputeBackendKind,
    pub inference_providers: &'static [InferenceProvider],
}

impl CompiledCapabilities {
    pub const fn current() -> Self {
        #[cfg(all(feature = "cuda", feature = "tensorrt"))]
        {
            return Self {
                compute_backend: ComputeBackendKind::Cuda,
                inference_providers: &[InferenceProvider::Cuda, InferenceProvider::TensorRt],
            };
        }

        #[cfg(all(feature = "cuda", not(feature = "tensorrt")))]
        {
            return Self {
                compute_backend: ComputeBackendKind::Cuda,
                inference_providers: &[InferenceProvider::Cuda],
            };
        }

        #[cfg(all(feature = "rocm", feature = "migraphx"))]
        {
            return Self {
                compute_backend: ComputeBackendKind::Rocm,
                inference_providers: &[InferenceProvider::Rocm, InferenceProvider::MiGraphX],
            };
        }

        #[cfg(all(feature = "rocm", not(feature = "migraphx")))]
        {
            return Self {
                compute_backend: ComputeBackendKind::Rocm,
                inference_providers: &[InferenceProvider::Rocm],
            };
        }

        #[allow(unreachable_code)]
        Self {
            compute_backend: ComputeBackendKind::Cuda,
            inference_providers: &[],
        }
    }

    pub fn supports(self, provider: InferenceProvider) -> bool {
        self.inference_providers.contains(&provider)
    }

    pub fn require(self, provider: InferenceProvider) -> Result<(), ProviderUnavailableError> {
        if self.supports(provider) {
            Ok(())
        } else {
            Err(ProviderUnavailableError {
                requested: provider,
                compiled: self,
            })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "inference provider {requested:?} is not compiled; available providers: {available:?}",
    available = .compiled.inference_providers
)]
pub struct ProviderUnavailableError {
    pub requested: InferenceProvider,
    pub compiled: CompiledCapabilities,
}

/// Number of lanes executing as one hardware subgroup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct SubgroupWidth(NonZeroU16);

impl SubgroupWidth {
    pub const WAVE32: Self = Self(NonZeroU16::new(32).unwrap());
    pub const WAVE64: Self = Self(NonZeroU16::new(64).unwrap());

    pub const fn new(width: u16) -> Option<Self> {
        match NonZeroU16::new(width) {
            Some(width) => Some(Self(width)),
            None => None,
        }
    }

    pub const fn get(self) -> u16 {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubgroupRequirement {
    Agnostic,
    Exact(SubgroupWidth),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceArchitecture {
    NvidiaSm(u16),
    AmdGfx(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KernelTarget {
    NvidiaFatbin { minimum_sm: u16 },
    AmdGfx(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Toolchain {
    Cuda { major: u16, minor: u16 },
    Rocm { major: u16, minor: u16 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceCapabilities {
    pub backend: ComputeBackendKind,
    pub architecture: DeviceArchitecture,
    pub native_subgroup_width: SubgroupWidth,
    pub supported_subgroup_widths: Vec<SubgroupWidth>,
    pub max_workgroup_size: u32,
    pub max_workgroup_dimensions: [u32; 3],
    pub shared_memory_per_workgroup: u64,
    pub total_memory_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchGeometry {
    pub workgroup: [u32; 3],
    pub dynamic_shared_memory_bytes: u64,
}

impl LaunchGeometry {
    pub const fn new(workgroup: [u32; 3], dynamic_shared_memory_bytes: u64) -> Self {
        Self {
            workgroup,
            dynamic_shared_memory_bytes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelArtifactDescriptor {
    pub name: &'static str,
    pub backend: ComputeBackendKind,
    pub toolchain: Toolchain,
    pub target: KernelTarget,
    pub subgroup: SubgroupRequirement,
}

impl KernelArtifactDescriptor {
    /// Validate immutable artifact and launch metadata before loading a module.
    pub fn validate_for(
        &self,
        device: &DeviceCapabilities,
        launch: LaunchGeometry,
    ) -> Result<(), CompatibilityError> {
        if self.backend != device.backend {
            return Err(CompatibilityError::BackendMismatch {
                required: self.backend,
                actual: device.backend,
            });
        }

        let target_matches = match (&self.target, &device.architecture) {
            (KernelTarget::NvidiaFatbin { minimum_sm }, DeviceArchitecture::NvidiaSm(actual)) => {
                actual >= minimum_sm
            }
            (KernelTarget::AmdGfx(required), DeviceArchitecture::AmdGfx(actual)) => {
                required == actual
            }
            _ => false,
        };
        if !target_matches {
            return Err(CompatibilityError::ArchitectureMismatch {
                required: self.target.clone(),
                actual: device.architecture.clone(),
            });
        }

        if let SubgroupRequirement::Exact(required) = self.subgroup
            && !device.supported_subgroup_widths.contains(&required)
        {
            return Err(CompatibilityError::SubgroupMismatch {
                required,
                supported: device.supported_subgroup_widths.clone(),
            });
        }

        let [x, y, z] = launch.workgroup;
        let threads = u64::from(x) * u64::from(y) * u64::from(z);
        let dimensions_fit = launch
            .workgroup
            .iter()
            .zip(device.max_workgroup_dimensions)
            .all(|(requested, maximum)| *requested > 0 && *requested <= maximum);
        if !dimensions_fit || threads > u64::from(device.max_workgroup_size) {
            return Err(CompatibilityError::WorkgroupTooLarge {
                requested: launch.workgroup,
                maximum_threads: device.max_workgroup_size,
                maximum_dimensions: device.max_workgroup_dimensions,
            });
        }

        if launch.dynamic_shared_memory_bytes > device.shared_memory_per_workgroup {
            return Err(CompatibilityError::SharedMemoryTooLarge {
                requested: launch.dynamic_shared_memory_bytes,
                maximum: device.shared_memory_per_workgroup,
            });
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CompatibilityError {
    #[error("kernel requires {required:?}, but device uses {actual:?}")]
    BackendMismatch {
        required: ComputeBackendKind,
        actual: ComputeBackendKind,
    },
    #[error("kernel target {required:?} is incompatible with device {actual:?}")]
    ArchitectureMismatch {
        required: KernelTarget,
        actual: DeviceArchitecture,
    },
    #[error("kernel requires subgroup width {required:?}, supported widths are {supported:?}")]
    SubgroupMismatch {
        required: SubgroupWidth,
        supported: Vec<SubgroupWidth>,
    },
    #[error(
        "workgroup {requested:?} exceeds {maximum_threads} threads or dimensions {maximum_dimensions:?}"
    )]
    WorkgroupTooLarge {
        requested: [u32; 3],
        maximum_threads: u32,
        maximum_dimensions: [u32; 3],
    },
    #[error("dynamic shared memory {requested} exceeds device maximum {maximum}")]
    SharedMemoryTooLarge { requested: u64, maximum: u64 },
}
