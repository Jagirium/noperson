use noperson::backend::{
    Buffer, CompatibilityError, CompiledCapabilities, ComputeBackendKind, ComputeContext,
    ComputeError, ComputeOps, ComputeStream, DeviceArchitecture, DeviceCapabilities,
    InferenceProvider, KernelArtifactDescriptor, KernelTarget, LaunchGeometry, PinnedHostBuffer,
    SubgroupRequirement, SubgroupWidth, Toolchain, query_device_capabilities,
};
use noperson::config::settings::ExecutionProvider;

fn cuda_device() -> DeviceCapabilities {
    DeviceCapabilities {
        backend: ComputeBackendKind::Cuda,
        architecture: DeviceArchitecture::NvidiaSm(86),
        native_subgroup_width: SubgroupWidth::WAVE32,
        supported_subgroup_widths: vec![SubgroupWidth::WAVE32],
        max_workgroup_size: 1_024,
        max_workgroup_dimensions: [1_024, 1_024, 64],
        shared_memory_per_workgroup: 48 * 1_024,
        total_memory_bytes: 24 * 1_024 * 1_024 * 1_024,
    }
}

fn cuda_artifact(subgroup: SubgroupRequirement) -> KernelArtifactDescriptor {
    KernelArtifactDescriptor {
        name: "detector_decode",
        backend: ComputeBackendKind::Cuda,
        toolchain: Toolchain::Cuda {
            major: 12,
            minor: 8,
        },
        target: KernelTarget::NvidiaFatbin { minimum_sm: 75 },
        subgroup,
    }
}

#[test]
fn active_backend_exposes_vendor_neutral_zero_cost_types() {
    fn type_exists<T>() {}

    type_exists::<Buffer<f32>>();
    type_exists::<PinnedHostBuffer<f32>>();
    type_exists::<ComputeContext>();
    type_exists::<ComputeStream>();
    type_exists::<ComputeError>();
    type_exists::<ComputeOps>();
    type_exists::<fn(&ComputeContext) -> Result<DeviceCapabilities, ComputeError>>();
    let _query = query_device_capabilities;
}

#[test]
fn compiled_capabilities_keep_compute_and_inference_separate() {
    let compiled = CompiledCapabilities::current();
    assert_eq!(compiled.compute_backend, ComputeBackendKind::Cuda);
    assert!(compiled.supports(InferenceProvider::Cuda));

    #[cfg(feature = "tensorrt")]
    assert!(compiled.supports(InferenceProvider::TensorRt));
    #[cfg(not(feature = "tensorrt"))]
    assert!(!compiled.supports(InferenceProvider::TensorRt));

    assert!(!compiled.supports(InferenceProvider::Rocm));
    assert!(!compiled.supports(InferenceProvider::MiGraphX));
}

#[test]
fn settings_use_the_shared_inference_provider_contract() {
    let provider: InferenceProvider = ExecutionProvider::TensorRt;

    assert_eq!(provider, InferenceProvider::TensorRt);
    assert_eq!(serde_json::to_string(&provider).unwrap(), "\"TensorRT\"");
    assert_eq!(
        serde_json::from_str::<InferenceProvider>("\"TensorRT\"").unwrap(),
        InferenceProvider::TensorRt
    );
}

#[test]
fn unavailable_provider_is_rejected_by_compiled_capabilities() {
    let error = CompiledCapabilities::current()
        .require(InferenceProvider::Rocm)
        .unwrap_err();

    assert_eq!(error.requested, InferenceProvider::Rocm);
    assert_eq!(error.compiled, CompiledCapabilities::current());
}

#[test]
fn inference_provider_declares_its_compute_backend() {
    assert_eq!(
        InferenceProvider::Cuda.compute_backend(),
        ComputeBackendKind::Cuda
    );
    assert_eq!(
        InferenceProvider::TensorRt.compute_backend(),
        ComputeBackendKind::Cuda
    );
    assert_eq!(
        InferenceProvider::Rocm.compute_backend(),
        ComputeBackendKind::Rocm
    );
    assert_eq!(
        InferenceProvider::MiGraphX.compute_backend(),
        ComputeBackendKind::Rocm
    );
}

#[test]
fn subgroup_width_rejects_zero() {
    assert!(SubgroupWidth::new(0).is_none());
    assert_eq!(SubgroupWidth::new(32), Some(SubgroupWidth::WAVE32));
    assert_eq!(SubgroupWidth::new(64), Some(SubgroupWidth::WAVE64));
}

#[test]
fn exact_subgroup_mismatch_is_rejected_before_kernel_load() {
    let error = cuda_artifact(SubgroupRequirement::Exact(SubgroupWidth::WAVE64))
        .validate_for(&cuda_device(), LaunchGeometry::new([256, 1, 1], 0))
        .unwrap_err();

    assert_eq!(
        error,
        CompatibilityError::SubgroupMismatch {
            required: SubgroupWidth::WAVE64,
            supported: vec![SubgroupWidth::WAVE32],
        }
    );
}

#[test]
fn architecture_mismatch_is_rejected_before_kernel_load() {
    let mut artifact = cuda_artifact(SubgroupRequirement::Agnostic);
    artifact.target = KernelTarget::NvidiaFatbin { minimum_sm: 90 };

    assert!(matches!(
        artifact.validate_for(&cuda_device(), LaunchGeometry::new([256, 1, 1], 0)),
        Err(CompatibilityError::ArchitectureMismatch { .. })
    ));
}

#[test]
fn oversized_workgroup_is_rejected_before_launch() {
    let error = cuda_artifact(SubgroupRequirement::Agnostic)
        .validate_for(&cuda_device(), LaunchGeometry::new([1_025, 1, 1], 0))
        .unwrap_err();

    assert_eq!(
        error,
        CompatibilityError::WorkgroupTooLarge {
            requested: [1_025, 1, 1],
            maximum_threads: 1_024,
            maximum_dimensions: [1_024, 1_024, 64],
        }
    );
}
