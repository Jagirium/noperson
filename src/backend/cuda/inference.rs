//! ONNX Runtime providers backed by the selected CUDA compute backend.

use ort::ep::ExecutionProviderDispatch;

use crate::backend::InferenceProvider;
use crate::backend::inference::InferenceSessionConfig;

pub(crate) fn execution_providers(
    config: InferenceSessionConfig<'_>,
) -> anyhow::Result<Vec<ExecutionProviderDispatch>> {
    let cuda = match config.compute_stream {
        Some(stream) => unsafe {
            ort::ep::CUDA::default()
                .with_device_id(config.device_id)
                .with_compute_stream(stream.get() as *mut ())
                .build()
                .error_on_failure()
        },
        None => ort::ep::CUDA::default()
            .with_device_id(config.device_id)
            .build()
            .error_on_failure(),
    };

    let mut providers = Vec::with_capacity(2);
    if config.provider == InferenceProvider::TensorRt {
        #[cfg(feature = "tensorrt")]
        providers.push(tensorrt_provider(&config)?);
    }
    providers.push(cuda);
    Ok(providers)
}

#[cfg(feature = "tensorrt")]
fn tensorrt_provider(
    config: &InferenceSessionConfig<'_>,
) -> anyhow::Result<ExecutionProviderDispatch> {
    let cache_path = config.cache_root.join("trt-cache-fp32-parity");
    std::fs::create_dir_all(&cache_path)?;
    let mut provider = ort::ep::TensorRT::default()
        .with_device_id(config.device_id)
        .with_engine_cache(true)
        .with_engine_cache_path(cache_path.to_string_lossy())
        .with_timing_cache(true)
        .with_timing_cache_path(cache_path.to_string_lossy())
        .with_layer_norm_fp32_fallback(true)
        .with_builder_optimization_level(5)
        .with_cuda_graph(false);
    if let Some(stream) = config.compute_stream {
        provider = unsafe { provider.with_compute_stream(stream.get() as *mut ()) };
    }
    Ok(provider.build().error_on_failure())
}
