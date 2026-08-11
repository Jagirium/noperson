//! Feature-aware ONNX Runtime provider construction.

use std::num::NonZeroUsize;
use std::path::Path;

use ort::ep::ExecutionProviderDispatch;

use super::{CompiledCapabilities, InferenceProvider};

pub(crate) struct InferenceSessionConfig<'a> {
    pub provider: InferenceProvider,
    pub device_id: i32,
    pub compute_stream: Option<NonZeroUsize>,
    #[cfg_attr(not(feature = "tensorrt"), allow(dead_code))]
    pub cache_root: &'a Path,
}

pub(crate) fn execution_providers(
    config: InferenceSessionConfig<'_>,
) -> anyhow::Result<Vec<ExecutionProviderDispatch>> {
    CompiledCapabilities::current().require(config.provider)?;

    #[cfg(feature = "cuda")]
    return super::cuda::inference::execution_providers(config);

    #[allow(unreachable_code)]
    Ok(Vec::new())
}
