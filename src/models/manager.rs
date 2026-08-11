//! ONNX session pool: lazy loading, CUDA EP, session options.
//!
//! Port of crosswap/app/processors/models_processor.py

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use ort::session::{IoBinding, Session};

use crate::backend::inference::{InferenceSessionConfig, execution_providers};
use crate::config::settings::ExecutionProvider;

/// Ordering contract between caller-owned CUDA buffers and ONNX Runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingFencePolicy {
    /// The CUDA EP and the caller use the exact same non-null stream. CUDA
    /// stream ordering is sufficient; device-wide IoBinding fences are wasteful.
    SameCudaStream,
    /// The provider may consume/produce on another stream. Fence both sides.
    FenceInputsAndOutputs,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ComputeStreamError {
    #[error("CUDA compute stream pointer must be non-null")]
    Null,
    #[error("CUDA compute stream cannot change after a session or binding is loaded")]
    RuntimeStateLoaded,
}

fn validate_compute_stream_update(
    current: Option<NonZeroUsize>,
    requested: *mut (),
    runtime_state_loaded: bool,
) -> Result<NonZeroUsize, ComputeStreamError> {
    let requested = NonZeroUsize::new(requested as usize).ok_or(ComputeStreamError::Null)?;
    if runtime_state_loaded && current != Some(requested) {
        return Err(ComputeStreamError::RuntimeStateLoaded);
    }
    Ok(requested)
}

/// ONNX session manager — loads models on demand with CUDA EP.
///
/// IoBinding reuse: one binding per model, cached across frames.
/// Python equivalent: thread-local `get_io_binding(model_name)` cache.
/// Reusing the binding avoids per-frame allocation and setup overhead.
pub struct ModelManager {
    /// Loaded ONNX sessions keyed by model name.
    sessions: HashMap<String, Session>,
    /// Reusable IoBindings keyed by model name.
    /// Split from `sessions` so we can split-borrow both fields simultaneously.
    bindings: HashMap<String, IoBinding>,
    /// Base directory for ONNX model files.
    pub(crate) models_dir: PathBuf,
    /// Primary execution provider selected for every loaded ONNX session.
    provider: ExecutionProvider,
    /// CUDA device selected by the immutable engine generation.
    device_id: i32,
    /// Inswapper128 emap matrix [512, 512] — extracted from ONNX initializer.
    pub emap: Option<Vec<f32>>,
    /// Shared CUDA compute stream pointer — ort sessions use this so that
    /// inference is ordered with cudarc kernels in the same stream (eliminates
    /// context switch + internal cudaMemcpy overhead for IoBinding inputs).
    compute_stream: Option<NonZeroUsize>,
}

impl ModelManager {
    pub fn new(models_dir: impl AsRef<Path>) -> Self {
        Self::with_provider(models_dir, ExecutionProvider::Cuda)
    }

    pub fn with_provider(models_dir: impl AsRef<Path>, provider: ExecutionProvider) -> Self {
        Self::with_execution(models_dir, provider, 0)
    }

    pub fn with_execution(
        models_dir: impl AsRef<Path>,
        provider: ExecutionProvider,
        device_id: i32,
    ) -> Self {
        Self {
            sessions: HashMap::new(),
            bindings: HashMap::new(),
            models_dir: models_dir.as_ref().to_path_buf(),
            provider,
            device_id,
            emap: None,
            compute_stream: None,
        }
    }

    pub fn provider(&self) -> ExecutionProvider {
        self.provider
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    /// Set the shared CUDA compute stream pointer for subsequent session loads.
    /// Must be called BEFORE `load()` for the stream sharing to take effect.
    pub fn set_compute_stream(&mut self, stream_ptr: *mut ()) -> Result<(), ComputeStreamError> {
        let runtime_state_loaded = !self.sessions.is_empty() || !self.bindings.is_empty();
        self.compute_stream = Some(validate_compute_stream_update(
            self.compute_stream,
            stream_ptr,
            runtime_state_loaded,
        )?);
        Ok(())
    }

    /// Select the minimum safe IoBinding synchronization policy for buffers
    /// produced and consumed on `work_stream`.
    pub fn binding_fence_policy(&self, work_stream: *mut ()) -> BindingFencePolicy {
        let work_stream = NonZeroUsize::new(work_stream as usize);
        if self.provider == ExecutionProvider::Cuda
            && work_stream.is_some()
            && self.compute_stream == work_stream
        {
            BindingFencePolicy::SameCudaStream
        } else {
            BindingFencePolicy::FenceInputsAndOutputs
        }
    }

    /// Load an ONNX model with CUDA execution provider.
    /// Session options match Python: ORT_ENABLE_ALL, ORT_SEQUENTIAL, 1 thread.
    pub fn load(&mut self, name: &str, filename: &str) -> anyhow::Result<()> {
        let model_path = self.models_dir.join(filename);
        self.load_path(name, &model_path)
    }

    pub fn load_path(&mut self, name: &str, model_path: &Path) -> anyhow::Result<()> {
        if self.sessions.contains_key(name) {
            return Ok(());
        }

        anyhow::ensure!(
            model_path.exists(),
            "Model not found: {}",
            model_path.display()
        );

        let providers = execution_providers(InferenceSessionConfig {
            provider: self.provider,
            device_id: self.device_id,
            compute_stream: self.compute_stream,
            cache_root: &self.models_dir,
        })?;

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_log_level(ort::logging::LogLevel::Warning)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_execution_providers(providers)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(model_path)?;

        self.sessions.insert(name.to_string(), session);
        tracing::info!("Loaded model: {name} from {}", model_path.display());
        Ok(())
    }

    /// Get a mutable reference to a loaded session.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Session> {
        self.sessions.get_mut(name)
    }

    /// Get a reference to a loaded session.
    pub fn get(&self, name: &str) -> Option<&Session> {
        self.sessions.get(name)
    }

    /// Get a (session, reusable binding) pair for `name`. Creates the binding on
    /// first call and caches it. Split-borrow: both fields are distinct so the
    /// borrow checker allows a simultaneous &mut to each.
    /// The pipeline's common `run_bound_values` helper is the only hot-path
    /// caller so binding cleanup and provider-aware ordering cannot diverge.
    pub fn session_and_binding(
        &mut self,
        name: &str,
    ) -> anyhow::Result<(&mut Session, &mut IoBinding)> {
        // Ensure the binding is created for this model. This has to be a
        // separate scope so the temporary &mut self.sessions borrow ends
        // before we take the final split borrow.
        if !self.bindings.contains_key(name) {
            let session = self
                .sessions
                .get_mut(name)
                .ok_or_else(|| anyhow::anyhow!("Model {name} not loaded"))?;
            let binding = session.create_binding()?;
            self.bindings.insert(name.to_string(), binding);
        }

        // Split-borrow: sessions and bindings are distinct HashMap fields.
        let session = self
            .sessions
            .get_mut(name)
            .ok_or_else(|| anyhow::anyhow!("Model {name} not loaded"))?;
        let binding = self
            .bindings
            .get_mut(name)
            .ok_or_else(|| anyhow::anyhow!("Binding for {name} not in cache"))?;
        Ok((session, binding))
    }

    /// Get models directory path.
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// Check if a model is loaded.
    pub fn is_loaded(&self, name: &str) -> bool {
        self.sessions.contains_key(name)
    }

    /// Unload a model, freeing its GPU memory.
    pub fn unload(&mut self, name: &str) {
        // Drop the binding first — it holds refs into the session.
        self.bindings.remove(name);
        if self.sessions.remove(name).is_some() {
            tracing::info!("Unloaded model: {name}");
        }
    }

    /// Extract emap from Inswapper128 ONNX file (last initializer, shape [512, 512]).
    /// Must be called before using the swapper.
    pub fn load_emap(&mut self, inswapper_filename: &str) -> anyhow::Result<()> {
        let model_path = self.models_dir.join(inswapper_filename);
        anyhow::ensure!(
            model_path.exists(),
            "Inswapper model not found: {}",
            model_path.display()
        );

        self.load_emap_file("emap.bin")
    }

    /// Load a pre-extracted Inswapper emap selected by an immutable generation.
    pub fn load_emap_file(&mut self, filename: &str) -> anyhow::Result<()> {
        let emap_path = self.models_dir.join(filename);
        if emap_path.exists() {
            let emap_bytes = std::fs::read(&emap_path)?;
            anyhow::ensure!(emap_bytes.len() == 512 * 512 * 4, "emap.bin wrong size");
            let emap: Vec<f32> = bytemuck::cast_slice(&emap_bytes).to_vec();
            self.emap = Some(emap);
            tracing::info!("Loaded emap from emap.bin ({} floats)", 512 * 512);
            return Ok(());
        }

        // Fallback: extract from ONNX protobuf (last initializer)
        // The emap was previously extracted in crosswap-core/build.rs
        anyhow::bail!(
            "Inswapper emap not found at {}. Extract it from the swapper ONNX graph.",
            emap_path.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{BindingFencePolicy, ModelManager, validate_compute_stream_update};
    use crate::config::settings::ExecutionProvider;

    fn stream(value: usize) -> *mut () {
        value as *mut ()
    }

    #[test]
    fn fence_policy_requires_the_exact_shared_cuda_stream() {
        let mut manager = ModelManager::with_execution("models", ExecutionProvider::Cuda, 0);
        manager.set_compute_stream(stream(0x1000)).unwrap();

        assert_eq!(
            manager.binding_fence_policy(stream(0x1000)),
            BindingFencePolicy::SameCudaStream
        );
        assert_eq!(
            manager.binding_fence_policy(stream(0x2000)),
            BindingFencePolicy::FenceInputsAndOutputs
        );
        assert_eq!(
            manager.binding_fence_policy(core::ptr::null_mut()),
            BindingFencePolicy::FenceInputsAndOutputs
        );
    }

    #[test]
    fn cuda_without_a_configured_stream_and_tensorrt_always_fence() {
        let cuda = ModelManager::with_execution("models", ExecutionProvider::Cuda, 0);
        assert_eq!(
            cuda.binding_fence_policy(stream(0x1000)),
            BindingFencePolicy::FenceInputsAndOutputs
        );

        let mut tensorrt = ModelManager::with_execution("models", ExecutionProvider::TensorRt, 0);
        tensorrt.set_compute_stream(stream(0x1000)).unwrap();
        assert_eq!(
            tensorrt.binding_fence_policy(stream(0x1000)),
            BindingFencePolicy::FenceInputsAndOutputs
        );
        assert_eq!(
            tensorrt.binding_fence_policy(stream(0x2000)),
            BindingFencePolicy::FenceInputsAndOutputs
        );
    }

    #[test]
    fn tensorrt_without_a_configured_compute_stream_fences() {
        let tensorrt = ModelManager::with_execution("models", ExecutionProvider::TensorRt, 0);

        assert_eq!(
            tensorrt.binding_fence_policy(stream(0x1000)),
            BindingFencePolicy::FenceInputsAndOutputs
        );
    }

    #[test]
    fn compute_stream_update_rejects_null_and_loaded_state_changes() {
        assert!(validate_compute_stream_update(None, core::ptr::null_mut(), false).is_err());

        let first = validate_compute_stream_update(None, stream(0x1000), false).unwrap();
        assert_eq!(first, NonZeroUsize::new(0x1000).unwrap());
        assert!(validate_compute_stream_update(Some(first), stream(0x2000), true).is_err());
        assert_eq!(
            validate_compute_stream_update(Some(first), stream(0x1000), true).unwrap(),
            first
        );
        assert_eq!(
            validate_compute_stream_update(Some(first), stream(0x2000), false).unwrap(),
            NonZeroUsize::new(0x2000).unwrap()
        );
    }

    #[test]
    fn public_setter_rejects_a_null_stream() {
        let mut manager = ModelManager::new("models");
        assert!(manager.set_compute_stream(core::ptr::null_mut()).is_err());
    }
}
