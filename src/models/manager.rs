//! ONNX session pool: lazy loading, CUDA EP, session options.
//!
//! Port of crosswap/app/processors/models_processor.py

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ort::session::{IoBinding, Session};

use crate::config::settings::ExecutionProvider;

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
    /// Inswapper128 emap matrix [512, 512] — extracted from ONNX initializer.
    pub emap: Option<Vec<f32>>,
    /// Shared CUDA compute stream pointer — ort sessions use this so that
    /// inference is ordered with cudarc kernels in the same stream (eliminates
    /// context switch + internal cudaMemcpy overhead for IoBinding inputs).
    compute_stream: Option<usize>,
}

impl ModelManager {
    pub fn new(models_dir: impl AsRef<Path>) -> Self {
        Self::with_provider(models_dir, ExecutionProvider::Cuda)
    }

    pub fn with_provider(models_dir: impl AsRef<Path>, provider: ExecutionProvider) -> Self {
        Self {
            sessions: HashMap::new(),
            bindings: HashMap::new(),
            models_dir: models_dir.as_ref().to_path_buf(),
            provider,
            emap: None,
            compute_stream: None,
        }
    }

    /// Set the shared CUDA compute stream pointer for subsequent session loads.
    /// Must be called BEFORE `load()` for the stream sharing to take effect.
    pub fn set_compute_stream(&mut self, stream_ptr: *mut ()) {
        self.compute_stream = Some(stream_ptr as usize);
    }

    /// Load an ONNX model with CUDA execution provider.
    /// Session options match Python: ORT_ENABLE_ALL, ORT_SEQUENTIAL, 1 thread.
    pub fn load(&mut self, name: &str, filename: &str) -> anyhow::Result<()> {
        if self.sessions.contains_key(name) {
            return Ok(());
        }

        let model_path = self.models_dir.join(filename);
        anyhow::ensure!(
            model_path.exists(),
            "Model not found: {}",
            model_path.display()
        );

        // Build CUDA EP. If a user compute stream was set, bind ort to it.
        let cuda_ep = match self.compute_stream {
            Some(ptr) => unsafe {
                ort::ep::CUDA::default()
                    .with_device_id(0)
                    .with_compute_stream(ptr as *mut ())
                    .build()
            },
            None => ort::ep::CUDA::default().build(),
        };

        let mut providers = Vec::new();
        if self.provider == ExecutionProvider::TensorRT {
            // Keep this namespace separate from engines built while Rust
            // forced TensorRT FP16. Cache entries do not encode every provider
            // option, so the old ghosted-face engine must never be reused.
            let cache_path = self.models_dir.join("trt-cache-fp32-parity");
            std::fs::create_dir_all(&cache_path)?;
            let mut trt = ort::ep::TensorRT::default()
                .with_device_id(0)
                .with_engine_cache(true)
                .with_engine_cache_path(cache_path.to_string_lossy())
                .with_timing_cache(true)
                .with_timing_cache_path(cache_path.to_string_lossy())
                // Match Crosswap: do not force global FP16, and keep
                // normalization layers in FP32 when TensorRT lowers the graph.
                .with_layer_norm_fp32_fallback(true)
                .with_builder_optimization_level(5)
                // GPEN uses a cached engine and stable zero-copy buffers; CUDA
                // graph capture remains disabled because multiple TRT sessions
                // on one user stream are unstable in ORT rc.12.
                .with_cuda_graph(false);
            if let Some(ptr) = self.compute_stream {
                trt = unsafe { trt.with_compute_stream(ptr as *mut ()) };
            }
            providers.push(trt.build());
        }
        providers.push(cuda_ep);

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_execution_providers(providers)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(&model_path)?;

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
    ///
    /// Usage pattern (hot path):
    /// ```ignore
    /// let (session, binding) = manager.session_and_binding("Inswapper128")?;
    /// // Bind inputs/outputs to `binding` via low-level API
    /// session.run_binding(binding)?;
    /// binding.clear();
    /// ```
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

        // The emap was pre-extracted to emap.bin (from ONNX graph's last initializer).
        let emap_path = self.models_dir.join("emap.bin");
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
            "emap.bin not found at {}. Extract it from {} using the build script.",
            emap_path.display(),
            model_path.display()
        );
    }
}
