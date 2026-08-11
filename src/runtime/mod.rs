//! GPU runtime bootstrap and architecture-specific library selection.

use std::path::{Path, PathBuf};

mod bootstrap;
mod install;
mod registry;

pub use bootstrap::{BootstrapOutcome, prepare, reexec};
pub use install::{RuntimeInstallError, ensure_runtime};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComputeCapability {
    pub major: i32,
    pub minor: i32,
}

impl ComputeCapability {
    pub fn tensorrt_shard(self) -> TensorRtShard {
        match (self.major, self.minor) {
            (7, 5) => TensorRtShard::Sm75,
            (8, 0) => TensorRtShard::Sm80,
            (8, 6) => TensorRtShard::Sm86,
            (8, 9) => TensorRtShard::Sm89,
            (9, 0) => TensorRtShard::Sm90,
            (10, 0) => TensorRtShard::Sm100,
            (12, 0) => TensorRtShard::Sm120,
            _ => TensorRtShard::Ptx,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorRtShard {
    Sm75,
    Sm80,
    Sm86,
    Sm89,
    Sm90,
    Sm100,
    Sm120,
    Ptx,
}

impl TensorRtShard {
    pub const fn directory(self) -> &'static str {
        match self {
            Self::Sm75 => "sm75",
            Self::Sm80 => "sm80",
            Self::Sm86 => "sm86",
            Self::Sm89 => "sm89",
            Self::Sm90 => "sm90",
            Self::Sm100 => "sm100",
            Self::Sm120 => "sm120",
            Self::Ptx => "ptx",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeLayout {
    root: PathBuf,
    shard: TensorRtShard,
}

impl RuntimeLayout {
    pub fn new(root: PathBuf, shard: TensorRtShard) -> Self {
        Self { root, shard }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn base(&self) -> PathBuf {
        self.root.join("base")
    }

    pub fn tensorrt_base(&self) -> PathBuf {
        self.root.join("trt/base")
    }

    pub fn tensorrt_shard(&self) -> PathBuf {
        self.root.join("trt").join(self.shard.directory())
    }

    pub fn launch_dir(&self) -> PathBuf {
        self.root.join("launch")
    }

    pub fn ort_shared_provider(&self) -> PathBuf {
        self.base().join(platform_ort_provider("shared"))
    }

    pub fn ort_cuda_provider(&self) -> PathBuf {
        self.base().join(platform_ort_provider("cuda"))
    }

    pub fn ort_tensorrt_provider(&self) -> PathBuf {
        self.tensorrt_base().join(platform_ort_provider("tensorrt"))
    }

    pub fn library_paths(&self) -> Vec<PathBuf> {
        let mut paths = vec![self.base()];
        if cfg!(feature = "tensorrt") {
            paths.extend([self.tensorrt_base(), self.tensorrt_shard()]);
        }
        paths
    }

    pub fn is_complete(&self) -> bool {
        let base = self.base();
        let base_complete = [
            base.join(platform_library("nppc")),
            base.join(platform_library("nppig")),
            base.join(platform_library("nppif")),
            base.join(platform_library("nppim")),
            self.ort_shared_provider(),
            self.ort_cuda_provider(),
        ]
        .into_iter()
        .all(|path| path.is_file());
        if !base_complete || !cfg!(feature = "tensorrt") {
            return base_complete;
        }

        let trt_base = self.tensorrt_base();
        let shard = self.tensorrt_shard();
        [
            trt_base.join(platform_trt_library()),
            self.ort_tensorrt_provider(),
        ]
        .into_iter()
        .all(|path| path.is_file())
            && shard.is_dir()
            && shard
                .read_dir()
                .is_ok_and(|mut entries| entries.any(|entry| entry.is_ok()))
    }
}

#[cfg(target_os = "windows")]
fn platform_library(stem: &str) -> String {
    format!("{stem}64_12.dll")
}

#[cfg(not(target_os = "windows"))]
fn platform_library(stem: &str) -> String {
    format!("lib{stem}.so.12")
}

#[cfg(target_os = "windows")]
fn platform_ort_provider(provider: &str) -> String {
    format!("onnxruntime_providers_{provider}.dll")
}

#[cfg(not(target_os = "windows"))]
fn platform_ort_provider(provider: &str) -> String {
    format!("libonnxruntime_providers_{provider}.so")
}

#[cfg(target_os = "windows")]
fn platform_trt_library() -> &'static str {
    "nvinfer_10.dll"
}

#[cfg(not(target_os = "windows"))]
fn platform_trt_library() -> &'static str {
    "libnvinfer.so.10"
}
