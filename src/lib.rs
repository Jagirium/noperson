//! noperson — pure Rust GPU-accelerated face swap.
//!
//! Backend: ort (ONNX Runtime) + CUDA EP + cudarc (GPU compute)
//! UI: egui (immediate mode, native)
//! I/O: nokhwa (webcam), rfd (file dialog), v4l2loopback (virtual camera)
//!
//! Models: ONNX files in models/ — no codegen, runtime loading via ort.

// GPU image kernels deliberately expose dimensions and buffers explicitly.
#![allow(clippy::too_many_arguments)]

pub mod app;
pub mod artifacts;
pub mod config;
pub mod engine;
pub mod gpu;
pub mod io;
pub mod live;
pub mod math;
pub mod models;
pub mod pipeline;
pub mod quality;
pub mod runtime;

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
