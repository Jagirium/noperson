//! Custom CUDA kernel loader and launcher.
//!
//! Loads pre-compiled PTX from build.rs output, caches kernel functions,
//! and provides typed launcher methods for each kernel.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

/// Holds all loaded CUDA kernel functions.
pub struct KernelCache {
    pub normalize: Option<CudaFunction>,
    pub denormalize: Option<CudaFunction>,
    pub interlace_extract: Option<CudaFunction>,
    pub interlace_scatter: Option<CudaFunction>,
    pub alpha_blend: Option<CudaFunction>,
    pub cosine_sim: Option<CudaFunction>,
    pub matmul_512: Option<CudaFunction>,
    pub l2_normalize: Option<CudaFunction>,
    pub hwc_to_chw: Option<CudaFunction>,
    pub chw_to_hwc: Option<CudaFunction>,
}

impl KernelCache {
    /// Load all PTX modules from the build output directory.
    /// Missing PTX files are silently skipped (kernel = None).
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Self, DriverError> {
        let out_dir = env!("OUT_DIR");

        let load_fn =
            |ctx: &Arc<CudaContext>, filename: &str, entry: &str| -> Option<CudaFunction> {
                let ptx_path = format!("{out_dir}/{filename}.ptx");
                if !std::path::Path::new(&ptx_path).exists() {
                    return None;
                }
                let ptx = Ptx::from_file(&ptx_path);
                let module = ctx.load_module(ptx).ok()?;
                module.load_function(entry).ok()
            };

        Ok(Self {
            normalize: load_fn(ctx, "normalize", "normalize_kernel"),
            denormalize: load_fn(ctx, "normalize", "denormalize_kernel"),
            interlace_extract: load_fn(ctx, "interlace", "interlace_extract_kernel"),
            interlace_scatter: load_fn(ctx, "interlace", "interlace_scatter_kernel"),
            alpha_blend: load_fn(ctx, "alpha_blend", "alpha_blend_kernel"),
            cosine_sim: load_fn(ctx, "cosine_sim", "cosine_sim_kernel"),
            matmul_512: load_fn(ctx, "matmul_512", "matmul_512_kernel"),
            l2_normalize: load_fn(ctx, "matmul_512", "l2_normalize_kernel"),
            hwc_to_chw: load_fn(ctx, "layout_convert", "hwc_to_chw_kernel"),
            chw_to_hwc: load_fn(ctx, "layout_convert", "chw_to_hwc_kernel"),
        })
    }
}

// ── Kernel launchers ──────────────────────────────────────────────────────

/// Normalize in-place: data[i] = data[i] / 255.0
pub fn launch_normalize(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    data: &mut CudaSlice<f32>,
) -> Result<(), DriverError> {
    let n = data.len() as u32;
    let cfg = LaunchConfig::for_num_elems(n);
    let mut builder = stream.launch_builder(kernel);
    builder.arg(data);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }?;
    Ok(())
}

/// Denormalize in-place: data[i] = clamp(data[i] * 255.0, 0, 255)
pub fn launch_denormalize(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    data: &mut CudaSlice<f32>,
) -> Result<(), DriverError> {
    let n = data.len() as u32;
    let cfg = LaunchConfig::for_num_elems(n);
    let mut builder = stream.launch_builder(kernel);
    builder.arg(data);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }?;
    Ok(())
}

/// Interlace extract: face[:, j::dim, i::dim] → tiles[j*dim+i] for batched inswapper.
pub fn launch_interlace_extract(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    face: &CudaSlice<f32>,
    tiles: &mut CudaSlice<f32>,
    dim: u32,
    channels: u32,
    tile_size: u32,
) -> Result<(), DriverError> {
    let n = tiles.len() as u32;
    let cfg = LaunchConfig::for_num_elems(n);
    let mut builder = stream.launch_builder(kernel);
    builder.arg(face);
    builder.arg(tiles);
    builder.arg(&dim);
    builder.arg(&channels);
    builder.arg(&tile_size);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }?;
    Ok(())
}

/// Interlace scatter: tiles[j*dim+i] → face[:, j::dim, i::dim] (inverse of extract).
pub fn launch_interlace_scatter(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    tiles: &CudaSlice<f32>,
    face: &mut CudaSlice<f32>,
    dim: u32,
    channels: u32,
    tile_size: u32,
) -> Result<(), DriverError> {
    let n = tiles.len() as u32;
    let cfg = LaunchConfig::for_num_elems(n);
    let mut builder = stream.launch_builder(kernel);
    builder.arg(tiles);
    builder.arg(face);
    builder.arg(&dim);
    builder.arg(&channels);
    builder.arg(&tile_size);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }?;
    Ok(())
}

/// Alpha blend: out[i] = src[i] * mask[pixel] + dst[i] * (1 - mask[pixel])
pub fn launch_alpha_blend(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    src: &CudaSlice<f32>,
    dst: &CudaSlice<f32>,
    mask: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    n_pixels: u32,
    channels: u32,
) -> Result<(), DriverError> {
    let total = n_pixels * channels;
    let cfg = LaunchConfig::for_num_elems(total);
    let mut builder = stream.launch_builder(kernel);
    builder.arg(src);
    builder.arg(dst);
    builder.arg(mask);
    builder.arg(out);
    builder.arg(&n_pixels);
    builder.arg(&channels);
    unsafe { builder.launch(cfg) }?;
    Ok(())
}
