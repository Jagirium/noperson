use std::sync::Arc;

use cudarc::driver::CudaContext;
use noperson::gpu::ops::GpuOps;

#[test]
fn converts_and_scales_solid_green_to_nv12_on_gpu() -> anyhow::Result<()> {
    let ctx = Arc::new(CudaContext::new(0)?);
    let stream = ctx.default_stream();
    let gpu = GpuOps::new(&ctx, stream)?;

    // 2x2 planar RGB: R plane, G plane, B plane.
    let chw = [0.0; 4]
        .into_iter()
        .chain([255.0; 4])
        .chain([0.0; 4])
        .collect::<Vec<_>>();
    let source = gpu.upload(&chw)?;
    let mut nv12 = gpu.alloc_zeros_u8(4 * 4 * 3 / 2)?;

    gpu.chw_f32_to_nv12_scaled(&source, &mut nv12, 2, 2, 4, 4)?;
    let output = gpu.download_u8(&nv12)?;

    assert!(output[..16].iter().all(|&value| value == 145));
    assert!(output[16..].chunks_exact(2).all(|uv| uv == [54, 34]));
    Ok(())
}
