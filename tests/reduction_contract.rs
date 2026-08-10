use std::fs;

const THREADS_PER_BLOCK: u32 = 256;
const MAX_REDUCTION_BLOCKS: u32 = 1024;

fn reduction_blocks(pixels: u32) -> u32 {
    pixels.div_ceil(THREADS_PER_BLOCK).min(MAX_REDUCTION_BLOCKS)
}

fn selected_count(mask: &[f32], cutoff: f32) -> u32 {
    mask.iter().filter(|&&value| value >= cutoff).count() as u32
}

fn semantic_count(classes: &[u8], region: u32) -> u32 {
    classes
        .iter()
        .filter(|&&class_id| match region {
            0 => class_id == 4 || class_id == 5,
            _ => class_id == 11 || class_id == 12 || class_id == 13,
        })
        .count() as u32
}

fn gray_sum(rgb: &[[f32; 3]]) -> u32 {
    rgb.iter()
        .map(|pixel| (pixel[0] * 0.2989 + pixel[1] * 0.587 + pixel[2] * 0.114).floor() as u32)
        .sum()
}

fn dfm_mean_with_total_pixel_denominator(values: &[f32], mask: &[f32], cutoff: f32) -> f32 {
    values
        .iter()
        .zip(mask)
        .filter_map(|(&value, &mask)| (mask >= cutoff).then_some(value))
        .sum::<f32>()
        / values.len() as f32
}

fn rust_method<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(name).expect("GPU method exists");
    let tail = &source[start..];
    &tail[..tail.find("\n    pub fn ").unwrap_or(tail.len())]
}

fn cuda_kernel<'a>(source: &'a str, name: &str) -> &'a str {
    let start = source.find(name).expect("CUDA kernel exists");
    let tail = &source[start..];
    &tail[..tail.find("extern \"C\" __global__").unwrap_or(tail.len())]
}

fn grid_stride_coverage(pixels: u32, blocks: u32) -> Vec<u32> {
    let stride = u64::from(THREADS_PER_BLOCK) * u64::from(blocks);
    let mut visits = vec![0u32; pixels as usize];
    for block in 0..blocks {
        for thread in 0..THREADS_PER_BLOCK {
            let mut pixel = u64::from(block) * u64::from(THREADS_PER_BLOCK) + u64::from(thread);
            while pixel < u64::from(pixels) {
                visits[pixel as usize] += 1;
                pixel += stride;
            }
        }
    }
    visits
}

fn grid_stride_iterations(pixels: u32, start: u64, stride: u64) -> u64 {
    assert!(stride > 0);
    if start >= u64::from(pixels) {
        0
    } else {
        1 + (u64::from(pixels) - 1 - start) / stride
    }
}

#[test]
fn scalar_fixtures_preserve_masks_exact_integer_reductions_and_dfm_denominator() {
    assert_eq!(reduction_blocks(0), 0);
    assert_eq!(reduction_blocks(1), 1);
    assert_eq!(reduction_blocks(512 * 512), 1024);
    assert_eq!(reduction_blocks(u32::MAX), 1024);

    let zero = [0.0; 4];
    let one = [1.0; 4];
    let sparse = [0.0, 1.0, 0.0, 1.0];
    let threshold = [0.199_999, 0.2, 0.200_001, 0.0];
    assert_eq!(selected_count(&zero, 0.2), 0);
    assert_eq!(selected_count(&one, 0.2), 4);
    assert_eq!(selected_count(&sparse, 0.2), 2);
    assert_eq!(selected_count(&threshold, 0.2), 2);

    assert_eq!(semantic_count(&[0, 4, 5, 11, 12, 13, 18], 0), 2);
    assert_eq!(semantic_count(&[0, 4, 5, 11, 12, 13, 18], 1), 3);
    assert_eq!(gray_sum(&[[0.0, 0.0, 0.0], [255.0, 255.0, 255.0]]), 254);
    assert_eq!(gray_sum(&[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]), 66);

    let values = [10.0, 20.0, 30.0, 40.0];
    let mask = [1.0, 0.0, 1.0, 0.0];
    assert_eq!(
        dfm_mean_with_total_pixel_denominator(&values, &mask, 0.2),
        10.0
    );
}

#[test]
fn reduction_sources_use_two_deterministic_stages_without_contended_outputs() {
    let dfm = fs::read_to_string("gpu_kernels/dfm_color.cu").unwrap();
    let mask = fs::read_to_string("gpu_kernels/mask_postprocess.cu").unwrap();
    let color = fs::read_to_string("gpu_kernels/color_adjust.cu").unwrap();

    for name in [
        "dfm_rct_stats_stage1_kernel",
        "dfm_rct_stats_stage2_kernel",
        "auto_color_dfl_stats_stage1_kernel",
        "auto_color_dfl_stats_stage2_kernel",
    ] {
        assert!(dfm.contains(name), "missing {name}");
    }
    for name in [
        "semantic_region_mask_stage1_kernel",
        "semantic_region_mask_stage2_kernel",
        "semantic_region_stats_stage1_kernel",
        "semantic_region_stats_stage2_kernel",
    ] {
        assert!(mask.contains(name), "missing {name}");
    }
    for name in [
        "color_adjust_prep_stage1_kernel",
        "color_adjust_prep_stage2_kernel",
    ] {
        assert!(color.contains(name), "missing {name}");
    }

    for source in [&dfm, &mask, &color] {
        assert!(!source.contains("atomicAdd(&stats"));
        assert!(!source.contains("atomicAdd(gray_sum"));
        assert!(!source.contains("atomicAdd(count"));
        assert!(source.contains("__shfl_down_sync"));
    }
    assert!(dfm.contains("float count = (float)pixels;"));
}

#[test]
fn workspace_and_launches_reuse_partials_without_host_zero_clears() {
    let workspace = fs::read_to_string("src/pipeline/workspace.rs").unwrap();
    let ops = fs::read_to_string("src/gpu/ops.rs").unwrap();

    assert!(workspace.contains("reduction_partials_f32: CudaSlice<f32>"));
    assert!(workspace.contains("reduction_partials_u32: CudaSlice<u32>"));
    assert!(workspace.contains("stream.alloc_zeros::<f32>(1024 * 13)?"));
    assert!(workspace.contains("stream.alloc_zeros::<u32>(1024)?"));
    assert!(ops.contains("const REDUCTION_THREADS: u32 = 256;"));
    assert!(ops.contains("const MAX_REDUCTION_BLOCKS: u32 = 1024;"));
    assert!(ops.contains("fn reduction_stage1_config"));
    assert!(ops.contains("grid_dim: (blocks, 1, 1)"));
    assert!(ops.contains("block_dim: (REDUCTION_THREADS, 1, 1)"));
    assert!(ops.contains("fn reduction_stage2_config"));
    assert!(ops.contains("grid_dim: (1, 1, 1)"));
    assert!(ops.contains("block_dim: (fields, 1, 1)"));

    for method in [
        "pub fn dfm_rct(",
        "pub fn auto_color_dfl(",
        "pub fn adjust_color(",
        "pub fn semantic_region_mask(",
        "pub fn semantic_region_stats(",
    ] {
        assert!(
            !rust_method(&ops, method).contains("memcpy_htod"),
            "{method} must not upload a zero reduction output"
        );
    }
}

#[test]
fn capped_stage_one_grid_stride_covers_all_pixels_and_live_contract_uses_u32_partials() {
    let pixels = 262_145;
    assert_eq!(reduction_blocks(pixels), MAX_REDUCTION_BLOCKS);
    assert!(
        grid_stride_coverage(pixels, MAX_REDUCTION_BLOCKS)
            .into_iter()
            .all(|visits| visits == 1)
    );

    let dfm = fs::read_to_string("gpu_kernels/dfm_color.cu").unwrap();
    let mask = fs::read_to_string("gpu_kernels/mask_postprocess.cu").unwrap();
    let color = fs::read_to_string("gpu_kernels/color_adjust.cu").unwrap();
    for (source, name) in [
        (&dfm, "dfm_rct_stats_stage1_kernel"),
        (&dfm, "auto_color_dfl_stats_stage1_kernel"),
        (&mask, "semantic_region_mask_stage1_kernel"),
        (&mask, "semantic_region_stats_stage1_kernel"),
        (&color, "color_adjust_prep_stage1_kernel"),
    ] {
        let kernel = cuda_kernel(source, name);
        assert!(kernel.contains("gridDim.x"), "{name} must use grid stride");
        assert!(
            kernel.contains("(unsigned long long)blockDim.x * (unsigned long long)gridDim.x"),
            "{name} must advance by its capped-grid stride"
        );
    }

    let live_contract = fs::read_to_string("tests/live_engine_contract.rs").unwrap();
    assert!(live_contract.contains("let mut gray_partials = stream.alloc_zeros::<u32>(1024)?;"));
    assert!(live_contract.contains("&mut gray_partials,"));
}

#[test]
fn grid_stride_uses_non_wrapping_64_bit_indices_at_u32_max() {
    let stride = u64::from(THREADS_PER_BLOCK) * u64::from(MAX_REDUCTION_BLOCKS);
    let iterations = grid_stride_iterations(u32::MAX, 0, stride);
    assert_eq!(iterations, 16_384);
    assert_eq!(iterations * stride, 1u64 << 32);

    let dfm = fs::read_to_string("gpu_kernels/dfm_color.cu").unwrap();
    let mask = fs::read_to_string("gpu_kernels/mask_postprocess.cu").unwrap();
    let color = fs::read_to_string("gpu_kernels/color_adjust.cu").unwrap();
    for (source, name) in [
        (&dfm, "dfm_rct_stats_stage1_kernel"),
        (&dfm, "auto_color_dfl_stats_stage1_kernel"),
        (&mask, "semantic_region_mask_stage1_kernel"),
        (&mask, "semantic_region_stats_stage1_kernel"),
        (&color, "color_adjust_prep_stage1_kernel"),
    ] {
        let kernel = cuda_kernel(source, name);
        assert!(
            kernel.contains("unsigned long long"),
            "{name} must use a 64-bit loop index"
        );
        assert!(
            kernel.contains("(unsigned long long)pixels"),
            "{name} must compare the 64-bit index to a widened pixel count"
        );
        assert!(
            kernel.contains("(unsigned long long)blockDim.x * (unsigned long long)gridDim.x"),
            "{name} must use a 64-bit stride"
        );
    }
}
