use std::fs;

#[test]
fn latent_projection_normalizes_and_projects_in_one_cuda_launch() {
    let kernel = fs::read_to_string("gpu_kernels/matmul_512.cu").unwrap();
    let ops = fs::read_to_string("src/gpu/ops.rs").unwrap();
    let recognizer = fs::read_to_string("src/pipeline/face_recognizer.rs").unwrap();

    assert!(kernel.contains("void calc_latent_512_kernel"));
    assert!(!kernel.contains("void matmul_512_kernel"));
    assert!(!kernel.contains("void l2_normalize_kernel"));
    assert!(ops.contains("calc_latent_512_fn"));
    assert!(recognizer.contains("gpu.calc_latent_512("));
    assert!(!recognizer.contains("gpu.matmul_512("));
    assert!(!recognizer.contains("gpu.l2_normalize("));
}

#[test]
fn detector_compaction_is_parallel_but_preserves_anchor_order() {
    let kernel = fs::read_to_string("gpu_kernels/detector_decode.cu").unwrap();
    let ops = fs::read_to_string("src/gpu/ops.rs").unwrap();

    assert!(kernel.contains("__ballot_sync"));
    assert!(kernel.contains("__shared__ unsigned int warp_offsets[8]"));
    assert!(kernel.contains("ordered_chunk_offset"));
    assert!(!kernel.contains("prefix[256]"));
    assert!(kernel.contains("for (unsigned int base = 0; base < anchors; base += 256)"));
    assert!(
        ops.matches("block_dim: (256, 1, 1)").count() >= 2,
        "both detector decoders must run one ordered 256-thread block"
    );
}

#[test]
fn release_binary_embeds_every_ptx_module_instead_of_reading_target_dir() {
    let build = fs::read_to_string("build.rs").unwrap();
    let ops = fs::read_to_string("src/gpu/ops.rs").unwrap();

    assert!(build.contains("embedded_ptx.rs"));
    assert!(build.contains("include_str!"));
    assert!(ops.contains("include!(concat!(env!(\"OUT_DIR\"), \"/embedded_ptx.rs\"))"));
    assert!(ops.contains("Ptx::from_src"));
    assert!(!ops.contains("Ptx::from_file"));
}

#[test]
fn dead_gpu_modules_are_not_embedded_or_jitted() {
    let ops = fs::read_to_string("src/gpu/ops.rs").unwrap();
    let resize = fs::read_to_string("gpu_kernels/warp_affine.cu").unwrap();
    let layout = fs::read_to_string("gpu_kernels/layout_convert.cu").unwrap();

    assert!(!std::path::Path::new("gpu_kernels/cosine_sim.cu").exists());
    assert!(!resize.contains("warp_affine_chw_kernel"));
    assert!(!layout.contains("void hwc_to_chw_kernel"));
    assert!(!layout.contains("void chw_to_hwc_kernel"));
    assert!(!ops.contains("warp_affine_fn"));
}

#[test]
fn gaussian_passes_stage_halo_tiles_in_shared_memory() {
    let kernel = fs::read_to_string("gpu_kernels/gaussian_blur.cu").unwrap();
    let ops = fs::read_to_string("src/gpu/ops.rs").unwrap();

    assert!(kernel.matches("extern __shared__ float tile[]").count() >= 4);
    assert!(kernel.contains("const unsigned int tile_width = blockDim.x + ks - 1"));
    assert!(kernel.contains("const unsigned int tile_height = blockDim.y + ks - 1"));
    assert!(
        ops.matches("block_dim: (32, 8, 1)").count() >= 4,
        "mask and CHW horizontal/vertical passes must use coalesced 2D tiles"
    );
}
