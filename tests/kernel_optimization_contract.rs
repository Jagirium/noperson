use std::{collections::BTreeSet, fs, path::Path};

fn build_enforces_blake3(source: &str) -> bool {
    let compact = source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>();
    compact.contains("letactual=blake3::hash(&bytes).to_hex().to_string();")
        && compact.contains("assert_eq!(actual,*expected,")
}

#[test]
fn latent_projection_normalizes_and_projects_in_one_cuda_launch() {
    let kernel = fs::read_to_string("gpu_kernels/nvidia/matmul_512.cu").unwrap();
    let ops = fs::read_to_string("src/backend/cuda/ops.rs").unwrap();
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
    let kernel = fs::read_to_string("gpu_kernels/nvidia/detector_decode.cu").unwrap();
    let ops = fs::read_to_string("src/backend/cuda/ops.rs").unwrap();

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
fn release_binary_embeds_every_fatbin_module_instead_of_reading_target_dir() {
    let build = fs::read_to_string("build.rs").unwrap();
    let compact_build = build
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>();
    let ops = fs::read_to_string("src/backend/cuda/ops.rs").unwrap();
    let generator = fs::read_to_string("scripts/kernels/build-fatbins.sh").unwrap();

    assert!(build.contains("embedded_fatbin.rs"));
    assert!(build.contains("include_bytes!"));
    assert!(build.contains("verify_manifest_file(&manifest, &source_relative)"));
    assert!(build.contains("verify_manifest_file(&manifest, &fatbin_relative)"));
    assert!(build_enforces_blake3(&build));
    assert!(build.contains("expected_manifest_paths"));
    assert!(build.contains("tracked_fatbin_paths"));
    assert!(compact_build.contains("assert_eq!(manifest_paths,expected_manifest_paths"));
    assert!(compact_build.contains("assert_eq!(tracked_fatbin_paths,expected_fatbin_paths"));
    assert!(!build.contains("Command::new(&nvcc)"));
    assert!(!build.contains("Command::new(\"nvcc\")"));
    assert!(ops.contains("include!(concat!(env!(\"OUT_DIR\"), \"/embedded_fatbin.rs\"))"));
    assert!(ops.contains("Ptx::from_binary"));
    assert!(!ops.contains("Ptx::from_file"));

    for target in ["75", "80", "86", "89", "90", "100", "120"] {
        assert!(
            generator.contains(&format!("arch=compute_{target},code=sm_{target}")),
            "missing sm_{target} fatbin target"
        );
    }
    assert!(generator.contains("sm_$architecture"));
    assert!(generator.contains("arch=compute_75,code=compute_75"));
    assert!(generator.contains("sm_75.ptx"));

    let source_paths = fs::read_dir("gpu_kernels/nvidia")
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "cu"))
        .map(|path| path.strip_prefix(".").unwrap_or(&path).to_path_buf())
        .collect::<BTreeSet<_>>();
    let fatbin_paths = fs::read_dir("gpu_kernels/prebuilt/nvidia/cuda-12.8")
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "fatbin")
        })
        .collect::<BTreeSet<_>>();
    let manifest_paths = fs::read_to_string(
        Path::new("gpu_kernels/prebuilt/nvidia/cuda-12.8").join("MANIFEST_BLAKE3.txt"),
    )
    .unwrap()
    .lines()
    .filter_map(|line| {
        line.split_once("  ")
            .map(|(_, path)| Path::new(path).to_path_buf())
    })
    .collect::<BTreeSet<_>>();
    let expected_manifest_paths = source_paths
        .iter()
        .cloned()
        .chain(fatbin_paths.iter().cloned())
        .collect::<BTreeSet<_>>();
    assert_eq!(manifest_paths, expected_manifest_paths);
    assert_eq!(source_paths.len(), fatbin_paths.len());
}

#[test]
fn fatbin_contract_rejects_a_noop_blake3_verifier() {
    let build = fs::read_to_string("build.rs").unwrap();
    let mutated = build.replace("blake3::hash", "disabled_blake3::hash");
    assert!(!build_enforces_blake3(&mutated));
}

#[test]
fn dead_gpu_modules_are_not_embedded_or_jitted() {
    let ops = fs::read_to_string("src/backend/cuda/ops.rs").unwrap();
    let resize = fs::read_to_string("gpu_kernels/nvidia/warp_affine.cu").unwrap();
    let layout = fs::read_to_string("gpu_kernels/nvidia/layout_convert.cu").unwrap();

    assert!(!std::path::Path::new("gpu_kernels/nvidia/cosine_sim.cu").exists());
    assert!(!resize.contains("warp_affine_chw_kernel"));
    assert!(!layout.contains("void hwc_to_chw_kernel"));
    assert!(!layout.contains("void chw_to_hwc_kernel"));
    assert!(!ops.contains("warp_affine_fn"));
}

#[test]
fn gaussian_passes_stage_halo_tiles_in_shared_memory() {
    let kernel = fs::read_to_string("gpu_kernels/nvidia/gaussian_blur.cu").unwrap();
    let ops = fs::read_to_string("src/backend/cuda/ops.rs").unwrap();

    assert!(kernel.matches("extern __shared__ float tile[]").count() >= 4);
    assert!(kernel.contains("const unsigned int tile_width = blockDim.x + ks - 1"));
    assert!(kernel.contains("const unsigned int tile_height = blockDim.y + ks - 1"));
    assert!(
        ops.matches("block_dim: (32, 8, 1)").count() >= 4,
        "mask and CHW horizontal/vertical passes must use coalesced 2D tiles"
    );
}
