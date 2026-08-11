#[path = "support/morphology.rs"]
mod morphology;

use std::fs;

use morphology::{AMOUNTS, FIXTURES, repeated_clamped_max, scalar_reference};

fn morphology_source(ops: &str) -> &str {
    let start = ops.find("    pub fn morphology_mask(").unwrap();
    let end = ops[start..]
        .find("    // ── Allocation helpers")
        .map(|offset| start + offset)
        .unwrap();
    &ops[start..end]
}

fn assert_exactly_two_launches(morphology: &str) {
    assert_eq!(
        morphology.matches(".launch(").count(),
        2,
        "morphology_mask must launch exactly one horizontal and one vertical kernel"
    );
}

#[test]
fn scalar_oracle_matches_literal_clamped_reference_fixtures() {
    let expected = [
        // 1x1
        [
            &[0.25][..],
            &[0.25][..],
            &[0.25][..],
            &[0.25][..],
            &[0.25][..],
            &[0.25][..],
            &[0.25][..],
        ],
        // 1xN
        [
            &[0.0, 0.0, 0.0, 0.0][..],
            &[0.0, 0.0, 0.0, 0.0][..],
            &[0.0, 0.0, 0.0, 0.0][..],
            &[0.0, 1.0, 0.0, 0.0][..],
            &[1.0, 1.0, 1.0, 0.0][..],
            &[1.0, 1.0, 1.0, 1.0][..],
            &[1.0, 1.0, 1.0, 1.0][..],
        ],
        // 2x2
        [
            &[0.0, 0.0, 0.0, 0.0][..],
            &[0.0, 0.0, 0.0, 0.0][..],
            &[0.0, 0.0, 0.0, 0.0][..],
            &[0.0, 1.0, 0.5, 0.25][..],
            &[1.0, 1.0, 1.0, 1.0][..],
            &[1.0, 1.0, 1.0, 1.0][..],
            &[1.0, 1.0, 1.0, 1.0][..],
        ],
        // edge impulse
        [
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][..],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][..],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][..],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][..],
            &[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0][..],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0][..],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0][..],
        ],
        // center impulse
        [
            &[0.0; 25][..],
            &[0.0; 25][..],
            &[0.0; 25][..],
            &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ][..],
            &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ][..],
            &[1.0; 25][..],
            &[1.0; 25][..],
        ],
        // monotonic gradient
        [
            &[0.0; 12][..],
            &[0.0; 12][..],
            &[
                0.0,
                0.0,
                0.100000024,
                0.19999999,
                0.0,
                0.0,
                0.100000024,
                0.19999999,
                0.39999998,
                0.39999998,
                0.5,
                0.6,
            ][..],
            &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0][..],
            &[0.5, 0.6, 0.7, 0.7, 0.9, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0][..],
            &[1.0; 12][..],
            &[1.0; 12][..],
        ],
    ];

    for (fixture, expected) in FIXTURES.iter().zip(expected) {
        for (amount, output) in AMOUNTS.into_iter().zip(expected) {
            let actual = scalar_reference(fixture.input, fixture.width, fixture.height, amount);
            assert_eq!(actual, output, "{} at amount {amount}", fixture.name);
            if amount > 0 {
                assert_eq!(
                    actual,
                    repeated_clamped_max(
                        fixture.input,
                        fixture.width,
                        fixture.height,
                        amount as usize
                    ),
                    "{} at positive amount {amount}",
                    fixture.name,
                );
            }
        }
    }
}

#[test]
fn morphology_uses_two_dynamic_shared_memory_custom_launches() {
    let kernels = fs::read_to_string("gpu_kernels/nvidia/mask_postprocess.cu").unwrap();
    let ops = fs::read_to_string("src/backend/cuda/ops.rs").unwrap();
    let morphology = morphology_source(&ops);

    assert!(kernels.contains("void morphology_mask_horizontal_kernel"));
    assert!(kernels.contains("void morphology_mask_vertical_kernel"));
    assert!(kernels.matches("extern __shared__ float tile[]").count() >= 2);
    assert!(kernels.contains("blockDim.x + 2 * radius"));
    assert!(kernels.contains("blockDim.y + 2 * radius"));
    assert!(kernels.contains("tile[(threadIdx.y + offset) * blockDim.x + threadIdx.x]"));
    assert!(!kernels.contains("threadIdx.y + radius + offset"));
    assert!(ops.contains("morphology_mask_horizontal_fn"));
    assert!(ops.contains("morphology_mask_vertical_fn"));
    assert!(morphology.contains("block_dim: (32, 8, 1)"));
    assert!(morphology.matches("shared_mem_bytes:").count() == 2);
    assert_exactly_two_launches(morphology);
    assert!(morphology.contains("if amount == 0"));
    assert!(morphology.contains("amount.unsigned_abs().min(100)"));
}

#[test]
fn morphology_has_no_npp_or_repeated_pass_or_parity_copy_path() {
    let ops = fs::read_to_string("src/backend/cuda/ops.rs").unwrap();
    let npp = fs::read_to_string("src/backend/cuda/npp.rs").unwrap();
    let morphology = morphology_source(&ops);

    assert!(!morphology.contains("npp::dilate"));
    assert!(!morphology.contains("npp::erode"));
    assert!(!morphology.contains("memcpy_dtod"));
    assert!(!morphology.contains("for "));
    assert!(!npp.contains("dilate_3x3_f32_c1"));
    assert!(!npp.contains("erode_3x3_f32_c1"));
    assert!(!npp.contains("nppiDilate3x3_32f_C1R"));
    assert!(!npp.contains("nppiErode3x3_32f_C1R"));
}
