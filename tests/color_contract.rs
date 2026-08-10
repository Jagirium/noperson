use noperson::{
    config::parameters::ColorAdjustParams,
    pipeline::color::{
        adjust_color_reference, dfl_transfer_reference, histogram_transfer_reference,
        jpeg_roundtrip_reference,
    },
};
use std::fs;

#[test]
fn color_prep_reduces_contrast_mean_without_a_second_full_image_pass() {
    let kernel = fs::read_to_string("gpu_kernels/color_adjust.cu").unwrap();
    let ops = fs::read_to_string("src/gpu/ops.rs").unwrap();

    let prep = kernel
        .split("void color_adjust_prep_stage1_kernel")
        .nth(1)
        .expect("prep kernel exists")
        .split("extern \"C\" __global__")
        .next()
        .unwrap();
    assert!(prep.contains("__shfl_down_sync"));
    assert!(!prep.contains("atomicAdd("));
    assert!(kernel.contains("void color_adjust_prep_stage2_kernel"));
    assert!(!kernel.contains("void color_contrast_sum_kernel"));
    assert!(!ops.contains("color_contrast_sum_fn"));
}

#[test]
#[allow(clippy::excessive_precision)] // Values captured verbatim from CrossSwap's Kornia path.
fn dfl_color_transfer_matches_crosswap_oracle() {
    let original = [
        [10.0, 20.0, 30.0],
        [40.0, 60.0, 80.0],
        [90.0, 110.0, 130.0],
        [160.0, 180.0, 220.0],
    ];
    let swapped = [
        [200.0, 190.0, 180.0],
        [140.0, 130.0, 120.0],
        [80.0, 70.0, 60.0],
        [20.0, 30.0, 40.0],
    ];
    let expected = [
        [169.875595, 178.016418, 192.816208],
        [112.162659, 119.175896, 132.397903],
        [55.289467, 60.652451, 71.635025],
        [4.0, 22.258215, 46.207233],
    ];

    let actual = dfl_transfer_reference(&original, &swapped, None, 0.8);
    assert_pixels_close(&actual, &expected, 8e-4);
}

#[test]
#[allow(clippy::excessive_precision)] // Values captured verbatim from CrossSwap's Kornia path.
fn masked_dfl_color_transfer_matches_crossswap_oracle() {
    let original = [
        [10.0, 20.0, 30.0],
        [40.0, 60.0, 80.0],
        [90.0, 110.0, 130.0],
        [160.0, 180.0, 220.0],
    ];
    let swapped = [
        [200.0, 190.0, 180.0],
        [140.0, 130.0, 120.0],
        [80.0, 70.0, 60.0],
        [20.0, 30.0, 40.0],
    ];
    let expected = [
        [111.999992, 126.000008, 140.0],
        [66.919167, 77.579872, 88.137535],
        [24.0, 29.999998, 36.000004],
        [4.0, 24.103857, 93.941940],
    ];

    let actual = dfl_transfer_reference(&original, &swapped, Some(&[1.0, 0.0, 1.0, 0.0]), 0.8);
    assert_pixels_close(&actual, &expected, 8e-4);
}

#[test]
#[allow(clippy::excessive_precision)] // Values captured verbatim from Kornia contrib.
fn global_histogram_transfer_matches_kornia_contrib_oracle() {
    let (original, swapped) = color_fixture();
    let expected = [
        [215.999985, 182.0, 164.0],
        [132.0, 113.99999, 96.0],
        [80.0, 62.000004, 44.0],
        [12.000001, 22.0, 32.0],
    ];
    let actual = histogram_transfer_reference(&original, &swapped, None, 0.8);
    assert_pixels_close(&actual, &expected, 3e-4);
}

#[test]
#[allow(clippy::excessive_precision)] // Values captured verbatim from Kornia contrib.
fn masked_histogram_transfer_matches_crossswap_intent() {
    let (original, swapped) = color_fixture();
    let expected = [
        [215.999985, 182.0, 164.0],
        [140.0, 130.0, 120.0],
        [80.0, 62.0, 44.0],
        [20.0, 30.0, 40.0],
    ];
    let actual =
        histogram_transfer_reference(&original, &swapped, Some(&[1.0, 0.0, 1.0, 0.0]), 0.8);
    assert_pixels_close(&actual, &expected, 3e-4);
}

#[test]
fn manual_color_stack_matches_torchvision_uint8_oracle() {
    let image = vec![
        [10.0, 20.0, 30.0],
        [40.0, 60.0, 80.0],
        [90.0, 110.0, 130.0],
        [160.0, 180.0, 220.0],
        [200.0, 210.0, 230.0],
        [240.0, 250.0, 255.0],
        [30.0, 40.0, 50.0],
        [70.0, 80.0, 90.0],
        [120.0, 130.0, 140.0],
    ];
    let params = ColorAdjustParams {
        enabled: true,
        red: 5.0,
        green: -7.0,
        blue: 12.0,
        brightness: 1.1,
        contrast: 0.8,
        saturation: 1.2,
        sharpness: 1.5,
        hue: 0.1,
        gamma: 0.8,
        noise: 0.0,
    };
    let expected = [
        [33.0, 10.0, 35.0],
        [45.0, 24.0, 52.0],
        [61.0, 38.0, 65.0],
        [82.0, 56.0, 88.0],
        [97.0, 72.0, 97.0],
        [96.0, 73.0, 94.0],
        [39.0, 18.0, 41.0],
        [52.0, 30.0, 53.0],
        [67.0, 44.0, 68.0],
    ];
    let actual = adjust_color_reference(&image, 3, 3, &params);
    assert_pixels_close(&actual, &expected, 0.0);
}

#[test]
fn jpeg_roundtrip_matches_torchvision_libjpeg_oracle() {
    let chw: Vec<u8> = (0..3 * 8 * 8).map(|value| value as u8).collect();
    let rgb: Vec<[u8; 3]> = (0..64)
        .map(|pixel| [chw[pixel], chw[64 + pixel], chw[128 + pixel]])
        .collect();
    let decoded = jpeg_roundtrip_reference(&rgb, 8, 8, 50).unwrap();
    let bytes: Vec<u8> = decoded.iter().flatten().copied().collect();
    assert_eq!(
        blake3::hash(&bytes).to_hex().to_string(),
        "95356540fe783343bd9603fd6a896dfb24c6ac4d6d508e0c12a106390e7f87fa"
    );
}

fn color_fixture() -> ([[f32; 3]; 4], [[f32; 3]; 4]) {
    (
        [
            [10.0, 20.0, 30.0],
            [40.0, 60.0, 80.0],
            [90.0, 110.0, 130.0],
            [160.0, 180.0, 220.0],
        ],
        [
            [200.0, 190.0, 180.0],
            [140.0, 130.0, 120.0],
            [80.0, 70.0, 60.0],
            [20.0, 30.0, 40.0],
        ],
    )
}

fn assert_pixels_close(actual: &[[f32; 3]], expected: &[[f32; 3]], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (pixel, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        for channel in 0..3 {
            assert!(
                (actual[channel] - expected[channel]).abs() <= tolerance,
                "pixel {pixel}, channel {channel}: {} != {}",
                actual[channel],
                expected[channel]
            );
        }
    }
}
