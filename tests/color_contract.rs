use noperson::pipeline::color::{dfl_transfer_reference, histogram_transfer_reference};

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
