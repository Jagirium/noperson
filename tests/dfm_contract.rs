use noperson::pipeline::dfm::{DfmContract, DfmTensor, lab_to_rgb, rct_reference, rgb_to_lab};

fn tensor(name: &str, shape: &[i64]) -> DfmTensor {
    DfmTensor {
        name: name.to_owned(),
        shape: shape.to_vec(),
    }
}

#[test]
fn dfm_contract_accepts_plain_and_amp_models() {
    let outputs = [
        tensor("out_face_mask:0", &[-1, 320, 320, 1]),
        tensor("out_celeb_face:0", &[-1, 320, 320, 3]),
        tensor("out_celeb_face_mask:0", &[-1, 320, 320, 1]),
    ];
    let plain = DfmContract::from_io(&[tensor("in_face:0", &[-1, 320, 320, 3])], &outputs).unwrap();
    assert_eq!(plain.input_size(), (320, 320));
    assert!(!plain.has_morph_value());

    let amp = DfmContract::from_io(
        &[
            tensor("in_face:0", &[-1, 320, 320, 3]),
            tensor("morph_value:0", &[1]),
        ],
        &outputs,
    )
    .unwrap();
    assert!(amp.has_morph_value());
}

#[test]
fn dfm_contract_rejects_wrong_layout_or_outputs() {
    let outputs = [
        tensor("out_face_mask:0", &[-1, 320, 320, 1]),
        tensor("out_celeb_face:0", &[-1, 320, 320, 3]),
        tensor("out_celeb_face_mask:0", &[-1, 320, 320, 1]),
    ];
    assert!(DfmContract::from_io(&[tensor("input", &[-1, 3, 320, 320])], &outputs).is_err());
    assert!(
        DfmContract::from_io(&[tensor("in_face:0", &[-1, 320, 320, 3])], &outputs[..2]).is_err()
    );
}

#[test]
fn dynamic_dfm_shapes_resolve_to_single_batch() {
    assert_eq!(
        DfmContract::convert_shape(&[-1, 320, 320, 3]),
        [1, 320, 320, 3]
    );
}

#[test]
#[allow(clippy::excessive_precision)] // Values captured verbatim from the Kornia oracle.
fn dfm_lab_conversion_matches_kornia_oracle() {
    let rgb = [0.1, 0.2, 0.3];
    let lab = rgb_to_lab(rgb);
    let expected = [20.47616577, -0.6531924, -18.63012314];
    for channel in 0..3 {
        assert!((lab[channel] - expected[channel]).abs() < 2e-4);
    }
    let roundtrip = lab_to_rgb(lab);
    for channel in 0..3 {
        assert!((roundtrip[channel] - rgb[channel]).abs() < 2e-5);
    }
}

#[test]
fn dfm_rct_statistics_match_crossswap_oracle() {
    let source = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.2, 0.1],
        [0.9, 0.8, 0.2],
    ];
    let like = [
        [0.2, 0.3, 0.4],
        [0.3, 0.5, 0.7],
        [0.6, 0.4, 0.2],
        [0.8, 0.7, 0.3],
    ];
    let actual = rct_reference(&source, &like, &[1.0, 0.0, 1.0, 1.0], 0.3);
    let expected = [
        [0.1548692, 0.23536989, 0.32268232],
        [0.42197287, 0.49864477, 0.5868289],
        [0.5433387, 0.35314816, 0.17773956],
        [0.8627594, 0.7557435, 0.33217335],
    ];
    for pixel in 0..4 {
        for channel in 0..3 {
            assert!((actual[pixel][channel] - expected[pixel][channel]).abs() < 4e-4);
        }
    }
}
