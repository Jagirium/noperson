use noperson::pipeline::dfm::{DfmContract, DfmTensor};

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
