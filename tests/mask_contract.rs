use noperson::config::parameters::{FaceParserMaskParams, FaceSwapParams};
use noperson::pipeline::face_mask::{
    compose_face_parser_mask, generate_border_mask, postprocess_occluder_mask,
    postprocess_xseg_mask,
};

#[test]
fn live_defaults_use_python_compatible_pixel_borders() {
    let params = FaceSwapParams::default();

    let mask = generate_border_mask(
        128,
        params.border_top,
        params.border_bottom,
        params.border_left,
        params.border_right,
    );

    assert_eq!(params.border_top, 10);
    assert_eq!(params.border_bottom, 10);
    assert_eq!(params.border_left, 10);
    assert_eq!(params.border_right, 10);
    assert_eq!(params.border_blur, 10);
    assert_eq!(mask[0], 0.0);
    assert_eq!(mask[9 * 128 + 64], 0.0);
    assert_eq!(mask[10 * 128 + 64], 1.0);
    assert_eq!(mask[64 * 128 + 64], 1.0);
}

#[test]
fn occluder_and_xseg_postprocessing_match_crosswap() {
    let mut occluder = [-0.5, 0.0, 0.001, 2.0];
    postprocess_occluder_mask(&mut occluder);
    assert_eq!(occluder, [0.0, 0.0, 1.0, 1.0]);

    let mut xseg = [-1.0, 0.09, 0.1, 0.75, 2.0];
    postprocess_xseg_mask(&mut xseg);
    assert_eq!(xseg, [0.0, 0.0, 0.1, 0.75, 1.0]);
}

#[test]
fn learned_mask_controls_use_crossswap_defaults() {
    let params = FaceSwapParams::default();
    assert_eq!(params.occluder_size, 0);
    assert_eq!(params.xseg_size, 0);
    assert_eq!(params.occluder_xseg_blur, 0);
}

#[test]
fn face_parser_attribute_growth_masks_the_selected_class_neighborhood() {
    let mut params = FaceParserMaskParams {
        face: 1,
        face_blur: 0,
        background_blur: 0,
        ..FaceParserMaskParams::default()
    };
    params.background = 0;
    let classes = [
        0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, //
        0, 0, 1, 0, 0, //
        0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0,
    ];

    let mask = compose_face_parser_mask(&classes, 5, 5, &params).unwrap();
    assert_eq!(
        mask,
        [
            1.0, 1.0, 1.0, 1.0, 1.0, //
            1.0, 0.0, 0.0, 0.0, 1.0, //
            1.0, 0.0, 0.0, 0.0, 1.0, //
            1.0, 0.0, 0.0, 0.0, 1.0, //
            1.0, 1.0, 1.0, 1.0, 1.0,
        ]
    );
}

#[test]
fn face_parser_background_growth_and_shrink_follow_crossswap_direction() {
    let classes = [0, 0, 0, 0, 1, 0, 0, 0, 0];
    let grow = FaceParserMaskParams {
        background: 1,
        background_blur: 0,
        face_blur: 0,
        ..FaceParserMaskParams::default()
    };
    assert_eq!(
        compose_face_parser_mask(&classes, 3, 3, &grow).unwrap(),
        [1.0; 9]
    );

    let shrink = FaceParserMaskParams {
        background: -1,
        background_blur: 0,
        face_blur: 0,
        ..FaceParserMaskParams::default()
    };
    assert_eq!(
        compose_face_parser_mask(&classes, 3, 3, &shrink).unwrap(),
        [0.0; 9]
    );
}
