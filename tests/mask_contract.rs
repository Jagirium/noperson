use noperson::config::parameters::{FaceParserMaskParams, FaceSwapParams};
use noperson::pipeline::face_mask::{
    RestoreEllipse, SemanticRegion, apply_restore_ellipse_mask_reference, compose_face_parser_mask,
    eye_restore_ellipses, fake_diff_mask, generate_border_mask, mouth_restore_ellipse,
    postprocess_occluder_mask, postprocess_xseg_mask, semantic_region_mask,
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

#[test]
fn semantic_restore_regions_use_crossswap_parser_classes() {
    let classes = [0, 4, 5, 6, 11, 12, 13, 17];
    assert_eq!(
        semantic_region_mask(&classes, SemanticRegion::Eyes),
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        semantic_region_mask(&classes, SemanticRegion::Mouth),
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    );
}

#[test]
fn restore_region_defaults_match_crossswap_controls() {
    let params = FaceSwapParams::default();
    assert_eq!(params.restore_eyes_params.blend, 0.5);
    assert_eq!(params.restore_eyes_params.feather, 10);
    assert_eq!(params.restore_eyes_params.size_factor, 3.0);
    assert_eq!(params.restore_mouth_params.blend, 0.5);
    assert_eq!(params.restore_mouth_params.feather, 10);
    assert_eq!(params.restore_mouth_params.size_factor, 0.25);
    assert_eq!(params.restore_eyes_mouth_blur, 0);
}

#[test]
fn landmark_mouth_geometry_matches_crossswap_integer_contract() {
    let points = [
        [100.9, 120.9],
        [300.9, 130.9],
        [200.0, 240.0],
        [150.8, 330.9],
        [250.2, 340.1],
    ];
    let params = FaceSwapParams::default();

    assert_eq!(
        mouth_restore_ellipse(&points, &params.restore_mouth_params),
        RestoreEllipse {
            center_x: 200,
            center_y: 335,
            radius_x: 25,
            radius_y: 25,
        }
    );
}

#[test]
fn landmark_eye_geometry_matches_crosswap_integer_contract() {
    let points = [
        [100.9, 120.9],
        [300.9, 130.9],
        [200.0, 240.0],
        [150.8, 330.9],
        [250.2, 340.1],
    ];
    let params = FaceSwapParams::default();

    assert_eq!(
        eye_restore_ellipses(&points, &params.restore_eyes_params),
        [
            RestoreEllipse {
                center_x: 100,
                center_y: 120,
                radius_x: 66,
                radius_y: 66,
            },
            RestoreEllipse {
                center_x: 300,
                center_y: 130,
                radius_x: 66,
                radius_y: 66,
            },
        ]
    );
}

#[test]
fn fallback_restore_ellipse_uses_crossswap_soft_blend() {
    let mut mask = vec![1.0; 7 * 7];
    apply_restore_ellipse_mask_reference(
        &mut mask,
        7,
        7,
        RestoreEllipse {
            center_x: 3,
            center_y: 3,
            radius_x: 2,
            radius_y: 2,
        },
        0.5,
        2,
    );

    assert_eq!(mask[3 * 7 + 3], 0.5);
    assert_eq!(mask[3 * 7 + 2], 0.75);
    assert_eq!(mask[3 * 7 + 1], 1.0);
    assert_eq!(mask[0], 1.0);
}

#[test]
fn fake_diff_uses_crossswap_per_channel_bimodal_threshold() {
    let swapped = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let original = [0.0, 9.0, 20.0, 30.0, 40.0, 49.0];

    assert_eq!(fake_diff_mask(&swapped, &original, 2, 4), [0.0, 1.0]);
}

#[test]
fn differencing_defaults_match_crossswap_controls() {
    let params = FaceSwapParams::default();
    assert!(!params.differencing_enabled);
    assert_eq!(params.differencing_amount, 4);
    assert_eq!(params.differencing_blur, 5);
}
