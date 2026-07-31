use noperson::pipeline::face_landmark::{LandmarkError, LandmarkModel, LandmarkResult};

fn points(count: usize) -> Vec<[f32; 2]> {
    (0..count)
        .map(|index| [index as f32, index as f32 + 0.25])
        .collect()
}

#[test]
fn all_crosswap_landmark_modes_are_addressable() {
    for (mode, expected) in [
        ("5", LandmarkModel::Points5),
        ("68", LandmarkModel::Points68),
        ("3d68", LandmarkModel::Points3d68),
        ("98", LandmarkModel::Points98),
        ("106", LandmarkModel::Points106),
        ("203", LandmarkModel::Points203),
        ("478", LandmarkModel::Points478),
    ] {
        assert_eq!(LandmarkModel::from_mode(mode).unwrap(), expected);
    }
    assert!(matches!(
        LandmarkModel::from_mode("unknown"),
        Err(LandmarkError::UnsupportedMode(_))
    ));
}

#[test]
fn landmark_five_point_reductions_match_crosswap_indices() {
    assert_eq!(
        LandmarkModel::Points68.to_five(&points(68)).unwrap(),
        [
            [37.5, 37.75],
            [43.5, 43.75],
            [30.0, 30.25],
            [48.0, 48.25],
            [54.0, 54.25]
        ]
    );
    assert_eq!(
        LandmarkModel::Points3d68.to_five(&points(68)).unwrap(),
        [
            [37.5, 37.75],
            [43.5, 43.75],
            [30.0, 30.25],
            [48.0, 48.25],
            [54.0, 54.25]
        ]
    );
    assert_eq!(
        LandmarkModel::Points98.to_five(&points(98)).unwrap(),
        [
            [96.0, 96.25],
            [97.0, 97.25],
            [54.0, 54.25],
            [76.0, 76.25],
            [82.0, 82.25]
        ]
    );
    assert_eq!(
        LandmarkModel::Points106.to_five(&points(106)).unwrap(),
        [
            [38.0, 38.25],
            [88.0, 88.25],
            [86.0, 86.25],
            [52.0, 52.25],
            [61.0, 61.25]
        ]
    );
    assert_eq!(
        LandmarkModel::Points203.to_five(&points(203)).unwrap(),
        [
            [197.0, 197.25],
            [198.0, 198.25],
            [201.0, 201.25],
            [48.0, 48.25],
            [66.0, 66.25]
        ]
    );
    assert_eq!(
        LandmarkModel::Points478.to_five(&points(478)).unwrap(),
        [
            [468.0, 468.25],
            [473.0, 473.25],
            [4.0, 4.25],
            [61.0, 61.25],
            [291.0, 291.25]
        ]
    );
}

#[test]
fn reduction_rejects_truncated_model_output() {
    assert_eq!(
        LandmarkModel::Points203.to_five(&points(202)),
        Err(LandmarkError::PointCount {
            expected: 203,
            actual: 202,
        })
    );
}

#[test]
fn per_model_normalization_matches_the_python_oracle() {
    assert_eq!(LandmarkModel::Points5.normalize(255.0, 0), 151.0);
    assert_eq!(LandmarkModel::Points5.normalize(128.0, 1), 11.0);
    assert_eq!(LandmarkModel::Points5.normalize(0.0, 2), -123.0);
    assert_eq!(LandmarkModel::Points68.normalize(255.0, 0), 1.0);
    assert_eq!(LandmarkModel::Points203.normalize(127.5, 2), 0.5);
    assert_eq!(LandmarkModel::Points3d68.normalize(255.0, 0), 255.0);
    assert_eq!(LandmarkModel::Points106.normalize(64.0, 2), 64.0);
}

#[test]
fn landmark_onnx_bindings_match_crossswap_models() {
    assert_eq!(LandmarkModel::Points5.input_name(), "input");
    assert_eq!(
        LandmarkModel::Points5.output_specs(),
        [("conf", 21_504), ("landmarks", 107_520)].as_slice()
    );
    assert_eq!(LandmarkModel::Points3d68.input_name(), "data");
    assert_eq!(
        LandmarkModel::Points3d68.output_specs(),
        [("fc1", 3_309)].as_slice()
    );
    assert_eq!(
        LandmarkModel::Points203.output_specs(),
        [("output", 214), ("853", 262), ("856", 406)].as_slice()
    );
    assert_eq!(
        LandmarkModel::Points478.output_specs(),
        [("Identity", 1_434), ("Identity_1", 1), ("Identity_2", 1)].as_slice()
    );
}

#[test]
fn bbox_crop_affines_match_crossswap_geometry() {
    let bbox = [10.0, 20.0, 110.0, 100.0];
    let fan = LandmarkModel::Points68.bbox_affine(bbox).unwrap();
    assert_eq!(fan, [[1.95, 0.0, 11.0], [0.0, 1.95, 11.0]]);

    let dense = LandmarkModel::Points106.bbox_affine(bbox).unwrap();
    assert!((dense[0][0] - 1.28).abs() < 1e-6);
    assert!((dense[0][2] - 19.2).abs() < 1e-5);
    assert!((dense[1][2] - 19.2).abs() < 1e-5);
}

#[test]
fn dense_scores_reduce_before_thresholding() {
    let scores: Vec<f32> = (0..98).map(|index| index as f32).collect();
    assert_eq!(
        LandmarkModel::Points98.to_five_scores(&scores).unwrap(),
        vec![96.0, 97.0, 54.0, 76.0, 82.0]
    );

    let scores: Vec<f32> = (0..68).map(|index| index as f32).collect();
    assert_eq!(
        LandmarkModel::Points68.to_five_scores(&scores).unwrap(),
        vec![37.5, 43.5, 30.0, 48.0, 54.0]
    );
}

#[test]
fn refiner_replaces_detector_only_when_crossswap_would() {
    let mut result = LandmarkResult {
        five: [[0.0; 2]; 5],
        points: Vec::new(),
        scores: vec![0.7, 0.9],
    };
    assert!(result.is_preferred_to(0.79));
    assert!(!result.is_preferred_to(0.8));
    result.scores.clear();
    assert!(result.is_preferred_to(0.99));
}
