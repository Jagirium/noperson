use noperson::pipeline::face_landmark::{LandmarkError, LandmarkModel};

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
