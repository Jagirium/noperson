use noperson::extra_gui::{ControlState, ControlValue, FrontendMode, control_catalog};

#[test]
fn control_state_starts_from_catalog_defaults() {
    let catalog = control_catalog().expect("catalog");
    let state = ControlState::from_catalog(&catalog).expect("default state");

    assert_eq!(state.len(), 141);
    assert_eq!(
        state.get("ProvidersPrioritySelection"),
        Some(&ControlValue::Choice("CUDA".to_owned()))
    );
    assert_eq!(
        state.get("SimilarityThresholdSlider"),
        Some(&ControlValue::Slider(60.0))
    );
}

#[test]
fn visibility_tracks_any_all_and_frontend_dependencies() {
    let catalog = control_catalog().expect("catalog");
    let mut state = ControlState::from_catalog(&catalog).expect("default state");

    assert!(!state.is_visible("DFMAmpMorphSlider", FrontendMode::Editor, &catalog));
    state
        .set(
            "SwapModelSelection",
            ControlValue::Choice("DeepFaceLive (DFM)".to_owned()),
            &catalog,
        )
        .expect("DFM selection");
    assert!(state.is_visible("DFMAmpMorphSlider", FrontendMode::Editor, &catalog));

    assert!(!state.is_visible("OccluderXSegBlurSlider", FrontendMode::Editor, &catalog));
    state
        .set("DFLXSegEnableToggle", ControlValue::Toggle(true), &catalog)
        .expect("xseg toggle");
    assert!(state.is_visible("OccluderXSegBlurSlider", FrontendMode::Editor, &catalog));

    state
        .set(
            "FaceParserEnableToggle",
            ControlValue::Toggle(true),
            &catalog,
        )
        .expect("parser toggle");
    assert!(!state.is_visible(
        "FaceParserHairMakeupRedSlider",
        FrontendMode::Editor,
        &catalog
    ));
    state
        .set(
            "FaceParserHairMakeupEnableToggle",
            ControlValue::Toggle(true),
            &catalog,
        )
        .expect("hair makeup toggle");
    assert!(state.is_visible(
        "FaceParserHairMakeupRedSlider",
        FrontendMode::Editor,
        &catalog
    ));

    assert!(!state.is_visible("WebcamCameraSelection", FrontendMode::Editor, &catalog));
    assert!(state.is_visible("WebcamCameraSelection", FrontendMode::Realtime, &catalog));
    assert!(state.is_visible("FrameEnhancerEnableToggle", FrontendMode::Editor, &catalog));
    assert!(!state.is_visible(
        "FrameEnhancerEnableToggle",
        FrontendMode::Realtime,
        &catalog
    ));
}

#[test]
fn invalid_types_ranges_and_choices_are_rejected() {
    let catalog = control_catalog().expect("catalog");
    let mut state = ControlState::from_catalog(&catalog).expect("default state");

    assert!(
        state
            .set("DetectorScoreSlider", ControlValue::Slider(101.0), &catalog,)
            .is_err()
    );
    assert!(
        state
            .set(
                "ProvidersPrioritySelection",
                ControlValue::Choice("CPU".to_owned()),
                &catalog,
            )
            .is_err()
    );
    assert!(
        state
            .set(
                "AutoSwapToggle",
                ControlValue::Choice("true".to_owned()),
                &catalog,
            )
            .is_err()
    );
}
