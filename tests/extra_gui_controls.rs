use std::collections::HashSet;

use noperson::extra_gui::{ControlKind, ControlScope, control_catalog};

#[test]
fn catalog_covers_crosswap_controls_without_runtime_exclusions() {
    let catalog = control_catalog().expect("catalog must parse and validate");

    assert_eq!(catalog.len(), 141);
    assert_eq!(
        catalog
            .iter()
            .filter(|control| control.scope == ControlScope::Settings)
            .count(),
        29
    );
    assert_eq!(
        catalog
            .iter()
            .filter(|control| control.scope == ControlScope::Common)
            .count(),
        10
    );
    assert_eq!(
        catalog
            .iter()
            .filter(|control| control.scope == ControlScope::Swapper)
            .count(),
        102
    );

    let ids: HashSet<_> = catalog.iter().map(|control| control.id.as_str()).collect();
    assert_eq!(ids.len(), catalog.len(), "control ids must be unique");
    assert!(!ids.contains("BackgroundRemovalEnableToggle"));

    let providers = catalog
        .iter()
        .find(|control| control.id == "ProvidersPrioritySelection")
        .expect("provider selection");
    assert_eq!(providers.choice_options(), Some(vec!["CUDA", "TensorRT"]));

    let threads = catalog
        .iter()
        .find(|control| control.id == "nThreadsSlider")
        .expect("FFmpeg thread budget");
    assert!(threads.help.contains("FFmpeg"));
    assert!(!threads.help.contains("VRAM"));

    let restorer = catalog
        .iter()
        .find(|control| control.id == "FaceRestorerTypeSelection")
        .expect("restorer selection");
    assert_eq!(
        restorer.choice_options(),
        Some(vec!["GPEN-256", "GPEN-512"])
    );

    let resolution = catalog
        .iter()
        .find(|control| control.id == "SwapperResSelection")
        .expect("resolution selection");
    assert_eq!(
        resolution.choice_options(),
        Some(vec!["128", "256", "384", "512"])
    );
}

#[test]
fn every_control_has_a_sound_default_and_resolvable_dependencies() {
    let catalog = control_catalog().expect("catalog must parse and validate");
    let ids: HashSet<_> = catalog.iter().map(|control| control.id.as_str()).collect();

    for control in &catalog {
        assert!(
            !control.label.trim().is_empty(),
            "{} has no label",
            control.id
        );
        for dependency in &control.visibility.dependencies {
            assert!(
                ids.contains(dependency.control.as_str()),
                "{} depends on missing {}",
                control.id,
                dependency.control
            );
        }
        match &control.kind {
            ControlKind::Slider {
                min,
                max,
                default,
                step,
            } => {
                assert!(min <= default && default <= max, "{} default", control.id);
                assert!(*step > 0.0, "{} step", control.id);
            }
            ControlKind::Choice {
                options, default, ..
            } if !options.is_empty() => {
                assert!(options.contains(default), "{} default option", control.id);
            }
            ControlKind::Toggle { .. } | ControlKind::Choice { .. } => {}
        }
    }
}
