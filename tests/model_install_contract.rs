use std::fs;

#[test]
fn startup_accepts_a_models_directory_and_installs_the_missing_live_set() {
    let main = fs::read_to_string("src/main.rs").expect("binary entrypoint exists");
    let installer = fs::read_to_string("src/models/install.rs").unwrap_or_default();

    assert!(main.contains("LaunchOptions::parse"));
    assert!(main.contains("options.models_dir"));
    assert!(main.contains("NOPERSON_MODELS_DIR"));
    assert!(main.contains("ensure_required_models"));
    for required in [
        "YoloFace8n",
        "Inswapper128ArcFace",
        "Inswapper128",
        "InswapperEMap",
    ] {
        assert!(
            installer.contains(required),
            "missing automatic model: {required}"
        );
    }
    assert!(installer.contains("ProgressBar"));
}
