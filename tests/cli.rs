use std::process::Command;

#[test]
fn help_prints_usage_without_bootstrapping_the_gpu_runtime() {
    let output = Command::new(env!("CARGO_BIN_EXE_noperson"))
        .arg("--help")
        .output()
        .expect("noperson binary runs");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert_eq!(
        stdout,
        concat!(
            "noperson — GPU-accelerated face swap\n",
            "\n",
            "Usage: noperson [OPTIONS]\n",
            "\n",
            "Commands:\n",
            "  headless-run              Process an image or video without a GUI\n",
            "\n",
            "Options:\n",
            "      --realtime            Launch the realtime UI (default)\n",
            "      --extra-gui           Launch the advanced editor UI\n",
            "      --runtime-check       Validate the CUDA and TensorRT runtime\n",
            "      --models-dir <PATH>   Store and load models from PATH\n",
            "  -h, --help                Print help\n",
            "\n",
            "Environment:\n",
            "  NOPERSON_MODELS_DIR       Default model directory when --models-dir is absent\n",
        )
    );
    assert!(!stderr.contains("starting up"));
    assert!(!stderr.contains("GPU runtime"));
}

#[test]
fn headless_help_prints_without_bootstrapping_the_gpu_runtime() {
    let output = Command::new(env!("CARGO_BIN_EXE_noperson"))
        .args(["headless-run", "--help"])
        .output()
        .expect("noperson binary runs");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stdout.contains("Usage: noperson headless-run [OPTIONS]"));
    assert!(stdout.contains("-s, --source-path <PATH>"));
    assert!(stdout.contains("-t, --target-path <PATH>"));
    assert!(stdout.contains("-o, --output-path <PATH>"));
    assert!(!stderr.contains("starting up"));
    assert!(!stderr.contains("GPU runtime"));
}

#[test]
fn startup_log_is_after_the_runtime_reexec_branch() {
    let main = std::fs::read_to_string("src/main.rs").unwrap();
    let reexec = main.find("BootstrapOutcome::Reexec").unwrap();
    let startup = main.find("starting up").unwrap();

    assert!(
        startup > reexec,
        "the bootstrap parent must not announce a full application startup"
    );
}
