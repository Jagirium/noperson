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
