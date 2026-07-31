use std::fs;

#[test]
fn linux_release_builder_pins_inputs_and_emits_deterministic_archive() {
    let script = fs::read_to_string("scripts/release/linux.sh").expect("release builder exists");
    for required in [
        "RUST_TOOLCHAIN=1.97.1",
        "docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:",
        "--native",
        "cargo build --locked --release",
        "SOURCE_DATE_EPOCH",
        "--sort=name",
        "--owner=0",
        "--group=0",
        "sha256sum",
        "git diff-index --quiet HEAD --",
        "run --rm --interactive",
        "libssl-dev",
        "output_uid=0",
        "OUTPUT_UID",
        "container did not export archive",
        "container did not export checksum",
    ] {
        assert!(
            script.contains(required),
            "missing reproducibility control: {required}"
        );
    }
    assert!(
        !script.contains("sudo "),
        "builder must not mutate the host"
    );
    assert!(
        !script.contains("models/"),
        "code release must not bundle models"
    );
    assert!(!script.contains("aarch64"), "ARM releases are out of scope");

    assert_eq!(
        script.matches("export ORT_CUDA_VERSION=12").count(),
        2,
        "native and container builds must select the CUDA 12 ORT distribution"
    );
    assert!(
        script.contains("libonnxruntime*.so*"),
        "the archive must include the ORT core and provider libraries"
    );
    assert!(
        script.contains("libcublasLt.so.12") && script.contains("libcudart.so.12"),
        "the builder must reject an ORT provider compiled for another CUDA ABI"
    );
}

#[test]
fn windows_release_builder_is_native_and_locked() {
    let script =
        fs::read_to_string("scripts/release/win.ps1").expect("native Windows builder exists");
    for required in [
        "$RustToolchain = '1.97.1'",
        "build --locked --release",
        "CUDA_PATH",
        "release 12.8",
        "SOURCE_DATE_EPOCH",
        "Compress-Archive",
        "Get-FileHash",
    ] {
        assert!(
            script.contains(required),
            "missing Windows release control: {required}"
        );
    }
    assert!(!script.contains("docker"));
    assert!(!script.contains("podman"));
    assert!(
        fs::read_to_string("scripts/release/win.bat")
            .unwrap()
            .contains("win.ps1")
    );
    assert!(
        script.contains("$env:ORT_CUDA_VERSION = '12'"),
        "Windows must select the CUDA 12 ORT distribution"
    );
}

#[test]
fn cargo_and_runtime_logging_pin_the_compatible_ort_contract() {
    let cargo = fs::read_to_string("Cargo.toml").expect("Cargo manifest exists");
    assert!(cargo.contains("version = \"=2.0.0-rc.12\""));

    let main = fs::read_to_string("src/main.rs").expect("binary entrypoint exists");
    assert!(
        main.contains("info,ort=warn"),
        "default tracing must suppress ORT info/debug noise"
    );

    for source in ["src/models/manager.rs", "src/models/live_catalog.rs"] {
        let source = fs::read_to_string(source).unwrap();
        assert!(
            source.contains("with_log_level(ort::logging::LogLevel::Warning)"),
            "every production ORT session must use warning severity"
        );
    }
}

#[test]
fn release_entrypoint_is_only_an_interactive_three_variant_router() {
    let script = fs::read_to_string("scripts/release.sh").expect("thin release router exists");
    assert!(script.contains("Linux GPU + Docker"));
    assert!(script.contains("Linux GPU native"));
    assert!(script.contains("Windows GPU native"));
    assert!(script.contains("release/linux.sh"));
    assert!(!script.contains("cargo build"));
    assert!(!script.contains("apt-get"));
}

#[test]
fn native_kernel_build_has_explicit_arch_and_windows_cuda_layout() {
    let build = fs::read_to_string("build.rs").unwrap();
    assert!(build.contains("NOPERSON_CUDA_ARCH"));
    assert!(build.contains("lib/x64"));
    assert!(build.contains("nvcc.exe"));
    assert!(!build.contains("--use_fast_math"));
}
