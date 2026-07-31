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
