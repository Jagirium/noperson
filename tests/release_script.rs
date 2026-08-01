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
        !script.contains("-exec install -m 0755 {} \"$stage/lib/\"")
            && !script.contains("LD_LIBRARY_PATH"),
        "the code archive must stay independent from downloadable GPU runtimes"
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
    assert!(
        !script.contains("onnxruntime*.dll"),
        "the Windows code archive must not duplicate downloadable providers"
    );
}

#[test]
fn cargo_and_runtime_logging_pin_the_compatible_ort_contract() {
    let cargo = fs::read_to_string("Cargo.toml").expect("Cargo manifest exists");
    assert!(cargo.contains("version = \"=2.0.0-rc.12\""));

    for profile in ["[profile.dev]", "[profile.test]"] {
        let start = cargo.find(profile).expect("compact local profile exists");
        let body = &cargo[start..];
        assert!(body.contains("debug = 0"));
        assert!(body.contains("incremental = false"));
        assert!(body.contains("codegen-units = 2"));
    }
    let cargo_config =
        fs::read_to_string(".cargo/config.toml").expect("project Cargo config exists");
    assert!(
        cargo_config.contains("jobs = 2"),
        "local builds must not recreate the linker process storm"
    );

    let main = fs::read_to_string("src/main.rs").expect("binary entrypoint exists");
    assert!(
        main.contains("info,ort=warn"),
        "default tracing must suppress ORT info/debug noise"
    );
    assert!(
        main.contains("--runtime-check"),
        "release validation needs a non-GUI bootstrap smoke path"
    );

    for source in ["src/models/manager.rs", "src/models/live_catalog.rs"] {
        let source = fs::read_to_string(source).unwrap();
        assert!(
            source.contains("with_log_level(ort::logging::LogLevel::Warning)"),
            "every production ORT session must use warning severity"
        );
    }

    let manager = fs::read_to_string("src/models/manager.rs").unwrap();
    assert!(
        manager.matches("error_on_failure()").count() >= 3,
        "GPU-only sessions must never fall back silently to CPU"
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
fn native_kernel_build_has_explicit_arch_and_no_downloadable_runtime_links() {
    let build = fs::read_to_string("build.rs").unwrap();
    assert!(build.contains("NOPERSON_CUDA_ARCH"));
    assert!(build.contains("nvcc.exe"));
    assert!(!build.contains("--use_fast_math"));
    assert!(
        !build.contains("rustc-link-lib=dylib=npp")
            && !build.contains("rustc-link-lib=dylib=cudart"),
        "the executable must reach main before the downloadable CUDA runtime exists"
    );

    let cargo = fs::read_to_string("Cargo.toml").unwrap();
    assert!(
        cargo.contains("libloading"),
        "NPP must be resolved after the runtime bootstrap"
    );

    let bootstrap = fs::read_to_string("src/runtime/bootstrap.rs").unwrap();
    assert!(bootstrap.contains("stage_launch_directory"));
    assert!(bootstrap.contains("materialize_atomically"));
    assert!(
        !bootstrap.contains("preload_ort_providers"),
        "ORT must load providers itself after its host is initialized"
    );
}

#[test]
fn linux_runtime_collector_splits_common_libraries_from_tensorrt_arch_resources() {
    let script = fs::read_to_string("scripts/runtime/collect-linux-libs.sh")
        .expect("Linux runtime collector exists");

    for required in [
        "$stage/base",
        "$stage/trt/base",
        "libonnxruntime_providers_cuda.so",
        "libonnxruntime_providers_tensorrt.so",
        "libcublasLt.so.12",
        "libcudnn.so.9",
        "libnppif.so.12",
        "libnvinfer.so.10",
        "libnvonnxparser.so.10",
        "for architecture in sm75 sm80 sm86 sm89 sm90 sm100 sm120 ptx",
        "libnvinfer_builder_resource_${architecture}.so.*",
        "b3sum",
        "readelf",
    ] {
        assert!(
            script.contains(required),
            "missing runtime contract: {required}"
        );
    }

    assert!(
        !script.contains("builder_resource_win_"),
        "Linux runtime must not package Windows builder resources"
    );
    assert!(
        !script.contains("libcuda.so"),
        "the NVIDIA driver library must never be redistributed"
    );
    assert!(
        script.find("RUNTIME-MANIFEST").unwrap() < script.find("BLAKE3SUMS").unwrap(),
        "the runtime metadata must be written before the BLAKE3 inventory"
    );
}

#[test]
fn linux_runtime_verifier_checks_hashes_links_and_loader_closure() {
    let script = fs::read_to_string("scripts/runtime/verify-linux-libs.sh")
        .expect("Linux runtime verifier exists");
    for required in [
        "BLAKE3SUMS",
        "RUNTIME-MANIFEST",
        "ldd",
        "not found",
        "libonnxruntime_providers_cuda.so",
        "libonnxruntime_providers_tensorrt.so",
        "find \"$root\" -xtype l",
        "-path \"$root/launch\" -prune",
    ] {
        assert!(
            script.contains(required),
            "missing verifier contract: {required}"
        );
    }
}

#[test]
fn linux_runtime_packer_emits_base_and_independent_sm_archives() {
    let script = fs::read_to_string("scripts/runtime/pack-linux-libs.sh")
        .expect("Linux runtime packer exists");
    for required in [
        "noperson-runtime-base-linux-x86_64-v1.tar.zst",
        "noperson-runtime-trt-base-linux-x86_64-v1.tar.zst",
        "for shard in sm75 sm80 sm86 sm89 sm90 sm100 sm120 ptx",
        "zstd -q -T2",
        "--sort=name",
        "ARCHIVES-BLAKE3",
        "stream.read(8 * 1024 * 1024)",
    ] {
        assert!(
            script.contains(required),
            "missing packer contract: {required}"
        );
    }
}
