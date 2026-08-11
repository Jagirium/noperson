use std::fs;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::process::{Command, Output};

#[test]
fn linux_release_builder_pins_inputs_and_emits_deterministic_archive() {
    let script = fs::read_to_string("scripts/release/linux.sh").expect("release builder exists");
    for required in [
        "docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:",
        "--native",
        "cargo build --locked --release",
        "SOURCE_DATE_EPOCH",
        "CARGO_PROFILE_RELEASE_LTO=true",
        "--sort=name",
        "--owner=0",
        "--group=0",
        ".tar.zst",
        "zstd -q -T0 -19",
        "mktemp \"$repo_root/dist/.",
        "mv -f -- \"$archive_tmp\" \"$archive\"",
        "sha256sum",
        "git diff-index --quiet HEAD --",
        "run --rm --interactive",
        "libssl-dev",
        "output_uid=0",
        "OUTPUT_UID",
        "container did not export archive",
        "container did not export checksum",
        "kernel_manifest_blake3=",
        "gpu_kernels/prebuilt/nvidia/cuda-12.8/MANIFEST_BLAKE3.txt",
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
    assert!(
        !script.contains("| gzip -n -9"),
        "Linux code releases must use the same zstd container as runtime packs"
    );

    assert_eq!(
        script.matches("export ORT_CUDA_VERSION=12").count(),
        2,
        "native and container builds must select the CUDA 12 ORT distribution"
    );
    assert_eq!(
        script
            .matches("export CARGO_PROFILE_RELEASE_LTO=true")
            .count(),
        2,
        "packaged releases must retain fat LTO while local release builds stay incremental"
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
fn linux_native_release_reads_cuda_version_from_the_complete_nvcc_output() {
    let script = fs::read_to_string("scripts/release/linux.sh").expect("release builder exists");

    assert!(
        script.contains("nvcc_version=$(\"$nvcc\" --version)")
            && script.contains("case \"$nvcc_version\" in"),
        "CUDA 12.8 detection must inspect the complete nvcc version output"
    );
    assert!(
        !script.contains("case \"$(\"$nvcc\" --version | tail -1)\" in"),
        "the final nvcc line contains the build identifier, not the release field"
    );
}

#[test]
fn linux_release_bundles_a_pinned_minimal_lgpl_ffmpeg_runtime() {
    let script = format!(
        "{}\n{}",
        fs::read_to_string("scripts/release/bootstrap.sh").expect("release bootstrap exists"),
        fs::read_to_string("scripts/release/linux.sh").expect("release builder exists")
    );
    for required in [
        "FFMPEG_VERSION=8.1.2",
        "464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c",
        "NV_CODEC_HEADERS_VERSION=n13.0.19.0",
        "86d15d1a7c0ac73a0eafdfc57bebfeba7da8264595bf531cf4d8db1c22940116",
        "--disable-gpl",
        "--disable-nonfree",
        "--disable-programs",
        "--enable-shared",
        "NOPERSON_REQUIRE_NV_CODEC_HEADERS=1",
        "NOPERSON_NV_CODEC_HEADERS",
        "libavformat.so",
        "libavcodec.so",
        "libavutil.so",
        "patchelf --force-rpath --set-rpath '$ORIGIN/lib'",
        "patchelf --print-rpath",
        "bundled FFmpeg loader closure is incomplete",
        "FFMPEG-SOURCE-OFFER",
        "rebuilding relocated minimal FFmpeg runtime",
        "pc_prefix",
        "NOPERSON_FFMPEG_CACHE_KEY",
        "canonical_prefix=\"/ffmpeg-runtime-v${FFMPEG_RUNTIME_CACHE_VERSION}\"",
        "--prefix=\"$canonical_prefix\"",
        "DESTDIR=\"$root\"",
        "sed -i \"s|^prefix=.*|prefix=$NOPERSON_RELEASE_FFMPEG_PREFIX|\"",
        "sed -i \"s|^libdir=.*|libdir=${NOPERSON_RELEASE_FFMPEG_PREFIX}/lib|\"",
        "sed -i \"s|^includedir=.*|includedir=${NOPERSON_RELEASE_FFMPEG_PREFIX}/include|\"",
    ] {
        assert!(
            script.contains(required),
            "missing native video release control: {required}"
        );
    }
    assert_eq!(
        script.matches(r#"make -s -j"$(nproc)""#).count(),
        2,
        "native and container FFmpeg builds must use every available CPU while hiding chatter"
    );
    assert!(
        !script.contains("export PKG_CONFIG_SYSROOT_DIR="),
        "FFmpeg relocation must not sysroot unrelated host pkg-config dependencies"
    );
}

#[test]
fn linux_release_dev_mode_only_bypasses_worktree_cleanliness() {
    let script = fs::read_to_string("scripts/release/linux.sh").expect("release builder exists");

    assert!(
        script.contains("--dev"),
        "Linux release builder must accept --dev"
    );
    assert!(
        script.contains("if test \"$dev_mode\" != true"),
        "dirty-worktree checks must be conditional on development mode"
    );
    assert!(
        script.contains("git diff-index --quiet HEAD --")
            && script.contains("git status --porcelain --untracked-files=normal"),
        "normal release mode must retain both cleanliness checks"
    );
    assert!(
        script.contains("NOPERSON_REQUIRE_NV_CODEC_HEADERS=1"),
        "development mode must not weaken the NVCodec dependency contract"
    );
    assert!(script.contains("export RUSTFLAGS=\"--remap-path-prefix=${repo_root}=."));
    assert!(
        !script.contains("if test \"$dev_mode\" != true; then\n        export RUSTFLAGS="),
        "development and clean release builds must share one remapped Cargo cache"
    );
    for cache_contract in [
        "NOPERSON_RELEASE_DEPENDENCY_ROOT",
        "NOPERSON_RELEASE_TARGET_DIR",
        "rustup run \"$RUST_TOOLCHAIN\" rustc --version",
    ] {
        assert!(
            script.contains(cache_contract),
            "development release cache is missing: {cache_contract}"
        );
    }
}

#[test]
fn linux_native_release_does_not_poison_the_normal_cargo_target() {
    let script = fs::read_to_string("scripts/release/linux.sh").expect("release builder exists");

    assert!(script.contains("CARGO_TARGET_DIR=\"$release_target_dir\""));
    assert!(script.contains("NOPERSON_RELEASE_DEPENDENCY_ROOT"));
    assert!(script.contains("NOPERSON_RELEASE_TARGET_DIR"));
    assert!(script.contains("$release_target_dir/release/noperson"));
    assert!(script.contains("verify_ort_cuda12 \"$release_target_dir/release\""));
}

#[test]
fn linux_release_uses_the_actual_pinned_nvcodec_archive_directory() {
    let script = fs::read_to_string("scripts/release/linux.sh").expect("release builder exists");

    assert!(
        script.contains("nv-codec-headers-${NV_CODEC_HEADERS_VERSION}/include"),
        "the GitHub tag archive retains the leading n in its extracted directory"
    );
    assert!(
        !script.contains("nv-codec-headers-${NV_CODEC_HEADERS_VERSION#n}/include"),
        "the builder must not point at a directory that the archive does not create"
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
fn windows_release_script_is_powershell_51_compatible() {
    let script = fs::read_to_string("scripts/release/win.ps1").expect("Windows builder exists");

    assert!(
        script.contains("[Environment]::OSVersion.Platform")
            && script.contains("[PlatformID]::Win32NT"),
        "the host check must work under StrictMode on Windows PowerShell 5.1"
    );
    assert!(
        !script.contains("$IsWindows"),
        "$IsWindows is undefined in Windows PowerShell 5.1"
    );
    assert!(
        !script.contains("utf8NoBOM"),
        "utf8NoBOM is not a Windows PowerShell 5.1 Set-Content encoding"
    );
    for required in [
        "[System.Text.UTF8Encoding]::new($false)",
        "[System.IO.File]::WriteAllLines",
        "BUILD-MANIFEST",
    ] {
        assert!(
            script.contains(required),
            "missing BOM-less PS5.1 manifest writer contract: {required}"
        );
    }
}

#[test]
fn windows_release_uses_the_native_batch_entrypoint() {
    let entry = fs::read_to_string("scripts/release.bat").expect("Windows entrypoint exists");
    let bridge =
        fs::read_to_string("scripts/release/win.bat").expect("Windows batch bridge exists");
    let builder = fs::read_to_string("scripts/release/win.ps1").expect("Windows builder exists");

    assert!(
        entry.contains("release\\win.bat"),
        "the public Windows entrypoint must delegate to the batch bridge"
    );
    assert!(
        entry.contains("--windows") && entry.contains("--dev"),
        "the public Windows entrypoint must support direct variant selection and dev mode"
    );
    assert!(
        bridge.contains("-Orchestrated"),
        "the batch bridge must authorize the internal PowerShell packager"
    );
    assert!(
        bridge.contains("--dev") && bridge.contains("-Dev"),
        "the batch bridge must translate the public CLI spelling for PowerShell 5.1"
    );
    assert!(
        builder.contains("scripts\\release.bat") && !builder.contains("scripts/release.sh"),
        "Windows errors must point users to the native batch entrypoint"
    );
}

#[test]
fn windows_release_uses_the_published_ort_build_input_via_an_atomic_local_cache() {
    let script = fs::read_to_string("scripts/release/win.ps1").expect("Windows builder exists");
    for required in [
        "https://huggingface.co/Jagirium/noperson-runtime/resolve/main/windows/ort/onnxruntime-1.24.2-x86_64-pc-windows-msvc-cu12.tar.lzma2",
        "81934635",
        "c507789d21f3988502925b900249efdba1909c7ac4f5459e9efbd1b9a343009c",
        "8a54165e2dfc85e9f6afbdaf154e7c1c74582e6269a2d0ec93b11e1459309555",
        ".cache\\release",
        ".part",
        "--continue-at",
        "Get-FileHash -Algorithm SHA256",
        ".complete",
        ".staging-",
        "Move-Item",
        "$env:ORT_CACHE_DIR",
        "dfbin\\x86_64-pc-windows-msvc\\$OrtSha256",
        "onnxruntime.lib",
    ] {
        assert!(
            script.contains(required),
            "missing Windows ORT cache contract: {required}"
        );
    }
    let ort_cache = script.find("$env:ORT_CACHE_DIR").unwrap();
    let cargo_build = script.find("build --locked --release").unwrap();
    assert!(
        ort_cache < cargo_build,
        "ORT_CACHE_DIR must be set before Cargo"
    );
    assert!(
        !script.contains("ORT_LIB_LOCATION"),
        "user-provided ORT linking skips Pyke's Windows system-link prerequisites"
    );
    assert!(
        !script.contains("81_934_635"),
        "Windows PowerShell 5.1 cannot parse underscore-separated numeric literals"
    );
    assert!(
        script.contains("Get-ChildItem -LiteralPath $Stage -Recurse")
            && script.contains("release archive contains forbidden ORT build input"),
        "the final zip needs an explicit ORT build-input exclusion gate"
    );
}

#[test]
fn windows_ort_extraction_is_pinned_and_repository_local() {
    let script = fs::read_to_string("scripts/release/win.ps1").expect("Windows builder exists");
    let manifest = fs::read_to_string("scripts/release/ort-extract/Cargo.toml")
        .expect("standalone ORT extractor manifest exists");
    let source = fs::read_to_string("scripts/release/ort-extract/src/main.rs")
        .expect("standalone ORT extractor source exists");
    assert!(
        std::path::Path::new("scripts/release/ort-extract/Cargo.lock").is_file(),
        "standalone ORT extractor must carry its own lockfile"
    );

    assert!(manifest.contains("lzma-rust2 = \"=0.15.8\""));
    assert!(manifest.contains("tar = \"=0.4.46\""));
    assert!(source.contains("Lzma2Reader::new"));
    assert!(source.contains("tar::Archive::new"));
    for required in [
        "scripts\\release\\ort-extract\\Cargo.toml",
        "build --locked --release --manifest-path",
        "NOPERSON_RELEASE_TOOL_ROOT",
        "noperson-ort-extract.exe",
    ] {
        assert!(
            script.contains(required),
            "missing pinned extractor contract: {required}"
        );
    }
    assert!(!script.contains("tar.exe"));
    assert!(!script.contains("7z"));
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
    let main = fs::read_to_string("src/main.rs").expect("binary entrypoint exists");
    let launch = fs::read_to_string("src/launch.rs").expect("launch mode parser exists");
    assert!(
        main.contains("info,ort=warn"),
        "default tracing must suppress ORT info/debug noise"
    );
    assert!(
        launch.contains("--runtime-check") && main.contains("LaunchMode::RuntimeCheck"),
        "release validation needs a non-GUI bootstrap smoke path"
    );

    for source in ["src/models/manager.rs", "src/models/live_catalog.rs"] {
        let source = fs::read_to_string(source).unwrap();
        assert!(
            source.contains("with_log_level(ort::logging::LogLevel::Warning)"),
            "every production ORT session must use warning severity"
        );
    }

    let provider = fs::read_to_string("src/backend/cuda/inference.rs").unwrap();
    assert!(
        provider.matches("error_on_failure()").count() >= 3,
        "GPU-only sessions must never fall back silently to CPU"
    );
}

#[test]
fn release_entrypoint_orchestrates_bootstrap_kernels_and_platform_build() {
    let script = fs::read_to_string("scripts/release.sh").expect("release orchestrator exists");
    assert!(script.contains("Linux GPU + Docker"));
    assert!(script.contains("Linux GPU native"));
    assert!(script.contains("Windows GPU native"));
    for stage in [
        "resolve_variant",
        "validate_host",
        "prepare_toolchain",
        "prepare_dependencies",
        "verify_kernel_artifacts",
        "run_platform_builder",
    ] {
        assert!(
            script.contains(stage),
            "missing orchestration stage: {stage}"
        );
    }
    assert!(script.contains("release/bootstrap.sh"));
    assert!(script.contains("kernels/build-fatbins.sh\" --check"));
    assert!(script.contains("release/linux.sh"));
    assert!(script.contains("win.bat"));
    assert!(
        script.contains("--dev") && script.contains("dev_args"),
        "the orchestrator must forward development mode to the platform builder"
    );
    assert!(!script.contains("cargo build"));
    assert!(!script.contains("apt-get"));
}

#[test]
fn release_bootstrap_is_pinned_resumable_and_repository_local() {
    let script = fs::read_to_string("scripts/release/bootstrap.sh")
        .expect("shared release bootstrap exists");
    for required in [
        "RUST_TOOLCHAIN=1.97.1",
        "RUSTUP_VERSION=1.29.0",
        "FFMPEG_VERSION=8.1.2",
        "NV_CODEC_HEADERS_VERSION=n13.0.19.0",
        "ZSTD_VERSION=1.5.7",
        "PATCHELF_VERSION=0.18.0",
        "NOPERSON_RELEASE_RUSTUP_HOME",
        "NOPERSON_RELEASE_CARGO_HOME",
        "NOPERSON_RELEASE_DEPENDENCY_ROOT",
        "NOPERSON_RELEASE_TARGET_DIR",
        ".cache/release",
        ".part",
        "curl -fL --retry 3 --continue-at -",
        "mv -f --",
        ".complete",
    ] {
        assert!(
            script.contains(required),
            "missing bootstrap contract: {required}"
        );
    }
    assert!(!script.contains("$HOME"));
    assert!(!script.contains("/usr/local/bin"));
}

#[test]
fn platform_packagers_require_the_orchestrator_contract() {
    let linux = fs::read_to_string("scripts/release/linux.sh").unwrap();
    assert!(linux.contains("--orchestrated"));
    assert!(linux.contains("NOPERSON_RELEASE_DEPENDENCY_ROOT"));
    assert!(linux.contains("NOPERSON_RELEASE_TARGET_DIR"));

    let windows = fs::read_to_string("scripts/release/win.ps1").unwrap();
    assert!(windows.contains("$Orchestrated"));
    assert!(windows.contains("NOPERSON_RELEASE_RUSTUP_HOME"));
    assert!(windows.contains("NOPERSON_RELEASE_CARGO_HOME"));
}

#[test]
fn cargo_build_embeds_verified_fatbins_without_invoking_nvcc() {
    let build = fs::read_to_string("build.rs").unwrap();
    assert!(
        build.contains("MANIFEST_BLAKE3.txt") && build.contains("embedded_fatbin"),
        "Cargo builds must verify and embed the tracked fatbin inventory"
    );
    assert!(
        !build.contains("Command::new(&nvcc)"),
        "ordinary Cargo builds must never execute nvcc"
    );
    assert!(
        build.contains(".cache/dependencies") && build.contains("nv-codec-headers-n13.0.19.0"),
        "local builds must discover the ignored repository dependency cache"
    );
    assert!(!build.contains("--use_fast_math"));
    assert!(
        build.contains("rustc-link-lib=static=jpeg")
            && !build.contains("rustc-link-lib=dylib=jpeg"),
        "the bootstrap executable must not need host libjpeg before main"
    );
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
        bootstrap.contains("parent.join(\"lib\")") && bootstrap.contains("bundled.is_dir()"),
        "re-exec must retain access to the bundled FFmpeg runtime"
    );
    assert!(
        !bootstrap.contains("preload_ort_providers"),
        "ORT must load providers itself after its host is initialized"
    );
}

#[test]
fn kernel_generator_emits_every_supported_sm_and_a_portable_ptx_fallback() {
    let script =
        fs::read_to_string("scripts/kernels/build-fatbins.sh").expect("fatbin generator exists");
    assert!(script.contains("--fatbin"));
    for architecture in ["75", "80", "86", "89", "90", "100", "120"] {
        assert!(
            script.contains(&format!(
                "arch=compute_{architecture},code=sm_{architecture}"
            )),
            "missing SASS payload for sm_{architecture}"
        );
    }
    assert!(script.contains("arch=compute_75,code=compute_75"));
    assert!(script.contains("b3sum"));
    assert!(
        script.contains("mv -f --"),
        "artifacts must be replaced atomically"
    );
    assert!(
        !script
            .lines()
            .any(|line| { line.contains("\"$nvcc\"") && line.trim_end().ends_with('&') }),
        "nvcc invocations must stay sequential"
    );
}

#[test]
fn every_cuda_source_has_one_tracked_fatbin_and_manifest_entry() {
    let output_dir = Path::new("gpu_kernels/prebuilt/nvidia/cuda-12.8");
    let manifest = fs::read_to_string(output_dir.join("MANIFEST_BLAKE3.txt"))
        .expect("fatbin BLAKE3 manifest exists");
    let mut sources = fs::read_dir("gpu_kernels/nvidia")
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "cu"))
        .collect::<Vec<_>>();
    sources.sort();

    assert!(!sources.is_empty(), "CUDA source inventory is empty");
    for source in sources {
        let stem = source.file_stem().unwrap().to_str().unwrap();
        let source_path = format!("gpu_kernels/nvidia/{stem}.cu");
        let fatbin_path = format!("gpu_kernels/prebuilt/nvidia/cuda-12.8/{stem}.fatbin");
        assert!(output_dir.join(format!("{stem}.fatbin")).is_file());
        assert_eq!(
            manifest
                .lines()
                .filter(|line| line.ends_with(&source_path))
                .count(),
            1,
            "source manifest entry must be unique: {source_path}"
        );
        assert_eq!(
            manifest
                .lines()
                .filter(|line| line.ends_with(&fatbin_path))
                .count(),
            1,
            "fatbin manifest entry must be unique: {fatbin_path}"
        );
    }
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
        "libnvjpeg.so.12",
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
fn linux_runtime_blake3_failures_are_not_hidden_by_command_substitution() {
    for path in [
        "scripts/runtime/collect-linux-libs.sh",
        "scripts/runtime/verify-linux-libs.sh",
        "scripts/runtime/pack-linux-libs.sh",
    ] {
        let script = fs::read_to_string(path).expect("runtime script exists");
        assert!(
            script.contains("hash=$(hash_file") || script.contains("actual=$(hash_file"),
            "{path} must capture BLAKE3 output before printing it"
        );
        assert!(
            script.contains("|| die \"BLAKE3"),
            "{path} must abort when the hasher fails"
        );
    }
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
fn windows_runtime_collector_keeps_only_the_verified_cuda_and_trt_closure() {
    let script = fs::read_to_string("scripts/runtime/collect-windows-libs.sh")
        .expect("Windows runtime collector exists");
    for required in [
        "x86_64-pc-windows-msvc",
        "8a54165e2dfc85e9f6afbdaf154e7c1c74582e6269a2d0ec93b11e1459309555",
        "onnxruntime_providers_cuda.dll",
        "onnxruntime_providers_tensorrt.dll",
        "cublasLt64_12.dll",
        "cudnn_engines_precompiled64_9.dll",
        "nvrtc64_120_0.dll",
        "nppif64_12.dll",
        "nvinfer_10.dll",
        "nvonnxparser_10.dll",
        "nvinfer_builder_resource_10.dll",
        "WINDOWS-UNIVERSAL-TRT",
        "TRT_MAJOR_ENTERPRISE 10\\r?$",
        "CUDNN_PATCHLEVEL 0$",
        "TRT_PATCH_ENTERPRISE 0\\r?$",
        "TRT_BUILD_ENTERPRISE 35\\r?$",
        "e635c9af06c64e599a781098466e91b51e19fd0f25f9ac12a23ab511aee3dacf",
        "0c2ff93897203d0115b88a010a76d268ed89ff2a2f628fbed30662310394c122",
        "2197a3aa79c23179ad5203e6b594a50d5c00e3afcd22a688727a38bd03a8a06e",
        "0c93a034083746bc0a2ca4e1fca8f9ba014b22ba2ff4f523f0a82fb3058e6f90",
        "5dfd44d256f3c87d7f173d3d5fc7e648a7476545d0e592dfe8b38c0f0fbd6f35",
        "d4716bdcb38a7c86e5da1d3bbfdd77ce759bfb43ef86fb2454a35aa9b3c9f170",
        "9af12af62cb9eddc8abc7566aa75ac1762bc03b5497de2e6149807e1bccaad75",
        "0fa44c9406bf2da0430df3c223d11bec36467e5e801a9eb59be28afc004bbb41",
        "d56bc3423265bc1f8499edb6a6fe19f300ac1861bf0cecf767fbaf060c007318",
        "df0860579a695aea3a6bd0b4213acbd51dc85dff41d715379c15520849268932",
        "4475cee39d6119a17cdd13450f4e9b4370ebb293dc09b713cd608c3112c812bf",
        "platform=windows-x86_64",
        "cuda=12.8.57",
        "cudnn=9.11.0.98",
        "tensorrt=10.13.0.35",
        "BLAKE3SUMS",
        "BLAKE3_PYTHON",
        "objdump",
    ] {
        assert!(
            script.contains(required),
            "missing Windows runtime contract: {required}"
        );
    }
    assert!(!script.contains("zlibwapi.dll"));
    assert!(!script.contains("curand64_10.dll"));
    assert!(!script.contains("nvinfer_plugin_10.dll"));
    assert!(!script.contains("nvcuda.dll\" \"$stage"));
}

#[test]
fn windows_runtime_packer_emits_three_archives_and_a_blake3_manifest() {
    let script = fs::read_to_string("scripts/runtime/pack-windows-libs.sh")
        .expect("Windows runtime packer exists");
    for required in [
        "noperson-runtime-base-windows-x86_64-v1.tar.zst",
        "noperson-runtime-trt-base-windows-x86_64-v1.tar.zst",
        "noperson-runtime-trt-universal-windows-x86_64-v1.tar.zst",
        "nvinfer_builder_resource_10.dll",
        "MANIFEST_BLAKE3.txt",
        "BLAKE3_PYTHON",
        "--sort=name",
        "--owner=0",
        "--group=0",
        "zstd -q -T2 -3",
    ] {
        assert!(
            script.contains(required),
            "missing Windows pack contract: {required}"
        );
    }
}

#[test]
fn windows_runtime_verifier_checks_blake3_pe_closure_and_universal_markers() {
    let script = fs::read_to_string("scripts/runtime/verify-windows-libs.sh")
        .expect("Windows runtime verifier exists");
    for required in [
        "BLAKE3SUMS",
        "BLAKE3_PYTHON",
        "objdump -p",
        "WINDOWS-UNIVERSAL-TRT",
        "nvinfer_builder_resource_10.dll",
        "msvcp140.dll",
        "vcruntime140_1.dll",
        "nvcuda.dll",
        "unexpected file count",
    ] {
        assert!(
            script.contains(required),
            "missing Windows verification: {required}"
        );
    }
}

#[cfg(unix)]
fn write_executable(path: &Path, source: &str) {
    fs::write(path, source).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

#[cfg(unix)]
#[test]
fn verified_bootstrap_download_is_reused_without_a_second_network_call() {
    let fixture = tempfile::tempdir().unwrap();
    let tools = fixture.path().join("tools");
    let source = fixture.path().join("source.bin");
    let output = fixture.path().join("repo/artifact.bin");
    let count = fixture.path().join("curl.count");
    fs::create_dir_all(&tools).unwrap();
    fs::create_dir_all(output.parent().unwrap()).unwrap();
    fs::write(&source, b"bootstrap payload").unwrap();
    write_executable(
        &tools.join("curl"),
        r#"#!/usr/bin/env bash
set -Eeuo pipefail
output=
while test "$#" -gt 0; do
    case "$1" in
        -o) output=$2; shift 2 ;;
        *) shift ;;
    esac
done
cp -- "$FIXTURE_SOURCE" "$output"
printf 'call\n' >> "$FIXTURE_COUNT"
"#,
    );
    let digest_output = Command::new("sha256sum").arg(&source).output().unwrap();
    assert!(digest_output.status.success());
    let digest_line = String::from_utf8(digest_output.stdout).unwrap();
    let digest = digest_line.split_whitespace().next().unwrap();
    let path = format!(
        "{}:{}",
        tools.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let command = r#"
source scripts/release/bootstrap.sh
bootstrap_init "$FIXTURE_REPO"
download_verified https://fixture.invalid/artifact "$FIXTURE_OUTPUT" "$FIXTURE_HASH"
"#;
    for _ in 0..2 {
        let result = Command::new("bash")
            .arg("-c")
            .arg(command)
            .env("PATH", &path)
            .env("FIXTURE_REPO", fixture.path().join("repo"))
            .env("FIXTURE_SOURCE", &source)
            .env("FIXTURE_OUTPUT", &output)
            .env("FIXTURE_HASH", digest)
            .env("FIXTURE_COUNT", &count)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "bootstrap failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    assert_eq!(fs::read_to_string(count).unwrap(), "call\n");
    assert_eq!(fs::read(output).unwrap(), b"bootstrap payload");
}

#[cfg(unix)]
fn write_windows_runtime_fixture(root: &Path) -> Vec<PathBuf> {
    fs::create_dir_all(root.join("base")).unwrap();
    fs::create_dir_all(root.join("trt/base")).unwrap();
    fs::write(root.join("base/provider.dll"), b"provider").unwrap();
    fs::write(
        root.join("trt/base/nvinfer_builder_resource_10.dll"),
        b"builder",
    )
    .unwrap();
    for architecture in [
        "sm75", "sm80", "sm86", "sm89", "sm90", "sm100", "sm120", "ptx",
    ] {
        let directory = root.join("trt").join(architecture);
        fs::create_dir_all(&directory).unwrap();
        fs::write(
            directory.join("WINDOWS-UNIVERSAL-TRT"),
            b"nvinfer_builder_resource_10.dll\n",
        )
        .unwrap();
    }
    let mut files = vec![
        PathBuf::from("base/provider.dll"),
        PathBuf::from("trt/base/nvinfer_builder_resource_10.dll"),
    ];
    files.extend(
        [
            "sm75", "sm80", "sm86", "sm89", "sm90", "sm100", "sm120", "ptx",
        ]
        .into_iter()
        .map(|architecture| PathBuf::from(format!("trt/{architecture}/WINDOWS-UNIVERSAL-TRT"))),
    );
    files
}

#[cfg(unix)]
fn fake_runtime_tools(directory: &Path, objdump_source: &str) {
    fs::create_dir_all(directory).unwrap();
    write_executable(&directory.join("objdump"), objdump_source);
    write_executable(
        &directory.join("b3sum"),
        "#!/usr/bin/env bash\nprintf '%064d\\n' 0\n",
    );
}

#[cfg(unix)]
fn write_fake_inventory(root: &Path, files: &[PathBuf]) {
    let mut inventory = String::new();
    for relative in files {
        inventory.push_str(&format!("{}  {}\n", "0".repeat(64), relative.display()));
    }
    fs::write(root.join("BLAKE3SUMS"), inventory).unwrap();
}

#[cfg(unix)]
fn run_windows_verifier(root: &Path, tools: &Path) -> Output {
    let path = format!(
        "{}:{}",
        tools.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    Command::new("bash")
        .arg("scripts/runtime/verify-windows-libs.sh")
        .arg(root)
        .env("PATH", path)
        .output()
        .unwrap()
}

#[cfg(unix)]
#[test]
fn windows_runtime_verifier_propagates_objdump_failures() {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path().join("runtime");
    let tools = fixture.path().join("tools");
    let files = write_windows_runtime_fixture(&root);
    write_fake_inventory(&root, &files);
    fake_runtime_tools(&tools, "#!/usr/bin/env bash\nexit 42\n");

    let output = run_windows_verifier(&root, &tools);
    assert!(!output.status.success(), "objdump failure was swallowed");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("could not inspect PE dependencies"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(unix)]
#[test]
fn windows_runtime_verifier_rejects_duplicate_and_untracked_inventory_paths() {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path().join("runtime");
    let tools = fixture.path().join("tools");
    let mut files = write_windows_runtime_fixture(&root);
    files.pop();
    files.push(files[0].clone());
    write_fake_inventory(&root, &files);
    fake_runtime_tools(&tools, "#!/usr/bin/env bash\nexit 0\n");

    let output = run_windows_verifier(&root, &tools);
    assert!(
        !output.status.success(),
        "duplicate inventory path was accepted"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("duplicate BLAKE3SUMS path"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(unix)]
#[test]
fn windows_runtime_verifier_accepts_a_trailing_root_separator() {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path().join("runtime");
    let tools = fixture.path().join("tools");
    let files = write_windows_runtime_fixture(&root);
    write_fake_inventory(&root, &files);
    fake_runtime_tools(&tools, "#!/usr/bin/env bash\nexit 0\n");

    let root_with_separator = PathBuf::from(format!("{}/", root.display()));
    let output = run_windows_verifier(&root_with_separator, &tools);
    assert!(
        output.status.success(),
        "trailing separator broke verification: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(unix)]
fn write_windows_pack_fixture(root: &Path, mode: u32) {
    fs::create_dir_all(root.join("base")).unwrap();
    fs::create_dir_all(root.join("trt/base")).unwrap();
    for relative in [
        "base/provider.dll",
        "trt/base/onnxruntime_providers_tensorrt.dll",
        "trt/base/nvinfer_10.dll",
        "trt/base/nvonnxparser_10.dll",
        "trt/base/nvinfer_builder_resource_10.dll",
        "RUNTIME-MANIFEST",
        "BLAKE3SUMS",
    ] {
        fs::write(root.join(relative), relative.as_bytes()).unwrap();
    }
    for architecture in [
        "sm75", "sm80", "sm86", "sm89", "sm90", "sm100", "sm120", "ptx",
    ] {
        let directory = root.join("trt").join(architecture);
        fs::create_dir_all(&directory).unwrap();
        fs::write(directory.join("WINDOWS-UNIVERSAL-TRT"), b"builder\n").unwrap();
    }
    for entry in walkdir(root) {
        fs::set_permissions(&entry, fs::Permissions::from_mode(mode)).unwrap();
    }
}

#[cfg(unix)]
fn walkdir(root: &Path) -> Vec<PathBuf> {
    let mut pending = vec![root.to_path_buf()];
    let mut entries = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path.clone());
            }
            entries.push(path);
        }
    }
    entries
}

#[cfg(unix)]
#[test]
fn windows_runtime_archives_ignore_source_umask_modes() {
    let fixture = tempfile::tempdir().unwrap();
    let tools = fixture.path().join("tools");
    fake_runtime_tools(&tools, "#!/usr/bin/env bash\nexit 0\n");
    let first = fixture.path().join("first");
    let second = fixture.path().join("second");
    write_windows_pack_fixture(&first, 0o755);
    write_windows_pack_fixture(&second, 0o700);
    let path = format!(
        "{}:{}",
        tools.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    for (input, output) in [
        (&first, fixture.path().join("out-first")),
        (&second, fixture.path().join("out-second")),
    ] {
        let result = Command::new("bash")
            .arg("scripts/runtime/pack-windows-libs.sh")
            .arg(input)
            .arg(&output)
            .env("PATH", &path)
            .env("SOURCE_DATE_EPOCH", "1700000000")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "packer failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    for archive in [
        "noperson-runtime-base-windows-x86_64-v1.tar.zst",
        "noperson-runtime-trt-base-windows-x86_64-v1.tar.zst",
        "noperson-runtime-trt-universal-windows-x86_64-v1.tar.zst",
    ] {
        let first_bytes = fs::read(fixture.path().join("out-first").join(archive)).unwrap();
        let second_bytes = fs::read(fixture.path().join("out-second").join(archive)).unwrap();
        assert_eq!(
            blake3::hash(&first_bytes),
            blake3::hash(&second_bytes),
            "archive mode bits depend on source umask: {archive}"
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
        "MANIFEST_BLAKE3.txt",
        "stream.read(8 * 1024 * 1024)",
    ] {
        assert!(
            script.contains(required),
            "missing packer contract: {required}"
        );
    }
}
