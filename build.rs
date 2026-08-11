//! Build script — validates and embeds precompiled backend kernels.
//!
//! Kernel compilation is an explicit release-maintenance step. Ordinary Cargo
//! builds need a C/C++ toolchain for the native bridges, but never CUDA/nvcc.
//! NPP is loaded after the runtime bootstrap so the executable can start on a
//! machine that only has an NVIDIA driver installed.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const NVIDIA_SOURCE_DIRECTORY: &str = "gpu_kernels/nvidia";
const NVIDIA_FATBIN_DIRECTORY: &str = "gpu_kernels/prebuilt/nvidia/cuda-12.8";
const NVIDIA_FATBIN_MANIFEST: &str = "gpu_kernels/prebuilt/nvidia/cuda-12.8/MANIFEST_BLAKE3.txt";
const NVIDIA_CUDA_RELEASE: &str = "12.8";
const NVIDIA_REQUIRED_SASS: &str = "sm75,sm80,sm86,sm89,sm90,sm100,sm120";
const NVIDIA_PTX_FALLBACK: &str = "compute_75";

fn main() {
    println!("cargo:rustc-check-cfg=cfg(noperson_static_test)");
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR is not set");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        let avformat = pkg_config::Config::new()
            .cargo_metadata(true)
            .probe("libavformat")
            .expect("libavformat development files are required for native video support");
        let avcodec = pkg_config::Config::new()
            .cargo_metadata(true)
            .probe("libavcodec")
            .expect("libavcodec development files are required for native video support");
        let avutil = pkg_config::Config::new()
            .cargo_metadata(true)
            .probe("libavutil")
            .expect("libavutil development files are required for native video support");
        let mut video_bridge = cc::Build::new();
        video_bridge
            .cpp(true)
            .std("c++20")
            .warnings(true)
            .flag_if_supported("-fvisibility=hidden")
            .files([
                "native/video/av_demuxer.cpp",
                "native/video/av_muxer.cpp",
                "native/video/nvcodec_shim.cpp",
                "native/video/nvcodec_encoder.cpp",
            ]);
        for include in avformat
            .include_paths
            .iter()
            .chain(&avcodec.include_paths)
            .chain(&avutil.include_paths)
        {
            video_bridge.include(include);
        }
        if let Some(include) = nvcodec_include_path() {
            video_bridge
                .include(include)
                .define("NP_VIDEO_HAS_NV_CODEC_HEADERS", None);
        } else if env::var_os("NOPERSON_REQUIRE_NV_CODEC_HEADERS").is_some() {
            panic!("nv-codec-headers were required but not found; set NOPERSON_NV_CODEC_HEADERS");
        } else {
            println!(
                "cargo:warning=nv-codec-headers not found; NVCodec capability probe is disabled"
            );
        }
        video_bridge.compile("noperson_video");
        println!("cargo:rerun-if-changed=native/video/video_ffi.h");
        println!("cargo:rerun-if-changed=native/video/av_demuxer.cpp");
        println!("cargo:rerun-if-changed=native/video/av_muxer.cpp");
        println!("cargo:rerun-if-changed=native/video/nvcodec_shim.cpp");
        println!("cargo:rerun-if-changed=native/video/nvcodec_encoder.cpp");
        println!("cargo:rerun-if-env-changed=NOPERSON_NV_CODEC_HEADERS");
        println!("cargo:rerun-if-env-changed=NOPERSON_REQUIRE_NV_CODEC_HEADERS");
        println!("cargo:rerun-if-env-changed=NOPERSON_FFMPEG_CACHE_KEY");
        println!(
            "cargo:rerun-if-changed=.cache/dependencies/nv-codec-headers-n13.0.19.0/include/ffnvcodec/nvEncodeAPI.h"
        );

        let object = format!("{out_dir}/jpeg_roundtrip.o");
        let archive = format!("{out_dir}/libnoperson_jpeg_roundtrip.a");
        let cc_status = Command::new("cc")
            .args([
                "-O3",
                "-fPIC",
                "-c",
                "native/jpeg_roundtrip.c",
                "-o",
                &object,
            ])
            .status()
            .expect("failed to compile native/jpeg_roundtrip.c");
        assert!(cc_status.success(), "C compiler failed for JPEG bridge");
        let ar_status = Command::new("ar")
            .args(["rcs", &archive, &object])
            .status()
            .expect("failed to archive JPEG bridge");
        assert!(ar_status.success(), "archiver failed for JPEG bridge");
        let jpeg_archive = Command::new("cc")
            .args(["-print-file-name=libjpeg.a"])
            .output()
            .expect("failed to locate static libjpeg with the C compiler");
        assert!(
            jpeg_archive.status.success(),
            "C compiler could not locate static libjpeg"
        );
        let jpeg_archive = PathBuf::from(
            String::from_utf8(jpeg_archive.stdout)
                .expect("C compiler returned a non-UTF-8 libjpeg path")
                .trim(),
        );
        assert!(
            jpeg_archive.is_absolute() && jpeg_archive.is_file(),
            "static libjpeg archive was not found; install libjpeg development files"
        );
        println!("cargo:rustc-link-search=native={out_dir}");
        println!(
            "cargo:rustc-link-search=native={}",
            jpeg_archive.parent().unwrap().display()
        );
        println!("cargo:rustc-link-lib=static=noperson_jpeg_roundtrip");
        // Keep the bootstrap executable independent from the host JPEG ABI.
        // Linux release images install libjpeg-dev, so only this small archive
        // is copied into the final executable; no JPEG .so is redistributed.
        println!("cargo:rustc-link-lib=static=jpeg");
        println!("cargo:rerun-if-changed=native/jpeg_roundtrip.c");
    }

    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        generate_embedded_fatbin_registry(Path::new(&out_dir));
        println!("cargo:rerun-if-changed={NVIDIA_FATBIN_MANIFEST}");
    } else {
        generate_empty_kernel_registry(Path::new(&out_dir));
    }

    println!("cargo:rerun-if-changed=build.rs");
}

fn generate_empty_kernel_registry(out_dir: &Path) {
    std::fs::write(
        out_dir.join("embedded_fatbin.rs"),
        "fn embedded_fatbin(_name: &str) -> Option<&'static [u8]> { None }\n",
    )
    .expect("failed to generate empty backend kernel registry");
}

fn generate_embedded_fatbin_registry(out_dir: &Path) {
    let manifest = read_blake3_manifest(Path::new(NVIDIA_FATBIN_MANIFEST));
    let mut sources = std::fs::read_dir(NVIDIA_SOURCE_DIRECTORY)
        .unwrap_or_else(|error| panic!("failed to read {NVIDIA_SOURCE_DIRECTORY}: {error}"))
        .map(|entry| entry.expect("invalid GPU kernel entry").path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "cu"))
        .collect::<Vec<_>>();
    sources.sort();
    assert!(!sources.is_empty(), "CUDA source inventory is empty");

    let mut registry = String::from(
        "fn embedded_fatbin(name: &str) -> Option<&'static [u8]> {\n    match name {\n",
    );
    let mut expected_manifest_paths = BTreeSet::new();
    let mut expected_fatbin_paths = BTreeSet::new();
    for source in &sources {
        let stem = source
            .file_stem()
            .and_then(|stem| stem.to_str())
            .expect("CUDA kernel filename is not valid UTF-8");
        let source_relative = format!("{NVIDIA_SOURCE_DIRECTORY}/{stem}.cu");
        let fatbin_relative = format!("{NVIDIA_FATBIN_DIRECTORY}/{stem}.fatbin");
        verify_manifest_file(&manifest, &source_relative);
        verify_manifest_file(&manifest, &fatbin_relative);
        expected_manifest_paths.insert(source_relative.clone());
        expected_manifest_paths.insert(fatbin_relative.clone());
        expected_fatbin_paths.insert(fatbin_relative.clone());
        registry.push_str(&format!(
            "        \"{stem}\" => Some(include_bytes!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/{fatbin_relative}\"))),\n"
        ));
        println!("cargo:rerun-if-changed={source_relative}");
        println!("cargo:rerun-if-changed={fatbin_relative}");
    }
    let manifest_paths = manifest.keys().cloned().collect::<BTreeSet<_>>();
    assert_eq!(
        manifest_paths, expected_manifest_paths,
        "BLAKE3 manifest contains stale or incomplete CUDA inventory"
    );
    let tracked_fatbin_paths = std::fs::read_dir(NVIDIA_FATBIN_DIRECTORY)
        .unwrap_or_else(|error| panic!("failed to read {NVIDIA_FATBIN_DIRECTORY}: {error}"))
        .map(|entry| entry.expect("invalid tracked fatbin entry").path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "fatbin")
        })
        .map(|path| path.to_string_lossy().replace('\\', "/"))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        tracked_fatbin_paths, expected_fatbin_paths,
        "tracked fatbin directory contains stale or incomplete CUDA inventory"
    );
    registry.push_str("        _ => None,\n    }\n}\n");
    std::fs::write(out_dir.join("embedded_fatbin.rs"), registry)
        .expect("failed to generate embedded fatbin registry");
}

fn read_blake3_manifest(path: &Path) -> BTreeMap<String, String> {
    let body = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    let mut inventory = BTreeMap::new();
    let mut metadata = BTreeMap::new();
    for (index, line) in body.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            assert!(
                metadata.insert(key.to_owned(), value.to_owned()).is_none(),
                "duplicate CUDA manifest metadata key: {key}"
            );
            continue;
        }
        let (digest, relative) = line.split_once("  ").unwrap_or_else(|| {
            panic!(
                "invalid BLAKE3 manifest record at {}:{}",
                path.display(),
                index + 1
            )
        });
        assert!(
            digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "invalid BLAKE3 digest at {}:{}",
            path.display(),
            index + 1
        );
        assert!(
            inventory
                .insert(relative.to_owned(), digest.to_owned())
                .is_none(),
            "duplicate BLAKE3 manifest path: {relative}"
        );
    }
    for (key, expected) in [
        ("cuda_release", NVIDIA_CUDA_RELEASE),
        ("sass", NVIDIA_REQUIRED_SASS),
        ("ptx", NVIDIA_PTX_FALLBACK),
    ] {
        assert_eq!(
            metadata.get(key).map(String::as_str),
            Some(expected),
            "CUDA manifest metadata {key} does not match the compiled kernel contract"
        );
    }
    let nvcc_version = metadata
        .get("nvcc_version")
        .expect("CUDA manifest does not record nvcc_version");
    assert!(
        nvcc_version.starts_with(NVIDIA_CUDA_RELEASE),
        "CUDA manifest nvcc_version does not match release {NVIDIA_CUDA_RELEASE}"
    );
    inventory
}

fn verify_manifest_file(manifest: &BTreeMap<String, String>, relative: &str) {
    let expected = manifest
        .get(relative)
        .unwrap_or_else(|| panic!("missing BLAKE3 manifest record: {relative}"));
    let bytes = std::fs::read(relative)
        .unwrap_or_else(|error| panic!("failed to read {relative}: {error}"));
    let actual = blake3::hash(&bytes).to_hex().to_string();
    assert_eq!(
        actual, *expected,
        "BLAKE3 mismatch for {relative}; expected {expected}, actual {actual}"
    );
}

fn nvcodec_include_path() -> Option<PathBuf> {
    let configured = env::var_os("NOPERSON_NV_CODEC_HEADERS").map(PathBuf::from);
    let project_cache = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default())
        .join(".cache/dependencies/nv-codec-headers-n13.0.19.0/include");
    let candidates = configured.into_iter().chain([
        project_cache,
        PathBuf::from("/usr/include"),
        PathBuf::from("/usr/local/include"),
    ]);
    for candidate in candidates {
        if candidate.join("ffnvcodec/nvEncodeAPI.h").is_file() {
            return Some(candidate);
        }
        let include = candidate.join("include");
        if include.join("ffnvcodec/nvEncodeAPI.h").is_file() {
            return Some(include);
        }
    }
    None
}
