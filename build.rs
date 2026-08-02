//! Build script — compiles CUDA kernels (.cu → .ptx).
//!
//! PTX files are loaded at runtime by gpu::ops and gpu::kernels.
//! NPP is loaded after the runtime bootstrap so the executable can start on a
//! machine that only has an NVIDIA driver installed.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let kernels_dir = Path::new("gpu_kernels");
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR is not set");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let cuda_path = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| {
            if target_os == "windows" {
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8".to_owned()
            } else {
                "/usr/local/cuda".to_owned()
            }
        });
    let nvcc = if target_os == "windows" {
        PathBuf::from(&cuda_path).join("bin").join("nvcc.exe")
    } else {
        PathBuf::from(&cuda_path).join("bin").join("nvcc")
    };
    // Keep embedded PTX portable across the complete supported NVIDIA floor.
    // The driver JIT specializes compute_75 PTX for Turing and every newer SM.
    let cuda_arch = env::var("NOPERSON_CUDA_ARCH").unwrap_or_else(|_| "compute_75".to_owned());

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

    // Compile each .cu file to .ptx
    let mut embedded_ptx =
        String::from("fn embedded_ptx(name: &str) -> Option<&'static str> {\n    match name {\n");
    if kernels_dir.exists() {
        let mut paths = std::fs::read_dir(kernels_dir)
            .expect("Failed to read gpu_kernels/")
            .map(|entry| entry.expect("invalid gpu kernel entry").path())
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths {
            if path.extension().is_some_and(|e| e == "cu") {
                let stem = path.file_stem().unwrap().to_str().unwrap();
                let ptx_out = format!("{out_dir}/{stem}.ptx");

                let status = Command::new(&nvcc)
                    .args([
                        "--ptx",
                        &format!("-arch={cuda_arch}"),
                        "-O3",
                        "-o",
                        &ptx_out,
                        path.to_str().unwrap(),
                    ])
                    .status()
                    .unwrap_or_else(|e| panic!("Failed to run nvcc for {stem}.cu: {e}"));

                assert!(status.success(), "nvcc failed for {stem}.cu");
                embedded_ptx.push_str(&format!(
                    "        \"{stem}\" => Some(include_str!(concat!(env!(\"OUT_DIR\"), \"/{stem}.ptx\"))),\n"
                ));
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
    embedded_ptx.push_str("        _ => None,\n    }\n}\n");
    std::fs::write(format!("{out_dir}/embedded_ptx.rs"), embedded_ptx)
        .expect("failed to generate embedded PTX registry");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=gpu_kernels/");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=NOPERSON_CUDA_ARCH");
}

fn nvcodec_include_path() -> Option<PathBuf> {
    let configured = env::var_os("NOPERSON_NV_CODEC_HEADERS").map(PathBuf::from);
    let candidates = configured.into_iter().chain([
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
