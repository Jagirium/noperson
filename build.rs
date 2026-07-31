//! Build script — compiles CUDA kernels (.cu → .ptx) and links NPP.
//!
//! PTX files are loaded at runtime by gpu::ops and gpu::kernels.
//! NPP libraries provide optimized image geometry (warp, resize, blur).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let kernels_dir = Path::new("gpu_kernels");
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
    let cuda_arch = env::var("NOPERSON_CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_owned());

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        let out_dir = env::var("OUT_DIR").unwrap();
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
        println!("cargo:rustc-link-search=native={out_dir}");
        println!("cargo:rustc-link-lib=static=noperson_jpeg_roundtrip");
        println!("cargo:rustc-link-lib=dylib=jpeg");
        println!("cargo:rerun-if-changed=native/jpeg_roundtrip.c");
    }

    // Compile each .cu file to .ptx
    if kernels_dir.exists() {
        let mut paths = std::fs::read_dir(kernels_dir)
            .expect("Failed to read gpu_kernels/")
            .map(|entry| entry.expect("invalid gpu kernel entry").path())
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths {
            if path.extension().is_some_and(|e| e == "cu") {
                let stem = path.file_stem().unwrap().to_str().unwrap();
                let out_dir = env::var("OUT_DIR").unwrap();
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
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }

    // Link NPP libraries
    let npp_lib_dir = if target_os == "windows" {
        PathBuf::from(&cuda_path).join("lib/x64")
    } else {
        PathBuf::from(&cuda_path).join("lib64")
    };
    println!("cargo:rustc-link-search=native={}", npp_lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=nppc"); // NPP core
    println!("cargo:rustc-link-lib=dylib=nppig"); // NPP image geometry (warp, resize)
    println!("cargo:rustc-link-lib=dylib=nppif"); // NPP image filtering (gaussian blur)
    println!("cargo:rustc-link-lib=dylib=nppim"); // NPP image morphology (dilate/erode)
    println!("cargo:rustc-link-lib=dylib=nppicc"); // NPP image color conversion
    println!("cargo:rustc-link-lib=dylib=nppial"); // NPP arithmetic and logical (dilate)
    println!("cargo:rustc-link-lib=dylib=nppidei"); // NPP data exchange and initialization

    // Also link CUDA runtime for NPP stream management
    println!("cargo:rustc-link-lib=dylib=cudart");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=gpu_kernels/");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=NOPERSON_CUDA_ARCH");
}
