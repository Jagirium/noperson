//! Build script — compiles CUDA kernels (.cu → .ptx) and links NPP.
//!
//! PTX files are loaded at runtime by gpu::ops and gpu::kernels.
//! NPP libraries provide optimized image geometry (warp, resize, blur).

use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    let kernels_dir = Path::new("gpu_kernels");
    let cuda_path = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = format!("{cuda_path}/bin/nvcc");

    // Compile each .cu file to .ptx
    if kernels_dir.exists() {
        for entry in std::fs::read_dir(kernels_dir).expect("Failed to read gpu_kernels/") {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "cu") {
                let stem = path.file_stem().unwrap().to_str().unwrap();
                let out_dir = env::var("OUT_DIR").unwrap();
                let ptx_out = format!("{out_dir}/{stem}.ptx");

                let status = Command::new(&nvcc)
                    .args([
                        "--ptx",
                        "-arch=sm_86", // RTX 3090
                        "-O3",
                        "--use_fast_math",
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
    let npp_lib_dir = format!("{cuda_path}/lib64");
    println!("cargo:rustc-link-search=native={npp_lib_dir}");
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
}
