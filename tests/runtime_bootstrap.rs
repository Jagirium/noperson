use std::path::PathBuf;

use noperson::runtime::{ComputeCapability, RuntimeLayout, TensorRtShard};

#[test]
fn compute_capability_selects_an_exact_tensorrt_builder_shard() {
    for ((major, minor), expected) in [
        ((7, 5), TensorRtShard::Sm75),
        ((8, 0), TensorRtShard::Sm80),
        ((8, 6), TensorRtShard::Sm86),
        ((8, 9), TensorRtShard::Sm89),
        ((9, 0), TensorRtShard::Sm90),
        ((10, 0), TensorRtShard::Sm100),
        ((12, 0), TensorRtShard::Sm120),
        ((13, 0), TensorRtShard::Ptx),
    ] {
        assert_eq!(
            ComputeCapability { major, minor }.tensorrt_shard(),
            expected
        );
    }
}

#[test]
fn runtime_layout_keeps_common_and_arch_specific_libraries_separate() {
    let layout = RuntimeLayout::new(PathBuf::from("/runtime/generation"), TensorRtShard::Sm89);
    assert_eq!(layout.base(), PathBuf::from("/runtime/generation/base"));
    assert_eq!(
        layout.tensorrt_base(),
        PathBuf::from("/runtime/generation/trt/base")
    );
    assert_eq!(
        layout.tensorrt_shard(),
        PathBuf::from("/runtime/generation/trt/sm89")
    );
    assert_eq!(
        layout.library_paths(),
        vec![
            PathBuf::from("/runtime/generation/base"),
            PathBuf::from("/runtime/generation/trt/base"),
            PathBuf::from("/runtime/generation/trt/sm89"),
        ]
    );
}

#[cfg(target_os = "linux")]
#[test]
fn runtime_layout_is_complete_only_with_every_early_and_provider_library() {
    let directory = tempfile::tempdir().unwrap();
    let layout = RuntimeLayout::new(directory.path().to_path_buf(), TensorRtShard::Sm86);
    for relative in [
        "base/libnppc.so.12",
        "base/libnppig.so.12",
        "base/libnppif.so.12",
        "base/libnppim.so.12",
        "base/libonnxruntime_providers_shared.so",
        "base/libonnxruntime_providers_cuda.so",
        "trt/base/libnvinfer.so.10",
        "trt/base/libonnxruntime_providers_tensorrt.so",
        "trt/sm86/libnvinfer_builder_resource_sm86.so.10",
    ] {
        let path = directory.path().join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"fixture").unwrap();
    }
    assert!(layout.is_complete());
    std::fs::remove_file(directory.path().join("base/libnppif.so.12")).unwrap();
    assert!(!layout.is_complete());
}
