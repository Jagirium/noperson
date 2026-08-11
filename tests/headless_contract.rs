use std::path::Path;

use noperson::config::parameters::SwapDim;
use noperson::config::settings::ExecutionProvider;
use noperson::headless::{FileWorkflow, build_plan, classify_workflow, prepare};
use noperson::launch::{HeadlessOptions, LaunchOptions};

#[test]
fn headless_routes_images_without_starting_a_video_pipeline() {
    assert_eq!(
        classify_workflow(
            Path::new("face.jpg"),
            Path::new("target.webp"),
            Path::new("result.png")
        )
        .unwrap(),
        FileWorkflow::Image
    );
}

#[test]
fn headless_routes_supported_video_containers_to_native_video() {
    assert_eq!(
        classify_workflow(
            Path::new("face.png"),
            Path::new("target.mkv"),
            Path::new("result.mp4")
        )
        .unwrap(),
        FileWorkflow::Video
    );
}

#[test]
fn headless_rejects_media_mismatches_before_gpu_initialization() {
    let error = classify_workflow(
        Path::new("face.mp4"),
        Path::new("target.png"),
        Path::new("result.png"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("source face must be an image"));

    let error = classify_workflow(
        Path::new("face.jpg"),
        Path::new("target.mp4"),
        Path::new("result.png"),
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("video target requires a video output")
    );
}

#[test]
fn headless_plan_maps_cli_quality_controls_into_the_engine_spec() {
    let options = LaunchOptions::parse([
        "headless-run".into(),
        "-s".into(),
        "face.jpg".into(),
        "-t".into(),
        "target.png".into(),
        "-o".into(),
        "result.png".into(),
        "--execution-provider".into(),
        "tensorrt".into(),
        "--swap-resolution".into(),
        "384".into(),
        "--face-detector-score".into(),
        "0.67".into(),
        "--max-faces".into(),
        "4".into(),
    ])
    .unwrap();

    let plan = build_plan(options.headless.as_ref().unwrap()).unwrap();
    assert_eq!(plan.provider, ExecutionProvider::TensorRt);
    assert_eq!(plan.params.dim, SwapDim::Dim3);
    assert_eq!(plan.params.detector_score, 0.67);
    assert_eq!(plan.params.max_faces, 4);
}

#[test]
fn headless_preflight_rejects_missing_inputs_before_runtime_bootstrap() {
    let options = HeadlessOptions {
        source_path: Path::new("definitely-missing-face.jpg").to_path_buf(),
        target_path: Path::new("definitely-missing-target.png").to_path_buf(),
        output_path: Path::new("result.png").to_path_buf(),
        ..HeadlessOptions::default()
    };

    let error = prepare(&options).unwrap_err();
    assert!(error.to_string().contains("source file not found"));
}
