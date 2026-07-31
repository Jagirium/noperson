use noperson::config::settings::DetectorModel;
use noperson::pipeline::face_detector::{AnchorDetector, AnchorDetectorKind, FaceDetector};

#[test]
fn runtime_detector_dispatch_preserves_the_selected_model() {
    for model in [
        DetectorModel::YoloFace8n,
        DetectorModel::RetinaFace,
        DetectorModel::Scrfd2_5g,
    ] {
        let detector = FaceDetector::from_model(model, 0.5);
        assert_eq!(detector.model(), model);
    }
}

#[test]
fn anchor_preprocessing_matches_crosswap_normalization() {
    let image = [255.0, 128.0, 0.0];
    let retina = AnchorDetector::new(AnchorDetectorKind::RetinaFace, 0.5);
    let scrfd = AnchorDetector::new(AnchorDetectorKind::Scrfd, 0.5);

    let (retina_input, retina_scale) = retina.preprocess(&image, 1, 1);
    let (scrfd_input, scrfd_scale) = scrfd.preprocess(&image, 1, 1);

    assert!((retina_input[0] - ((255.0 - 127.5) / 128.0)).abs() < 1e-6);
    assert!((retina_input[512 * 512] - ((128.0 - 127.5) / 128.0)).abs() < 1e-6);
    assert!((retina_input[2 * 512 * 512] - (-127.5 / 128.0)).abs() < 1e-6);
    assert_eq!(scrfd_input[0], 255.0);
    assert_eq!(scrfd_input[512 * 512], 128.0);
    assert_eq!(scrfd_input[2 * 512 * 512], 0.0);
    assert_eq!(retina_scale, 512.0);
    assert_eq!(scrfd_scale, 512.0);
}

#[test]
fn anchor_iobinding_contract_matches_supplied_onnx_graphs() {
    assert_eq!(AnchorDetectorKind::RetinaFace.input_name(), "input.1");
    assert_eq!(AnchorDetectorKind::Scrfd.input_name(), "input.1");
    assert_eq!(
        AnchorDetectorKind::RetinaFace.output_names(),
        [
            "448", "471", "494", "451", "474", "497", "454", "477", "500"
        ]
    );
    assert_eq!(
        AnchorDetectorKind::Scrfd.output_names(),
        [
            "446", "466", "486", "449", "469", "489", "452", "472", "492"
        ]
    );
    assert_eq!(
        AnchorDetectorKind::output_lengths(),
        [
            12_800, 3_200, 800, 51_200, 12_800, 3_200, 128_000, 32_000, 8_000
        ]
    );
    assert_eq!(AnchorDetectorKind::packed_output_len(), 252_000);
}
