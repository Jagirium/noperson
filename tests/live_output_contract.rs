use noperson::config::parameters::{EnhancerModel, FaceSwapParams};
use noperson::live::output_dimensions;

#[test]
fn frame_enhancer_changes_photo_video_dimensions_but_disabled_mode_does_not() {
    let disabled = FaceSwapParams::default();
    assert_eq!(
        output_dimensions(&disabled, 1920, 1080).unwrap(),
        (1920, 1080)
    );

    let mut x2 = disabled.clone();
    x2.enhancer_enabled = true;
    assert_eq!(output_dimensions(&x2, 1920, 1080).unwrap(), (3840, 2160));

    let mut x4 = x2;
    x4.enhancer_model = EnhancerModel::UltraSharpX4;
    assert_eq!(output_dimensions(&x4, 500, 300).unwrap(), (2000, 1200));
}

#[test]
fn frame_enhancer_dimensions_reject_overflow() {
    let mut params = FaceSwapParams::default();
    params.enhancer_enabled = true;
    params.enhancer_model = EnhancerModel::UltraMixX4;
    assert!(output_dimensions(&params, u32::MAX, 1).is_err());
}
