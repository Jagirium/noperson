use noperson::config::parameters::FaceSwapParams;
use noperson::pipeline::face_mask::generate_border_mask;

#[test]
fn live_defaults_use_python_compatible_pixel_borders() {
    let params = FaceSwapParams::default();

    let mask = generate_border_mask(
        128,
        params.border_top,
        params.border_bottom,
        params.border_left,
        params.border_right,
    );

    assert_eq!(params.border_top, 10);
    assert_eq!(params.border_bottom, 10);
    assert_eq!(params.border_left, 10);
    assert_eq!(params.border_right, 10);
    assert_eq!(params.border_blur, 10);
    assert_eq!(mask[0], 0.0);
    assert_eq!(mask[9 * 128 + 64], 0.0);
    assert_eq!(mask[10 * 128 + 64], 1.0);
    assert_eq!(mask[64 * 128 + 64], 1.0);
}
