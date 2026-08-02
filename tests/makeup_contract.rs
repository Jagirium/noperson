use noperson::config::parameters::MakeupParams;
use noperson::pipeline::color::apply_makeup_reference;

#[test]
fn parser_makeup_tints_only_hair_and_lip_classes() {
    let image = vec![[100.0, 120.0, 140.0]; 5];
    let classes = vec![17, 12, 13, 1, 0];
    let hair = MakeupParams {
        enabled: true,
        color: [200.0, 20.0, 40.0],
        blend: 0.5,
    };
    let lips = MakeupParams {
        enabled: true,
        color: [240.0, 40.0, 80.0],
        blend: 0.25,
    };

    let output = apply_makeup_reference(&image, &classes, &hair, &lips);
    assert_eq!(output[0], [150.0, 70.0, 90.0]);
    assert_eq!(output[1], [135.0, 100.0, 125.0]);
    assert_eq!(output[2], [135.0, 100.0, 125.0]);
    assert_eq!(output[3], image[3]);
    assert_eq!(output[4], image[4]);
}

#[test]
fn disabled_makeup_is_an_exact_noop() {
    let image = vec![[17.0, 42.0, 99.0]; 3];
    let classes = vec![17, 12, 13];
    let output = apply_makeup_reference(
        &image,
        &classes,
        &MakeupParams::default(),
        &MakeupParams::default(),
    );
    assert_eq!(output, image);
}
