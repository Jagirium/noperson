use noperson::quality::compare_rgb;

#[test]
fn identical_frames_have_perfect_quality_metrics() {
    let frame = vec![42_u8; 32 * 24 * 3];
    let metrics = compare_rgb(&frame, &frame, 32, 24).unwrap();

    assert_eq!(metrics.mae, 0.0);
    assert!(metrics.psnr.is_infinite());
    assert_eq!(metrics.seam_p99, 0.0);
}

#[test]
fn seam_metric_penalizes_a_hard_rectangle_more_than_a_feathered_one() {
    let (width, height) = (64_u32, 64_u32);
    let original = vec![0_u8; (width * height * 3) as usize];
    let mut hard = original.clone();
    let mut soft = original.clone();

    for y in 12..52 {
        for x in 12..52 {
            let hard_value = 200_u8;
            let distance = (x - 12).min(51 - x).min((y - 12).min(51 - y));
            let soft_value = ((distance.min(8) as f32 / 8.0) * 200.0).round() as u8;
            for channel in 0..3 {
                let index = ((y * width + x) * 3 + channel) as usize;
                hard[index] = hard_value;
                soft[index] = soft_value;
            }
        }
    }

    let hard_metrics = compare_rgb(&original, &hard, width, height).unwrap();
    let soft_metrics = compare_rgb(&original, &soft, width, height).unwrap();
    assert!(
        hard_metrics.seam_p99 > soft_metrics.seam_p99 * 4.0,
        "hard={} soft={}",
        hard_metrics.seam_p99,
        soft_metrics.seam_p99
    );
}
