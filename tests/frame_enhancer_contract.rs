use noperson::pipeline::frame_enhancer::{CROSSSWAP_TILE_SIZE, EnhancerModel, TilePlan};

#[test]
fn production_enhancer_uses_crosswap_frame_worker_tile_size() {
    assert_eq!(CROSSSWAP_TILE_SIZE, 512);
}

#[test]
fn crosswap_enhancer_names_select_the_exact_registry_model_and_scale() {
    let cases = [
        ("RealEsrgan-x2-Plus", "RealEsrganx2Plus", 2),
        ("RealEsrgan-x4-Plus", "RealEsrganx4Plus", 4),
        ("BSRGan-x2", "BSRGANx2", 2),
        ("BSRGan-x4", "BSRGANx4", 4),
        ("UltraSharp-x4", "UltraSharpx4", 4),
        ("UltraMix-x4", "UltraMixx4", 4),
        ("RealEsr-General-x4v3", "RealEsrx4v3", 4),
    ];

    for (crosswap_name, registry_name, scale) in cases {
        let model = EnhancerModel::from_crosswap_name(crosswap_name).unwrap();
        assert_eq!(model.crosswap_name(), crosswap_name);
        assert_eq!(model.registry_name(), registry_name);
        assert_eq!(model.scale(), scale);
    }
    assert!(EnhancerModel::from_crosswap_name("unknown").is_err());
}

#[test]
fn tile_plan_matches_crosswap_batch_order_padding_and_output_crop() {
    let plan = TilePlan::new(500, 300, 256, 4).unwrap();

    assert_eq!(plan.padded_width, 512);
    assert_eq!(plan.padded_height, 512);
    assert_eq!(plan.output_width, 2000);
    assert_eq!(plan.output_height, 1200);
    assert_eq!(plan.tiles.len(), 4);

    let input_origins: Vec<_> = plan
        .tiles
        .iter()
        .map(|tile| (tile.input_x, tile.input_y))
        .collect();
    assert_eq!(input_origins, [(0, 0), (256, 0), (0, 256), (256, 256)]);

    let output_origins: Vec<_> = plan
        .tiles
        .iter()
        .map(|tile| (tile.output_x, tile.output_y))
        .collect();
    assert_eq!(output_origins, [(0, 0), (1024, 0), (0, 1024), (1024, 1024)]);
}

#[test]
fn tile_plan_sizes_one_persistent_batched_workspace() {
    let plan = TilePlan::new(500, 300, 256, 4).unwrap();

    assert_eq!(plan.input_shape(), [4, 3, 256, 256]);
    assert_eq!(plan.batched_input_elements().unwrap(), 4 * 3 * 256 * 256);
    assert_eq!(plan.batched_output_elements().unwrap(), 4 * 3 * 1024 * 1024);
    assert_eq!(plan.output_elements().unwrap(), 3 * 2000 * 1200);
}

#[test]
fn tile_plan_rejects_empty_frames_and_tiles() {
    assert!(TilePlan::new(0, 300, 256, 4).is_err());
    assert!(TilePlan::new(500, 0, 256, 4).is_err());
    assert!(TilePlan::new(500, 300, 0, 4).is_err());
    assert!(TilePlan::new(500, 300, 256, 0).is_err());
}
