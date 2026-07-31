use noperson::config::settings::ExecutionProvider;
use noperson::models::manager::ModelManager;

#[test]
fn model_manager_preserves_generation_execution_target() {
    let manager = ModelManager::with_execution("models", ExecutionProvider::TensorRT, 2);

    assert_eq!(manager.provider(), ExecutionProvider::TensorRT);
    assert_eq!(manager.device_id(), 2);
}

#[test]
fn emap_filename_comes_from_the_generation_spec() {
    let root = std::env::temp_dir().join(format!(
        "noperson-emap-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("generation.emap"), vec![0_u8; 512 * 512 * 4]).unwrap();
    let mut manager = ModelManager::new(&root);

    manager.load_emap_file("generation.emap").unwrap();

    assert_eq!(manager.emap.as_ref().unwrap().len(), 512 * 512);
    std::fs::remove_dir_all(root).unwrap();
}
