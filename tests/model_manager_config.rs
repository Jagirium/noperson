use noperson::config::settings::ExecutionProvider;
use noperson::models::manager::ModelManager;

#[test]
fn model_manager_preserves_generation_execution_target() {
    let manager = ModelManager::with_execution("models", ExecutionProvider::TensorRT, 2);

    assert_eq!(manager.provider(), ExecutionProvider::TensorRT);
    assert_eq!(manager.device_id(), 2);
}
