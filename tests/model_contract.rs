use std::path::Path;

use noperson::models::live_catalog::{
    CANONICAL_SWAPPER_BLAKE3, CANONICAL_SWAPPER_FILENAME, ModelContractError, validate_live_models,
    validate_model_file, validate_model_file_against_any, validate_swapper_multibatch_contract,
};

#[test]
fn canonical_live_assets_pass_digest_validation() {
    validate_live_models(Path::new("models")).expect("canonical live model set must validate");
}

#[test]
fn wrong_digest_reports_the_model_path_and_both_digests() {
    let path = std::env::temp_dir().join(format!(
        "noperson-model-contract-{}-{}.onnx",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    std::fs::write(&path, b"abc").unwrap();

    let error = validate_model_file(&path, "00").unwrap_err();
    let _ = std::fs::remove_file(&path);

    match error {
        ModelContractError::DigestMismatch {
            path: actual_path,
            expected,
            actual,
        } => {
            assert_eq!(actual_path, path);
            assert_eq!(expected, "00");
            assert_eq!(
                actual,
                "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"
            );
        }
        other => panic!("expected a digest mismatch, got {other}"),
    }
}

#[test]
fn release_digest_is_accepted_alongside_the_source_digest() {
    let path = std::env::temp_dir().join(format!(
        "noperson-release-model-contract-{}-{}.onnx",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    std::fs::write(&path, b"").unwrap();

    validate_model_file_against_any(
        &path,
        &[
            "source-digest",
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
        ],
    )
    .unwrap();
    let _ = std::fs::remove_file(&path);
}

#[test]
fn canonical_swapper_contract_matches_the_supplied_asset() {
    assert_eq!(CANONICAL_SWAPPER_FILENAME, "inswapper_128.fp16.onnx");
    assert_eq!(
        CANONICAL_SWAPPER_BLAKE3,
        "65d91b580afffee6c0358c0fb48f9743a5502cbf8fff997e8051f76e07ae7bd0"
    );
    validate_swapper_multibatch_contract(Path::new("models").join(CANONICAL_SWAPPER_FILENAME))
        .expect("canonical swapper must expose a dynamic batch axis");
}
