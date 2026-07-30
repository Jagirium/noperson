use std::path::Path;

use noperson::models::live_catalog::{
    CANONICAL_SWAPPER_FILENAME, CANONICAL_SWAPPER_SHA256, ModelContractError, validate_live_models,
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
    std::fs::write(&path, b"definitely not an ONNX model").unwrap();

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
                "990e756c46e9ea4b6aebdc33b4ef5adb4cac3b97bfbcfb614134d8d2381f0901"
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
    std::fs::write(&path, b"release candidate").unwrap();

    validate_model_file_against_any(
        &path,
        &[
            "source-digest",
            "4b5297a5261624acded347f2aec687e6e3ac4153d1a056c571e48b7651197d40",
        ],
    )
    .unwrap();
    let _ = std::fs::remove_file(&path);
}

#[test]
fn canonical_swapper_contract_matches_the_supplied_asset() {
    assert_eq!(CANONICAL_SWAPPER_FILENAME, "inswapper_128.fp16.onnx");
    assert_eq!(
        CANONICAL_SWAPPER_SHA256,
        "f29a902862df018264ad4fd0c25387acd0581e168a9baa0372d71c465b65bf27"
    );
    validate_swapper_multibatch_contract(Path::new("models").join(CANONICAL_SWAPPER_FILENAME))
        .expect("canonical swapper must expose a dynamic batch axis");
}
