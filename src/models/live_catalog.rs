//! Canonical model contract for the Rust-native live pipeline.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::ValueType;
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const CANONICAL_SWAPPER_FILENAME: &str = "inswapper_128.fp16.onnx";
pub const CANONICAL_SWAPPER_SHA256: &str =
    "f29a902862df018264ad4fd0c25387acd0581e168a9baa0372d71c465b65bf27";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveModel {
    pub logical_name: &'static str,
    pub filename: &'static str,
    pub sha256: &'static str,
    pub release_sha256: Option<&'static str>,
}

pub const LIVE_MODELS: &[LiveModel] = &[
    LiveModel {
        logical_name: "YoloFace8n",
        filename: "yoloface_8n.onnx",
        sha256: "84d5bb985b0ea75fc851d7454483897b1494c71c211759b4fec3a22ac196d206",
        release_sha256: None,
    },
    LiveModel {
        logical_name: "Inswapper128ArcFace",
        filename: "w600k_r50.onnx",
        sha256: "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        release_sha256: Some("0ddde02d672b5063bb79641844d4e938c9e1f1a66607b4e2436ef15036fe7c9a"),
    },
    LiveModel {
        logical_name: "Inswapper128",
        filename: CANONICAL_SWAPPER_FILENAME,
        sha256: CANONICAL_SWAPPER_SHA256,
        release_sha256: None,
    },
    LiveModel {
        logical_name: "InswapperEMap",
        filename: "emap.bin",
        sha256: "370af5bf707dafdbea8a40448d697d9697610bd223ecf92887af9c9cc7055ac8",
        release_sha256: None,
    },
];

#[derive(Debug, Error)]
pub enum ModelContractError {
    #[error("required model is missing: {path}")]
    Missing { path: PathBuf },
    #[error("failed to read model {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("model digest mismatch for {path}: expected {expected}, got {actual}")]
    DigestMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("invalid multibatch contract for {path}: {details}")]
    InvalidMultibatchContract { path: PathBuf, details: String },
}

pub fn validate_model_file(path: &Path, expected_sha256: &str) -> Result<(), ModelContractError> {
    validate_model_file_against_any(path, &[expected_sha256])
}

pub fn validate_model_file_against_any(
    path: &Path,
    expected_sha256: &[&str],
) -> Result<(), ModelContractError> {
    let mut file = File::open(path).map_err(|source| {
        if source.kind() == io::ErrorKind::NotFound {
            ModelContractError::Missing {
                path: path.to_path_buf(),
            }
        } else {
            ModelContractError::Read {
                path: path.to_path_buf(),
                source,
            }
        }
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| ModelContractError::Read {
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if !expected_sha256.contains(&actual.as_str()) {
        return Err(ModelContractError::DigestMismatch {
            path: path.to_path_buf(),
            expected: expected_sha256.join(" or "),
            actual,
        });
    }
    Ok(())
}

pub fn validate_live_models(root: &Path) -> Result<(), ModelContractError> {
    for model in LIVE_MODELS {
        let accepted = [model.sha256, model.release_sha256.unwrap_or(model.sha256)];
        validate_model_file_against_any(&root.join(model.filename), &accepted)?;
    }
    Ok(())
}

pub fn validate_swapper_multibatch_contract(
    path: impl AsRef<Path>,
) -> Result<(), ModelContractError> {
    let path = path.as_ref();
    let invalid = |details: String| ModelContractError::InvalidMultibatchContract {
        path: path.to_path_buf(),
        details,
    };
    let session = Session::builder()
        .and_then(|mut builder| builder.commit_from_file(path))
        .map_err(|error| invalid(error.to_string()))?;

    for (name, expected) in [
        ("target", &[-1_i64, 3, 128, 128][..]),
        ("source", &[-1_i64, 512][..]),
    ] {
        let outlet = session
            .inputs()
            .iter()
            .find(|input| input.name() == name)
            .ok_or_else(|| invalid(format!("missing input {name}")))?;
        let ValueType::Tensor { shape, .. } = outlet.dtype() else {
            return Err(invalid(format!("input {name} is not a tensor")));
        };
        if shape.as_ref() != expected {
            return Err(invalid(format!(
                "input {name} has shape {:?}, expected {expected:?}",
                shape.as_ref()
            )));
        }
    }
    Ok(())
}
