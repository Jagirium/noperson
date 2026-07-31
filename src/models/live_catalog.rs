//! Canonical model contract for the Rust-native live pipeline.

use std::io;
use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::ValueType;
use thiserror::Error;

use super::digest::file_blake3;

pub const CANONICAL_SWAPPER_FILENAME: &str = "inswapper_128.fp16.onnx";
pub const CANONICAL_SWAPPER_BLAKE3: &str =
    "65d91b580afffee6c0358c0fb48f9743a5502cbf8fff997e8051f76e07ae7bd0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveModel {
    pub logical_name: &'static str,
    pub filename: &'static str,
    pub blake3: &'static str,
    pub release_blake3: Option<&'static str>,
}

pub const LIVE_MODELS: &[LiveModel] = &[
    LiveModel {
        logical_name: "YoloFace8n",
        filename: "yoloface_8n.onnx",
        blake3: "d9e2c8b06b021310fa58059e35af1db46940e183f352a94fb51a786ccb094a26",
        release_blake3: None,
    },
    LiveModel {
        logical_name: "Inswapper128ArcFace",
        filename: "w600k_r50.onnx",
        blake3: "c9cc033a308d5cbe0006b8f2d695f13fe716c985cc4b676ad5c0a20a497a07cc",
        release_blake3: None,
    },
    LiveModel {
        logical_name: "Inswapper128",
        filename: CANONICAL_SWAPPER_FILENAME,
        blake3: CANONICAL_SWAPPER_BLAKE3,
        release_blake3: None,
    },
    LiveModel {
        logical_name: "InswapperEMap",
        filename: "emap.bin",
        blake3: "a54d56087b426005a6d77c9df3dcddfdb81ebc610d6af2235181ef2f97bae4b7",
        release_blake3: None,
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

pub fn validate_model_file(path: &Path, expected_blake3: &str) -> Result<(), ModelContractError> {
    validate_model_file_against_any(path, &[expected_blake3])
}

pub fn validate_model_file_against_any(
    path: &Path,
    expected_blake3: &[&str],
) -> Result<(), ModelContractError> {
    let actual = file_blake3(path).map_err(|source| {
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
    if !expected_blake3.contains(&actual.as_str()) {
        return Err(ModelContractError::DigestMismatch {
            path: path.to_path_buf(),
            expected: expected_blake3.join(" or "),
            actual,
        });
    }
    Ok(())
}

pub fn validate_live_models(root: &Path) -> Result<(), ModelContractError> {
    for model in LIVE_MODELS {
        let accepted = [model.blake3, model.release_blake3.unwrap_or(model.blake3)];
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
