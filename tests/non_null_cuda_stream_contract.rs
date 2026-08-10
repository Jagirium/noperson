//! Source-level regression guard for the GPU/ORT/NPP stream identity contract.

use std::path::Path;

fn collect_rust_files(root: &Path, files: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rust_files(&path, files);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            files.push(path);
        }
    }
}

#[test]
fn application_cuda_streams_are_owned_and_non_null() {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
    let forbidden = [".default_", "stream()"].concat();
    let mut rust_files = Vec::new();
    for directory in ["src", "tests", "examples"] {
        collect_rust_files(&repository.join(directory), &mut rust_files);
    }

    let offenders: Vec<_> = rust_files
        .into_iter()
        .filter_map(|path| {
            let source = std::fs::read_to_string(&path).ok()?;
            source
                .contains(&forbidden)
                .then(|| path.strip_prefix(repository).unwrap_or(&path).to_path_buf())
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "default CUDA stream is null and violates ORT/NPP stream identity: {offenders:?}"
    );
}
