use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use fs2::FileExt;
use tokio_util::sync::CancellationToken;

use crate::models::download::{DownloadConfig, DownloadProgress, ModelDownloader};

use super::registry::{GENERATION, artifacts_for};
use super::{RuntimeLayout, TensorRtShard};

#[derive(Debug, thiserror::Error)]
pub enum RuntimeInstallError {
    #[error("automatic GPU runtime installation is not published for {0} yet")]
    UnsupportedPlatform(&'static str),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Download(#[from] crate::models::download::DownloadError),
    #[error("failed to unpack {path}: {source}")]
    Unpack {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("installed runtime generation is incomplete: {0}")]
    Incomplete(PathBuf),
    #[error("runtime installation task failed: {0}")]
    Join(#[from] tokio::task::JoinError),
}

struct InstallLock(File);

impl Drop for InstallLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.0);
    }
}

pub async fn ensure_runtime(
    runtime_root: &Path,
    shard: TensorRtShard,
) -> Result<RuntimeLayout, RuntimeInstallError> {
    if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        return Err(RuntimeInstallError::UnsupportedPlatform(
            std::env::consts::OS,
        ));
    }
    let generation_name = format!("{}-{}", GENERATION, shard.directory());
    let final_root = runtime_root.join("generations").join(&generation_name);
    let layout = RuntimeLayout::new(final_root.clone(), shard);
    if layout.is_complete() {
        return Ok(layout);
    }

    fs::create_dir_all(runtime_root)?;
    let lock_path = runtime_root.join("install.lock");
    let lock = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(lock_path)?;
    FileExt::lock_exclusive(&lock)?;
    let _lock = InstallLock(lock);
    if layout.is_complete() {
        return Ok(layout);
    }

    let downloads = runtime_root.join("downloads");
    let downloader = ModelDownloader::new(DownloadConfig::default())?;
    let mut archives = Vec::new();
    for artifact in artifacts_for(shard) {
        tracing::info!(
            "Preparing GPU runtime artifact: {} ({:.1} MiB)",
            artifact.filename,
            artifact.size as f64 / (1024.0 * 1024.0)
        );
        let progress = DownloadProgress::default();
        let monitor_progress = progress.clone();
        let monitor_done = CancellationToken::new();
        let monitor_stop = monitor_done.clone();
        let artifact_name = artifact.filename;
        let monitor = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
            loop {
                tokio::select! {
                    () = monitor_stop.cancelled() => break,
                    _ = interval.tick() => {
                        let snapshot = monitor_progress.snapshot();
                        if snapshot.total_bytes > 0 {
                            tracing::info!(
                                "Runtime download: {artifact_name} {:.1}% ({}/{})",
                                snapshot.downloaded_bytes as f64 * 100.0 / snapshot.total_bytes as f64,
                                snapshot.downloaded_bytes,
                                snapshot.total_bytes
                            );
                        }
                    }
                }
            }
        });
        let archive = downloader
            .download(artifact, &downloads, CancellationToken::new(), progress)
            .await;
        monitor_done.cancel();
        let _ = monitor.await;
        let archive = archive?;
        archives.push(archive);
    }

    let staging = runtime_root
        .join("generations")
        .join(format!(".{generation_name}.staging-{}", std::process::id()));
    remove_directory_if_exists(&staging)?;
    fs::create_dir_all(&staging)?;
    let unpack_root = staging.clone();
    let archives_to_unpack = archives.clone();
    tokio::task::spawn_blocking(move || unpack_archives(&archives_to_unpack, &unpack_root))
        .await??;

    let staged_layout = RuntimeLayout::new(staging.clone(), shard);
    if !staged_layout.is_complete() {
        remove_directory_if_exists(&staging)?;
        return Err(RuntimeInstallError::Incomplete(staging));
    }
    fs::create_dir_all(final_root.parent().expect("generation has a parent"))?;
    if final_root.exists() {
        let quarantine = runtime_root.join("generations").join(format!(
            ".{generation_name}.incomplete-{}",
            std::process::id()
        ));
        remove_directory_if_exists(&quarantine)?;
        fs::rename(&final_root, quarantine)?;
    }
    fs::rename(&staging, &final_root)?;
    for archive in archives {
        if let Err(error) = fs::remove_file(&archive) {
            tracing::warn!(
                "Could not remove downloaded runtime archive {}: {error}",
                archive.display()
            );
        }
    }
    Ok(RuntimeLayout::new(final_root, shard))
}

fn unpack_archives(archives: &[PathBuf], destination: &Path) -> Result<(), RuntimeInstallError> {
    for path in archives {
        let file = File::open(path)?;
        let decoder = zstd::stream::read::Decoder::new(file).map_err(|source| {
            RuntimeInstallError::Unpack {
                path: path.clone(),
                source,
            }
        })?;
        let mut archive = tar::Archive::new(decoder);
        archive
            .unpack(destination)
            .map_err(|source| RuntimeInstallError::Unpack {
                path: path.clone(),
                source,
            })?;
    }
    Ok(())
}

fn remove_directory_if_exists(path: &Path) -> Result<(), io::Error> {
    match fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn trusted_zstd_archives_unpack_beneath_the_generation_root() {
        let fixture = tempfile::tempdir().unwrap();
        let archive_path = fixture.path().join("fixture.tar.zst");
        let archive_file = File::create(&archive_path).unwrap();
        let encoder = zstd::stream::write::Encoder::new(archive_file, 1).unwrap();
        let mut archive = tar::Builder::new(encoder);
        let bytes = b"runtime";
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();
        archive
            .append_data(&mut header, "base/libnppc.so.12", &bytes[..])
            .unwrap();
        let encoder = archive.into_inner().unwrap();
        encoder.finish().unwrap().flush().unwrap();

        let output = fixture.path().join("generation");
        fs::create_dir(&output).unwrap();
        unpack_archives(&[archive_path], &output).unwrap();
        assert_eq!(fs::read(output.join("base/libnppc.so.12")).unwrap(), bytes);
    }
}
