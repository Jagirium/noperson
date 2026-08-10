use std::fs::{self, File, OpenOptions};
use std::io::{self, IsTerminal};
use std::path::{Path, PathBuf};
use std::time::Duration;

use fs2::FileExt;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use tokio_util::sync::CancellationToken;

use crate::models::download::{DownloadConfig, DownloadPhase, DownloadProgress, ModelDownloader};

use super::registry::{RuntimePlatform, artifacts_for, generation_name_for};
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
    let platform = platform_for(std::env::consts::OS, std::env::consts::ARCH)?;
    let generation_name = generation_name_for(platform, shard);
    let final_root = runtime_root.join("generations").join(&generation_name);
    let layout = RuntimeLayout::new(final_root.clone(), shard);
    if layout.is_complete() {
        return Ok(layout);
    }

    fs::create_dir_all(runtime_root)?;
    let lock_path = runtime_root.join("install.lock");
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)?;
    FileExt::lock_exclusive(&lock)?;
    let _lock = InstallLock(lock);
    if layout.is_complete() {
        return Ok(layout);
    }

    let downloads = runtime_tmp_directory(runtime_root)?;
    let downloader = ModelDownloader::new(DownloadConfig::default())?;
    let mut archives = Vec::new();
    for artifact in artifacts_for(platform, shard) {
        tracing::info!(
            "Preparing GPU runtime artifact: {} ({:.1} MiB)",
            artifact.filename,
            artifact.size as f64 / (1024.0 * 1024.0)
        );
        let progress = DownloadProgress::default();
        let monitor_progress = progress.clone();
        let progress_bar = runtime_progress_bar(artifact.size, io::stderr().is_terminal());
        let monitor_bar = progress_bar.clone();
        let monitor_done = CancellationToken::new();
        let monitor_stop = monitor_done.clone();
        let monitor = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(100));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    () = monitor_stop.cancelled() => {
                        update_runtime_progress(&monitor_bar, &monitor_progress);
                        break;
                    },
                    _ = interval.tick() => {
                        update_runtime_progress(&monitor_bar, &monitor_progress);
                    }
                }
            }
        });
        let archive = downloader
            .download(
                artifact,
                &downloads,
                CancellationToken::new(),
                progress.clone(),
            )
            .await;
        monitor_done.cancel();
        let _ = monitor.await;
        match progress.snapshot().phase {
            DownloadPhase::Completed => progress_bar.finish_with_message("GPU runtime ready"),
            DownloadPhase::Cancelled => progress_bar.abandon_with_message("GPU runtime cancelled"),
            DownloadPhase::Failed => progress_bar.abandon_with_message("GPU runtime failed"),
            _ => progress_bar.finish_and_clear(),
        }
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

fn platform_for(
    os: &'static str,
    architecture: &'static str,
) -> Result<RuntimePlatform, RuntimeInstallError> {
    match (os, architecture) {
        ("linux", "x86_64") => Ok(RuntimePlatform::LinuxX86_64),
        ("windows", "x86_64") => Ok(RuntimePlatform::WindowsX86_64),
        _ => Err(RuntimeInstallError::UnsupportedPlatform(os)),
    }
}

fn runtime_tmp_directory(runtime_root: &Path) -> io::Result<PathBuf> {
    let temporary = runtime_root.join("tmp");
    let legacy = runtime_root.join("downloads");
    if !temporary.exists() && legacy.exists() {
        fs::rename(&legacy, &temporary)?;
    }
    fs::create_dir_all(&temporary)?;
    Ok(temporary)
}

fn runtime_progress_bar(total_bytes: u64, terminal: bool) -> ProgressBar {
    let target = if terminal {
        ProgressDrawTarget::stderr_with_hz(10)
    } else {
        ProgressDrawTarget::hidden()
    };
    let bar = ProgressBar::with_draw_target(Some(total_bytes), target);
    bar.set_style(
        ProgressStyle::with_template(
            "GPU runtime [{bar:36.cyan/blue}] {bytes}/{total_bytes} {percent}% · {bytes_per_sec} · ETA {eta}",
        )
        .expect("runtime progress template is valid")
        .progress_chars("━━─"),
    );
    bar
}

fn update_runtime_progress(bar: &ProgressBar, progress: &DownloadProgress) {
    let snapshot = progress.snapshot();
    if snapshot.total_bytes > 0 {
        bar.set_length(snapshot.total_bytes);
    }
    bar.set_position(snapshot.downloaded_bytes.min(snapshot.total_bytes));
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

    #[test]
    fn legacy_downloads_migrate_to_the_private_runtime_tmp_directory() {
        let fixture = tempfile::tempdir().unwrap();
        let legacy = fixture.path().join("downloads");
        fs::create_dir(&legacy).unwrap();
        fs::write(legacy.join("runtime.tar.zst.tmp"), b"partial").unwrap();

        let temporary = runtime_tmp_directory(fixture.path()).unwrap();

        assert_eq!(temporary, fixture.path().join("tmp"));
        assert_eq!(
            fs::read(temporary.join("runtime.tar.zst.tmp")).unwrap(),
            b"partial"
        );
        assert!(!legacy.exists());
    }

    #[test]
    fn non_terminal_runtime_progress_is_hidden() {
        let progress = runtime_progress_bar(1024, false);
        assert!(progress.is_hidden());
        assert_eq!(progress.length(), Some(1024));
    }

    #[test]
    fn only_published_platform_architecture_pairs_are_accepted() {
        assert_eq!(
            platform_for("linux", "x86_64").unwrap(),
            RuntimePlatform::LinuxX86_64
        );
        assert_eq!(
            platform_for("windows", "x86_64").unwrap(),
            RuntimePlatform::WindowsX86_64
        );
        for (os, arch) in [
            ("linux", "aarch64"),
            ("windows", "aarch64"),
            ("macos", "x86_64"),
        ] {
            assert!(matches!(
                platform_for(os, arch),
                Err(RuntimeInstallError::UnsupportedPlatform(value)) if value == os
            ));
        }
    }
}
