//! Automatic installation of the minimal model set required by both frontends.

use std::io::{self, IsTerminal};
use std::path::Path;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use tokio_util::sync::CancellationToken;

use super::download::{DownloadConfig, DownloadPhase, DownloadProgress, ModelDownloader};
use super::registry::find_model;

const REQUIRED_MODELS: &[&str] = &[
    "YoloFace8n",
    "Inswapper128ArcFace",
    "Inswapper128",
    "InswapperEMap",
];

#[derive(Debug, thiserror::Error)]
pub enum ModelInstallError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Download(#[from] super::download::DownloadError),
    #[error("required model is absent from the registry: {0}")]
    MissingRegistryEntry(&'static str),
    #[error("model progress task failed: {0}")]
    Join(#[from] tokio::task::JoinError),
}

/// Ensure only the four models needed by the live swap pipeline are present.
/// Existing user-provided files are preserved verbatim.
pub async fn ensure_required_models(models_dir: &Path) -> Result<(), ModelInstallError> {
    tokio::fs::create_dir_all(models_dir).await?;
    let downloader = ModelDownloader::new(DownloadConfig::default())?;

    for logical_name in REQUIRED_MODELS {
        let model = find_model(logical_name)
            .ok_or(ModelInstallError::MissingRegistryEntry(logical_name))?;
        let destination = models_dir.join(model.filename);
        if destination.is_file() {
            continue;
        }

        tracing::info!(
            "Downloading required model: {} ({:.1} MiB)",
            model.filename,
            model.size as f64 / (1024.0 * 1024.0)
        );
        let progress = DownloadProgress::default();
        let progress_bar = model_progress_bar(model.size, io::stderr().is_terminal());
        progress_bar.set_message(model.filename);
        let stop = CancellationToken::new();
        let monitor = spawn_progress_monitor(progress.clone(), progress_bar.clone(), stop.clone());

        let result = downloader
            .download(
                model,
                models_dir,
                CancellationToken::new(),
                progress.clone(),
            )
            .await;
        stop.cancel();
        monitor.await?;
        match progress.snapshot().phase {
            DownloadPhase::Completed => progress_bar.finish_with_message(model.filename),
            DownloadPhase::Cancelled => progress_bar.abandon_with_message("download cancelled"),
            DownloadPhase::Failed => progress_bar.abandon_with_message("download failed"),
            _ => progress_bar.finish_and_clear(),
        }
        result?;
    }

    Ok(())
}

fn spawn_progress_monitor(
    progress: DownloadProgress,
    bar: ProgressBar,
    stop: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(100));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                () = stop.cancelled() => {
                    update_progress(&bar, &progress);
                    break;
                }
                _ = interval.tick() => update_progress(&bar, &progress),
            }
        }
    })
}

fn update_progress(bar: &ProgressBar, progress: &DownloadProgress) {
    let snapshot = progress.snapshot();
    if snapshot.total_bytes > 0 {
        bar.set_length(snapshot.total_bytes);
    }
    bar.set_position(snapshot.downloaded_bytes);
}

fn model_progress_bar(size: u64, visible: bool) -> ProgressBar {
    let target = if visible {
        ProgressDrawTarget::stderr_with_hz(10)
    } else {
        ProgressDrawTarget::hidden()
    };
    let bar = ProgressBar::with_draw_target(Some(size), target);
    bar.set_style(
        ProgressStyle::with_template(
            "Model {msg} [{bar:32.cyan/blue}] {bytes}/{total_bytes} {percent}% · {bytes_per_sec} · ETA {eta}",
        )
        .expect("static progress template is valid")
        .progress_chars("█▓▒░ "),
    );
    bar
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{REQUIRED_MODELS, model_progress_bar};
    use crate::models::registry::find_model;

    #[test]
    fn installs_only_the_minimum_live_pipeline() {
        assert_eq!(
            REQUIRED_MODELS,
            [
                "YoloFace8n",
                "Inswapper128ArcFace",
                "Inswapper128",
                "InswapperEMap"
            ]
        );
    }

    #[test]
    fn progress_is_hidden_when_stderr_is_not_a_terminal() {
        assert!(model_progress_bar(1024, false).is_hidden());
    }

    #[tokio::test]
    async fn existing_user_models_are_not_replaced_or_downloaded() {
        let directory = tempfile::tempdir().unwrap();
        for logical_name in REQUIRED_MODELS {
            let filename = find_model(logical_name).unwrap().filename;
            fs::write(directory.path().join(filename), b"user model").unwrap();
        }

        super::ensure_required_models(directory.path())
            .await
            .unwrap();

        for logical_name in REQUIRED_MODELS {
            let filename = find_model(logical_name).unwrap().filename;
            assert_eq!(
                fs::read(directory.path().join(filename)).unwrap(),
                b"user model"
            );
        }
    }
}
