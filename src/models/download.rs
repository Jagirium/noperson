//! Resumable, integrity-checked model downloads with adaptive mirror failover.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use futures_util::{StreamExt, future::join_all, stream};
use reqwest::header::{CONTENT_RANGE, RANGE};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use super::digest::file_blake3;
use super::registry::{ModelEntry, ModelMirror, find_model};

const RESUME_VERSION: u8 = 1;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum DownloadPhase {
    #[default]
    Idle,
    Probing,
    Downloading,
    Verifying,
    Completed,
    Cancelled,
    Failed,
}

impl DownloadPhase {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Probing,
            2 => Self::Downloading,
            3 => Self::Verifying,
            4 => Self::Completed,
            5 => Self::Cancelled,
            6 => Self::Failed,
            _ => Self::Idle,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DownloadSnapshot {
    pub phase: DownloadPhase,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub active_mirror: Option<String>,
}

#[derive(Default)]
struct ProgressInner {
    phase: AtomicU8,
    downloaded_bytes: AtomicU64,
    total_bytes: AtomicU64,
    active_mirror: std::sync::Mutex<Option<String>>,
}

#[derive(Clone, Default)]
pub struct DownloadProgress(Arc<ProgressInner>);

impl DownloadProgress {
    pub fn snapshot(&self) -> DownloadSnapshot {
        DownloadSnapshot {
            phase: DownloadPhase::from_u8(self.0.phase.load(Ordering::Acquire)),
            downloaded_bytes: self.0.downloaded_bytes.load(Ordering::Acquire),
            total_bytes: self.0.total_bytes.load(Ordering::Acquire),
            active_mirror: self
                .0
                .active_mirror
                .lock()
                .ok()
                .and_then(|value| value.clone()),
        }
    }

    fn set_phase(&self, phase: DownloadPhase) {
        self.0.phase.store(phase as u8, Ordering::Release);
    }

    fn set_mirror(&self, name: &str) {
        if let Ok(mut mirror) = self.0.active_mirror.lock() {
            *mirror = Some(name.to_owned());
        }
    }
}

#[derive(Clone, Debug)]
pub struct DownloadConfig {
    pub chunk_size: u64,
    pub concurrency: usize,
    pub probe_bytes: u64,
    pub request_timeout: Duration,
    pub stall_timeout: Duration,
    pub speed_window: Duration,
    pub min_throughput_bytes_per_second: u64,
    pub max_attempts_per_chunk: usize,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            chunk_size: 8 * 1024 * 1024,
            concurrency: 8,
            probe_bytes: 1024 * 1024,
            request_timeout: Duration::from_secs(30),
            stall_timeout: Duration::from_secs(8),
            speed_window: Duration::from_millis(500),
            min_throughput_bytes_per_second: 300 * 1024,
            max_attempts_per_chunk: 6,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("invalid downloader configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("all mirrors failed their range probe")]
    NoHealthyMirrors,
    #[error("model is not registered: {0}")]
    UnknownModel(String),
    #[error("download cancelled")]
    Cancelled,
    #[error("model download failed: {0}")]
    Transfer(String),
    #[error("downloaded size mismatch: expected {expected}, got {actual}")]
    SizeMismatch { expected: u64, actual: u64 },
    #[error("BLAKE3 mismatch: expected {expected}, got {actual}")]
    Blake3Mismatch { expected: String, actual: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Http(#[from] reqwest::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Join(#[from] tokio::task::JoinError),
}

#[derive(Clone)]
pub struct ModelDownloader {
    client: reqwest::Client,
    config: DownloadConfig,
}

#[derive(Clone, Debug)]
struct RankedMirror {
    mirror: ModelMirror,
    elapsed: Duration,
    supports_ranges: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ResumeState {
    version: u8,
    size: u64,
    chunk_size: u64,
    blake3: String,
    completed: Vec<bool>,
}

impl ModelDownloader {
    pub fn new(config: DownloadConfig) -> Result<Self, DownloadError> {
        if config.chunk_size == 0 {
            return Err(DownloadError::InvalidConfig("chunk_size must be non-zero"));
        }
        if config.concurrency == 0 {
            return Err(DownloadError::InvalidConfig("concurrency must be non-zero"));
        }
        if config.probe_bytes == 0 {
            return Err(DownloadError::InvalidConfig("probe_bytes must be non-zero"));
        }
        if config.max_attempts_per_chunk == 0 {
            return Err(DownloadError::InvalidConfig(
                "max_attempts_per_chunk must be non-zero",
            ));
        }

        let client = reqwest::Client::builder()
            .connect_timeout(config.request_timeout)
            .timeout(config.request_timeout)
            .redirect(reqwest::redirect::Policy::limited(10))
            .build()?;
        Ok(Self { client, config })
    }

    /// Resolve a logical registry name and ensure its content is present.
    /// Engine/application lifecycle code can call this before building a new
    /// immutable generation without knowing filenames or mirror URLs.
    pub async fn download_registered(
        &self,
        name: &str,
        models_dir: &Path,
        cancellation: CancellationToken,
        progress: DownloadProgress,
    ) -> Result<PathBuf, DownloadError> {
        let model = find_model(name).ok_or_else(|| DownloadError::UnknownModel(name.to_owned()))?;
        self.download(model, models_dir, cancellation, progress)
            .await
    }

    pub async fn download(
        &self,
        model: &ModelEntry,
        models_dir: &Path,
        cancellation: CancellationToken,
        progress: DownloadProgress,
    ) -> Result<PathBuf, DownloadError> {
        progress.0.total_bytes.store(model.size, Ordering::Release);
        tokio::fs::create_dir_all(models_dir).await?;
        let final_path = models_dir.join(model.filename);

        if self.valid_file(&final_path, model).await? {
            progress
                .0
                .downloaded_bytes
                .store(model.size, Ordering::Release);
            progress.set_phase(DownloadPhase::Completed);
            return Ok(final_path);
        }

        let result = self
            .download_inner(model, models_dir, &final_path, &cancellation, &progress)
            .await;
        if result.is_err() {
            progress.set_phase(if cancellation.is_cancelled() {
                DownloadPhase::Cancelled
            } else {
                DownloadPhase::Failed
            });
        }
        result
    }

    async fn download_inner(
        &self,
        model: &ModelEntry,
        models_dir: &Path,
        final_path: &Path,
        cancellation: &CancellationToken,
        progress: &DownloadProgress,
    ) -> Result<PathBuf, DownloadError> {
        progress.set_phase(DownloadPhase::Probing);
        let ranked = self.probe_mirrors(model, cancellation).await?;

        let temporary_path = models_dir.join(format!("{}.tmp", model.filename));
        let state_path = models_dir.join(format!("{}.state", model.filename));
        let ranged = ranked
            .iter()
            .filter(|mirror| mirror.supports_ranges)
            .cloned()
            .collect::<Vec<_>>();
        if ranged.is_empty() {
            return self
                .download_single_stream(
                    model,
                    final_path,
                    &temporary_path,
                    &state_path,
                    &ranked,
                    cancellation,
                    progress,
                )
                .await;
        }
        let chunk_count = model.size.div_ceil(self.config.chunk_size) as usize;
        let state = self
            .load_or_create_state(model, &temporary_path, &state_path, chunk_count)
            .await?;
        let state = Arc::new(Mutex::new(state));
        let completed_bytes = {
            let state = state.lock().await;
            state
                .completed
                .iter()
                .enumerate()
                .filter(|(_, completed)| **completed)
                .map(|(index, _)| self.chunk_len(index, model.size))
                .sum()
        };
        progress
            .0
            .downloaded_bytes
            .store(completed_bytes, Ordering::Release);
        progress.set_phase(DownloadPhase::Downloading);

        let pending = {
            let state = state.lock().await;
            state
                .completed
                .iter()
                .enumerate()
                .filter_map(|(index, completed)| (!completed).then_some(index))
                .collect::<Vec<_>>()
        };
        let health = Arc::new(
            (0..ranged.len())
                .map(|_| AtomicU64::new(0))
                .collect::<Vec<_>>(),
        );

        let results = stream::iter(pending.into_iter().map(|chunk_index| {
            let state = Arc::clone(&state);
            let health = Arc::clone(&health);
            let temporary_path = temporary_path.clone();
            let state_path = state_path.clone();
            let progress = progress.clone();
            let cancellation = cancellation.clone();
            let ranked = ranged.clone();
            async move {
                self.download_chunk(
                    model,
                    chunk_index,
                    &temporary_path,
                    &ranked,
                    &health,
                    &cancellation,
                    &progress,
                )
                .await?;
                let mut state = state.lock().await;
                state.completed[chunk_index] = true;
                persist_state(&state_path, &state).await
            }
        }))
        .buffer_unordered(self.config.concurrency)
        .collect::<Vec<_>>()
        .await;

        for result in results {
            result?;
        }
        if cancellation.is_cancelled() {
            return Err(DownloadError::Cancelled);
        }

        progress.set_phase(DownloadPhase::Verifying);
        let temporary = tokio::fs::OpenOptions::new()
            .read(true)
            .open(&temporary_path)
            .await?;
        temporary.sync_all().await?;
        drop(temporary);
        let metadata = tokio::fs::metadata(&temporary_path).await?;
        if metadata.len() != model.size {
            return Err(DownloadError::SizeMismatch {
                expected: model.size,
                actual: metadata.len(),
            });
        }
        let actual = hash_file(temporary_path.clone()).await?;
        if actual != model.blake3 {
            return Err(DownloadError::Blake3Mismatch {
                expected: model.blake3.to_owned(),
                actual,
            });
        }

        tokio::fs::rename(&temporary_path, final_path).await?;
        remove_if_exists(&state_path).await?;
        progress
            .0
            .downloaded_bytes
            .store(model.size, Ordering::Release);
        progress.set_phase(DownloadPhase::Completed);
        Ok(final_path.to_path_buf())
    }

    async fn probe_mirrors(
        &self,
        model: &ModelEntry,
        cancellation: &CancellationToken,
    ) -> Result<Vec<RankedMirror>, DownloadError> {
        let end = model
            .size
            .saturating_sub(1)
            .min(self.config.probe_bytes.saturating_sub(1));
        let probes = model.mirrors.iter().copied().map(|mirror| async move {
            let started = Instant::now();
            let response = tokio::select! {
                () = cancellation.cancelled() => return Err(DownloadError::Cancelled),
                response = self.client.get(mirror.url).header(RANGE, format!("bytes=0-{end}")).send() => response?,
            };
            let supports_ranges = response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
            if supports_ranges {
                validate_range_response(&response, 0, end, model.size)?;
                let bytes = response.bytes().await?;
                if bytes.len() as u64 != end + 1 {
                    return Err(DownloadError::Transfer(format!(
                        "{} returned an incomplete probe",
                        mirror.name
                    )));
                }
            } else if response.status() != reqwest::StatusCode::OK {
                return Err(DownloadError::Transfer(format!(
                    "{} probe returned HTTP {}",
                    mirror.name,
                    response.status()
                )));
            }
            Ok(RankedMirror {
                mirror,
                elapsed: started.elapsed(),
                supports_ranges,
            })
        });
        let probe_results = join_all(probes).await;
        if cancellation.is_cancelled() {
            return Err(DownloadError::Cancelled);
        }
        let mut ranked = probe_results
            .into_iter()
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        ranked.sort_by_key(|probe| probe.elapsed);
        if ranked.is_empty() {
            return Err(DownloadError::NoHealthyMirrors);
        }
        Ok(ranked)
    }

    #[allow(clippy::too_many_arguments)]
    async fn download_single_stream(
        &self,
        model: &ModelEntry,
        final_path: &Path,
        temporary_path: &Path,
        state_path: &Path,
        mirrors: &[RankedMirror],
        cancellation: &CancellationToken,
        progress: &DownloadProgress,
    ) -> Result<PathBuf, DownloadError> {
        // A chunked temporary file is preallocated, so its length is not valid
        // single-stream progress. Discard only that incompatible partial state.
        if tokio::fs::try_exists(state_path).await? {
            remove_if_exists(state_path).await?;
            remove_if_exists(temporary_path).await?;
        }
        let mut offset = tokio::fs::metadata(temporary_path)
            .await
            .map(|metadata| metadata.len().min(model.size))
            .unwrap_or(0);
        progress.0.downloaded_bytes.store(offset, Ordering::Release);
        progress.set_phase(DownloadPhase::Downloading);
        let health = (0..mirrors.len())
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>();
        let mut last_error = String::new();

        for _ in 0..self.config.max_attempts_per_chunk {
            if cancellation.is_cancelled() {
                return Err(DownloadError::Cancelled);
            }
            let mirror_index = (0..mirrors.len())
                .min_by_key(|index| health[*index].load(Ordering::Acquire))
                .expect("at least one probed mirror");
            let mirror = mirrors[mirror_index].mirror;
            progress.set_mirror(mirror.name);
            let response = tokio::select! {
                () = cancellation.cancelled() => return Err(DownloadError::Cancelled),
                response = self.client.get(mirror.url).header(RANGE, format!("bytes={offset}-{}", model.size - 1)).send() => response,
            };
            let response = match response {
                Ok(response) => response,
                Err(error) => {
                    last_error = error.to_string();
                    health[mirror_index].fetch_add(1, Ordering::AcqRel);
                    continue;
                }
            };

            let append = response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
            if append {
                if let Err(error) =
                    validate_range_response(&response, offset, model.size - 1, model.size)
                {
                    last_error = error.to_string();
                    health[mirror_index].fetch_add(1, Ordering::AcqRel);
                    continue;
                }
            } else if response.status() == reqwest::StatusCode::OK {
                offset = 0;
                progress.0.downloaded_bytes.store(0, Ordering::Release);
            } else {
                last_error = format!("{} returned HTTP {}", mirror.name, response.status());
                health[mirror_index].fetch_add(1, Ordering::AcqRel);
                continue;
            }

            let mut file = if append {
                tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(temporary_path)
                    .await?
            } else {
                tokio::fs::File::create(temporary_path).await?
            };
            let mut body = response.bytes_stream();
            let mut window_started = Instant::now();
            let mut window_bytes = 0_u64;
            let mut failed = false;
            while offset < model.size {
                let next = tokio::select! {
                    () = cancellation.cancelled() => return Err(DownloadError::Cancelled),
                    result = tokio::time::timeout(self.config.stall_timeout, body.next()) => result,
                };
                let next = match next {
                    Ok(next) => next,
                    Err(_) => {
                        last_error = format!("{} stalled", mirror.name);
                        failed = true;
                        break;
                    }
                };
                let Some(bytes) = next else { break };
                let bytes = match bytes {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        last_error = error.to_string();
                        failed = true;
                        break;
                    }
                };
                let remaining = model.size - offset;
                let bytes = &bytes[..bytes.len().min(remaining as usize)];
                file.write_all(bytes).await?;
                offset += bytes.len() as u64;
                window_bytes += bytes.len() as u64;
                progress
                    .0
                    .downloaded_bytes
                    .fetch_add(bytes.len() as u64, Ordering::AcqRel);
                let elapsed = window_started.elapsed();
                if elapsed >= self.config.speed_window {
                    let throughput = window_bytes as f64 / elapsed.as_secs_f64();
                    if throughput < self.config.min_throughput_bytes_per_second as f64 {
                        last_error = format!("{} dropped to {throughput:.0} B/s", mirror.name);
                        failed = true;
                        break;
                    }
                    window_started = Instant::now();
                    window_bytes = 0;
                }
            }
            file.flush().await?;
            file.sync_all().await?;
            if offset == model.size {
                break;
            }
            if !failed {
                last_error = format!("{} ended at byte {offset}", mirror.name);
            }
            health[mirror_index].fetch_add(1, Ordering::AcqRel);
        }

        if offset != model.size {
            return Err(DownloadError::Transfer(format!(
                "single-stream download stopped at {offset}/{}: {last_error}",
                model.size
            )));
        }
        progress.set_phase(DownloadPhase::Verifying);
        let actual = hash_file(temporary_path.to_path_buf()).await?;
        if actual != model.blake3 {
            return Err(DownloadError::Blake3Mismatch {
                expected: model.blake3.to_owned(),
                actual,
            });
        }
        tokio::fs::rename(temporary_path, final_path).await?;
        progress.set_phase(DownloadPhase::Completed);
        Ok(final_path.to_path_buf())
    }

    async fn download_chunk(
        &self,
        model: &ModelEntry,
        chunk_index: usize,
        temporary_path: &Path,
        mirrors: &[RankedMirror],
        health: &[AtomicU64],
        cancellation: &CancellationToken,
        progress: &DownloadProgress,
    ) -> Result<(), DownloadError> {
        let start = chunk_index as u64 * self.config.chunk_size;
        let end = (start + self.config.chunk_size - 1).min(model.size - 1);
        let mut cursor = start;
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .open(temporary_path)
            .await?;
        file.seek(std::io::SeekFrom::Start(start)).await?;
        let mut last_error = String::new();

        for _ in 0..self.config.max_attempts_per_chunk {
            if cancellation.is_cancelled() {
                return Err(DownloadError::Cancelled);
            }
            let mirror_index = (0..mirrors.len())
                .min_by_key(|index| health[*index].load(Ordering::Acquire))
                .expect("at least one probed mirror");
            let mirror = mirrors[mirror_index].mirror;
            progress.set_mirror(mirror.name);

            match self
                .transfer_range(
                    mirror,
                    cursor,
                    end,
                    model.size,
                    &mut file,
                    cancellation,
                    progress,
                )
                .await
            {
                Ok(transferred) => {
                    cursor += transferred;
                    if cursor > end {
                        file.flush().await?;
                        return Ok(());
                    }
                    last_error = format!("{} ended at byte {cursor}", mirror.name);
                }
                Err((transferred, error)) => {
                    cursor += transferred;
                    if cursor > end {
                        file.flush().await?;
                        return Ok(());
                    }
                    if matches!(error, DownloadError::Cancelled) {
                        return Err(error);
                    }
                    last_error = error.to_string();
                }
            }
            health[mirror_index].fetch_add(1, Ordering::AcqRel);
        }

        Err(DownloadError::Transfer(format!(
            "chunk {chunk_index} exhausted mirrors at byte {cursor}: {last_error}"
        )))
    }

    async fn transfer_range(
        &self,
        mirror: ModelMirror,
        start: u64,
        end: u64,
        total_size: u64,
        file: &mut tokio::fs::File,
        cancellation: &CancellationToken,
        progress: &DownloadProgress,
    ) -> Result<u64, (u64, DownloadError)> {
        let response = tokio::select! {
            () = cancellation.cancelled() => return Err((0, DownloadError::Cancelled)),
            response = self.client.get(mirror.url).header(RANGE, format!("bytes={start}-{end}")).send() => {
                response.map_err(|error| (0, DownloadError::Http(error)))?
            },
        };
        validate_range_response(&response, start, end, total_size).map_err(|error| (0, error))?;
        let mut body = response.bytes_stream();
        let mut transferred = 0_u64;
        let mut window_started = Instant::now();
        let mut window_bytes = 0_u64;

        while start + transferred <= end {
            let next = tokio::select! {
                () = cancellation.cancelled() => return Err((transferred, DownloadError::Cancelled)),
                result = tokio::time::timeout(self.config.stall_timeout, body.next()) => result,
            };
            let next = next.map_err(|_| {
                (
                    transferred,
                    DownloadError::Transfer(format!("{} stalled", mirror.name)),
                )
            })?;
            let Some(bytes) = next else {
                break;
            };
            let bytes = bytes.map_err(|error| (transferred, DownloadError::Http(error)))?;
            let remaining = (end - start + 1) - transferred;
            let bytes = &bytes[..bytes.len().min(remaining as usize)];
            file.write_all(bytes)
                .await
                .map_err(|error| (transferred, DownloadError::Io(error)))?;
            transferred += bytes.len() as u64;
            window_bytes += bytes.len() as u64;
            progress
                .0
                .downloaded_bytes
                .fetch_add(bytes.len() as u64, Ordering::AcqRel);

            let elapsed = window_started.elapsed();
            if elapsed >= self.config.speed_window {
                let throughput = window_bytes as f64 / elapsed.as_secs_f64();
                if throughput < self.config.min_throughput_bytes_per_second as f64 {
                    return Err((
                        transferred,
                        DownloadError::Transfer(format!(
                            "{} dropped to {throughput:.0} B/s",
                            mirror.name
                        )),
                    ));
                }
                window_started = Instant::now();
                window_bytes = 0;
            }
        }
        Ok(transferred)
    }

    async fn load_or_create_state(
        &self,
        model: &ModelEntry,
        temporary_path: &Path,
        state_path: &Path,
        chunk_count: usize,
    ) -> Result<ResumeState, DownloadError> {
        if let (Ok(bytes), Ok(metadata)) = (
            tokio::fs::read(state_path).await,
            tokio::fs::metadata(temporary_path).await,
        ) && let Ok(state) = serde_json::from_slice::<ResumeState>(&bytes)
            && state.version == RESUME_VERSION
            && state.size == model.size
            && state.chunk_size == self.config.chunk_size
            && state.blake3 == model.blake3
            && state.completed.len() == chunk_count
            && metadata.len() == model.size
        {
            return Ok(state);
        }

        let file = tokio::fs::File::create(temporary_path).await?;
        file.set_len(model.size).await?;
        let state = ResumeState {
            version: RESUME_VERSION,
            size: model.size,
            chunk_size: self.config.chunk_size,
            blake3: model.blake3.to_owned(),
            completed: vec![false; chunk_count],
        };
        persist_state(state_path, &state).await?;
        Ok(state)
    }

    fn chunk_len(&self, chunk_index: usize, total_size: u64) -> u64 {
        let start = chunk_index as u64 * self.config.chunk_size;
        (total_size - start).min(self.config.chunk_size)
    }

    async fn valid_file(&self, path: &Path, model: &ModelEntry) -> Result<bool, DownloadError> {
        let Ok(metadata) = tokio::fs::metadata(path).await else {
            return Ok(false);
        };
        if metadata.len() != model.size {
            return Ok(false);
        }
        Ok(hash_file(path.to_path_buf()).await? == model.blake3)
    }
}

fn validate_range_response(
    response: &reqwest::Response,
    start: u64,
    end: u64,
    total_size: u64,
) -> Result<(), DownloadError> {
    if response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(DownloadError::Transfer(format!(
            "expected HTTP 206, got {}",
            response.status()
        )));
    }
    let expected = format!("bytes {start}-{end}/{total_size}");
    let actual = response
        .headers()
        .get(CONTENT_RANGE)
        .and_then(|value| value.to_str().ok());
    if actual != Some(expected.as_str()) {
        return Err(DownloadError::Transfer(format!(
            "invalid Content-Range: expected {expected}, got {actual:?}"
        )));
    }
    Ok(())
}

async fn hash_file(path: PathBuf) -> Result<String, DownloadError> {
    tokio::task::spawn_blocking(move || file_blake3(&path).map_err(DownloadError::Io)).await?
}

async fn persist_state(path: &Path, state: &ResumeState) -> Result<(), DownloadError> {
    let temporary = path.with_extension("state.new");
    let bytes = serde_json::to_vec(state)?;
    let mut file = tokio::fs::File::create(&temporary).await?;
    file.write_all(&bytes).await?;
    file.sync_all().await?;
    drop(file);
    #[cfg(target_os = "windows")]
    remove_if_exists(path).await?;
    tokio::fs::rename(temporary, path).await?;
    Ok(())
}

async fn remove_if_exists(path: &Path) -> Result<(), std::io::Error> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}
