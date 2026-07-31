use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use noperson::models::download::{
    DownloadConfig, DownloadError, DownloadPhase, DownloadProgress, ModelDownloader,
};
use noperson::models::registry::{ModelEntry, ModelMirror};
use tokio_util::sync::CancellationToken;

struct RangeServer {
    url: &'static str,
    ranges: Arc<Mutex<Vec<(u64, u64)>>>,
}

impl RangeServer {
    fn start(data: Arc<Vec<u8>>, fail_large_after: Option<usize>, delay: Duration) -> Self {
        Self::start_with_ranges(data, fail_large_after, delay, true)
    }

    fn start_without_ranges(data: Arc<Vec<u8>>, delay: Duration) -> Self {
        Self::start_with_ranges(data, None, delay, false)
    }

    fn start_with_ranges(
        data: Arc<Vec<u8>>,
        fail_large_after: Option<usize>,
        delay: Duration,
        supports_ranges: bool,
    ) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fixture");
        let address = listener.local_addr().expect("fixture address");
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&ranges);

        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                let data = Arc::clone(&data);
                let seen = Arc::clone(&seen);
                thread::spawn(move || {
                    serve(
                        stream,
                        &data,
                        &seen,
                        fail_large_after,
                        delay,
                        supports_ranges,
                    )
                });
            }
        });

        Self {
            url: Box::leak(format!("http://{address}/model.onnx").into_boxed_str()),
            ranges,
        }
    }
}

fn serve(
    mut stream: TcpStream,
    data: &[u8],
    seen: &Mutex<Vec<(u64, u64)>>,
    fail_large_after: Option<usize>,
    delay: Duration,
    supports_ranges: bool,
) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone fixture stream"));
    let mut range = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
            break;
        }
        if line.to_ascii_lowercase().starts_with("range: bytes=") {
            let value = line.split_once('=').expect("range separator").1.trim();
            let (start, end) = value.split_once('-').expect("valid test range");
            range = Some((
                start.parse::<u64>().expect("range start"),
                end.parse::<u64>().expect("range end"),
            ));
        }
    }

    let (requested_start, requested_end) = range.unwrap_or((0, data.len() as u64 - 1));
    let (start, end) = if supports_ranges {
        (requested_start, requested_end)
    } else {
        (0, data.len() as u64 - 1)
    };
    seen.lock().expect("range lock").push((start, end));
    let body = &data[start as usize..=end.min(data.len() as u64 - 1) as usize];
    thread::sleep(delay);
    if supports_ranges {
        write!(
            stream,
            "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {start}-{end}/{}\r\nConnection: close\r\n\r\n",
            body.len(),
            data.len()
        )
        .expect("fixture headers");
    } else {
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .expect("fixture headers");
    }

    let write_len = fail_large_after
        .filter(|_| body.len() > 16 * 1024)
        .map_or(body.len(), |limit| limit.min(body.len()));
    stream.write_all(&body[..write_len]).expect("fixture body");
}

fn fixture_model(primary: &'static str, fallback: &'static str, data: &[u8]) -> ModelEntry {
    ModelEntry {
        name: "fixture",
        filename: "model.onnx",
        size: data.len() as u64,
        blake3: Box::leak(blake3::hash(data).to_hex().to_string().into_boxed_str()),
        mirrors: [
            ModelMirror {
                name: "primary",
                url: primary,
            },
            ModelMirror {
                name: "fallback",
                url: fallback,
            },
        ],
    }
}

#[tokio::test]
async fn failed_chunk_resumes_from_exact_offset_on_fallback() {
    let data = Arc::new((0..256 * 1024).map(|index| (index % 251) as u8).collect());
    let primary = RangeServer::start(Arc::clone(&data), Some(20 * 1024), Duration::ZERO);
    let fallback = RangeServer::start(Arc::clone(&data), None, Duration::from_millis(15));
    let model = fixture_model(primary.url, fallback.url, &data);
    let directory = tempfile::tempdir().expect("temp model directory");
    let progress = DownloadProgress::default();
    let downloader = ModelDownloader::new(DownloadConfig {
        chunk_size: 64 * 1024,
        concurrency: 2,
        probe_bytes: 8 * 1024,
        min_throughput_bytes_per_second: 0,
        ..DownloadConfig::default()
    })
    .expect("valid config");

    let path = downloader
        .download(
            &model,
            directory.path(),
            CancellationToken::new(),
            progress.clone(),
        )
        .await
        .expect("fallback download");

    assert_eq!(std::fs::read(path).expect("final model"), *data);
    assert_eq!(progress.snapshot().phase, DownloadPhase::Completed);
    assert_eq!(progress.snapshot().downloaded_bytes, data.len() as u64);
    assert!(!directory.path().join("model.onnx.tmp").exists());
    assert!(!directory.path().join("model.onnx.state").exists());

    let fallback_ranges = fallback.ranges.lock().expect("fallback ranges");
    assert!(
        fallback_ranges
            .iter()
            .any(|(start, end)| start % (64 * 1024) != 0 && end - start > 8 * 1024),
        "fallback must continue a partial chunk instead of restarting it: {fallback_ranges:?}"
    );
}

#[tokio::test]
async fn completed_chunks_survive_cancellation_and_restart() {
    let data = Arc::new((0..512 * 1024).map(|index| (index % 239) as u8).collect());
    let server = RangeServer::start(Arc::clone(&data), None, Duration::from_millis(10));
    let model = fixture_model(server.url, server.url, &data);
    let directory = tempfile::tempdir().expect("temp model directory");
    let progress = DownloadProgress::default();
    let cancellation = CancellationToken::new();
    let downloader = ModelDownloader::new(DownloadConfig {
        chunk_size: 64 * 1024,
        concurrency: 1,
        probe_bytes: 8 * 1024,
        min_throughput_bytes_per_second: 0,
        ..DownloadConfig::default()
    })
    .expect("valid config");

    let task = tokio::spawn({
        let downloader = downloader.clone();
        let cancellation = cancellation.clone();
        let progress = progress.clone();
        let path = directory.path().to_path_buf();
        async move {
            downloader
                .download(&model, &path, cancellation, progress)
                .await
        }
    });
    while progress.snapshot().downloaded_bytes < 64 * 1024 {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    cancellation.cancel();
    assert!(task.await.expect("download task").is_err());
    assert!(directory.path().join("model.onnx.state").exists());

    server.ranges.lock().expect("range lock").clear();
    let path = downloader
        .download(
            &model,
            directory.path(),
            CancellationToken::new(),
            DownloadProgress::default(),
        )
        .await
        .expect("resumed download");
    assert_eq!(std::fs::read(path).expect("final model"), *data);
    assert!(
        !server
            .ranges
            .lock()
            .expect("range lock")
            .contains(&(0, 64 * 1024 - 1)),
        "the completed first chunk must not be requested again"
    );
}

#[tokio::test]
async fn falls_back_to_single_stream_when_ranges_are_ignored() {
    let data = Arc::new((0..96 * 1024).map(|index| (index % 233) as u8).collect());
    let primary = RangeServer::start_without_ranges(Arc::clone(&data), Duration::ZERO);
    let fallback = RangeServer::start_without_ranges(Arc::clone(&data), Duration::from_millis(10));
    let model = fixture_model(primary.url, fallback.url, &data);
    let directory = tempfile::tempdir().expect("temp model directory");
    let progress = DownloadProgress::default();
    let downloader = ModelDownloader::new(DownloadConfig {
        chunk_size: 32 * 1024,
        concurrency: 2,
        probe_bytes: 8 * 1024,
        min_throughput_bytes_per_second: 0,
        ..DownloadConfig::default()
    })
    .expect("valid config");

    let path = downloader
        .download(
            &model,
            directory.path(),
            CancellationToken::new(),
            progress.clone(),
        )
        .await
        .expect("single-stream download");

    assert_eq!(std::fs::read(path).expect("final model"), *data);
    assert_eq!(progress.snapshot().phase, DownloadPhase::Completed);
}

#[tokio::test]
async fn probe_routes_chunks_to_the_fastest_mirror() {
    let data = Arc::new((0..128 * 1024).map(|index| (index % 229) as u8).collect());
    let slow = RangeServer::start(Arc::clone(&data), None, Duration::from_millis(30));
    let fast = RangeServer::start(Arc::clone(&data), None, Duration::ZERO);
    let model = fixture_model(slow.url, fast.url, &data);
    let directory = tempfile::tempdir().expect("temp model directory");
    let downloader = ModelDownloader::new(DownloadConfig {
        chunk_size: 64 * 1024,
        concurrency: 2,
        probe_bytes: 8 * 1024,
        min_throughput_bytes_per_second: 0,
        ..DownloadConfig::default()
    })
    .expect("valid config");

    downloader
        .download(
            &model,
            directory.path(),
            CancellationToken::new(),
            DownloadProgress::default(),
        )
        .await
        .expect("fastest mirror download");

    assert_eq!(
        *slow.ranges.lock().expect("slow ranges"),
        vec![(0, 8 * 1024 - 1)],
        "slow mirror should only receive the probe"
    );
    assert!(
        fast.ranges.lock().expect("fast ranges").len() >= 3,
        "fast mirror should receive one probe and both chunks"
    );
}

#[test]
fn progress_snapshot_is_idle_before_download() {
    let snapshot = DownloadProgress::default().snapshot();
    assert_eq!(snapshot.phase, DownloadPhase::Idle);
    assert_eq!(snapshot.downloaded_bytes, 0);
    assert_eq!(snapshot.total_bytes, 0);
}

#[tokio::test]
async fn lifecycle_entrypoint_rejects_unknown_logical_model() {
    let downloader = ModelDownloader::new(DownloadConfig::default()).expect("valid config");
    let directory = tempfile::tempdir().expect("temp model directory");
    let error = downloader
        .download_registered(
            "not-a-model",
            directory.path(),
            CancellationToken::new(),
            DownloadProgress::default(),
        )
        .await
        .expect_err("unknown model must fail before networking");
    assert!(matches!(error, DownloadError::UnknownModel(name) if name == "not-a-model"));
}

#[tokio::test]
async fn cancellation_is_preserved_as_a_terminal_state() {
    let data = vec![1_u8; 1024];
    let model = fixture_model(
        "http://127.0.0.1:1/model.onnx",
        "http://127.0.0.1:2/model.onnx",
        &data,
    );
    let directory = tempfile::tempdir().expect("temp model directory");
    let progress = DownloadProgress::default();
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let error = ModelDownloader::new(DownloadConfig::default())
        .expect("valid config")
        .download(&model, directory.path(), cancellation, progress.clone())
        .await
        .expect_err("cancelled download");

    assert!(matches!(error, DownloadError::Cancelled));
    assert_eq!(progress.snapshot().phase, DownloadPhase::Cancelled);
}
