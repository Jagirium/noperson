use std::env;
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use cudarc::driver::CudaContext;

use crate::gpu::npp;

use super::{ComputeCapability, RuntimeLayout, ensure_runtime};

const READY_ENV: &str = "NOPERSON_RUNTIME_READY";

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BootstrapOutcome {
    Ready(RuntimeLayout),
    Reexec(RuntimeLayout),
}

pub fn prepare() -> anyhow::Result<BootstrapOutcome> {
    let capability = detect_compute_capability()?;
    let shard = capability.tensorrt_shard();

    if let Some(root) = env::var_os(READY_ENV) {
        let layout = RuntimeLayout::new(PathBuf::from(root), shard);
        anyhow::ensure!(layout.is_complete(), "activated GPU runtime is incomplete");
        npp::initialize_runtime(&layout.base())?;
        return Ok(BootstrapOutcome::Ready(layout));
    }

    for root in local_runtime_candidates()? {
        let layout = RuntimeLayout::new(root, shard);
        if layout.is_complete() {
            return Ok(BootstrapOutcome::Reexec(layout));
        }
    }

    let runtime_root = persistent_runtime_root()?;
    let tokio = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let layout = tokio.block_on(ensure_runtime(&runtime_root, shard))?;
    Ok(BootstrapOutcome::Reexec(layout))
}

pub fn reexec(layout: &RuntimeLayout) -> io::Error {
    let source_executable = match env::current_exe() {
        Ok(path) => path,
        Err(error) => return error,
    };
    let executable = match stage_launch_directory(layout, &source_executable) {
        Ok(path) => path,
        Err(error) => return error,
    };
    let mut command = Command::new(executable);
    command.args(env::args_os().skip(1));
    command.env(READY_ENV, layout.root());

    #[cfg(target_os = "windows")]
    const LIBRARY_ENV: &str = "PATH";
    #[cfg(not(target_os = "windows"))]
    const LIBRARY_ENV: &str = "LD_LIBRARY_PATH";

    let mut paths = layout.library_paths();
    if let Some(existing) = env::var_os(LIBRARY_ENV) {
        paths.extend(env::split_paths(&existing));
    }
    let joined = match env::join_paths(paths) {
        Ok(value) => value,
        Err(error) => return io::Error::new(io::ErrorKind::InvalidInput, error),
    };
    command.env(LIBRARY_ENV, joined);

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.exec()
    }
    #[cfg(windows)]
    {
        match command.status() {
            Ok(status) => std::process::exit(status.code().unwrap_or(1)),
            Err(error) => error,
        }
    }
}

fn stage_launch_directory(layout: &RuntimeLayout, executable: &Path) -> io::Result<PathBuf> {
    let launch = layout.launch_dir();
    std::fs::create_dir_all(&launch)?;

    for provider in [
        layout.ort_shared_provider(),
        layout.ort_cuda_provider(),
        layout.ort_tensorrt_provider(),
    ] {
        let filename = provider.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "provider path has no filename")
        })?;
        materialize_atomically(&provider, &launch.join(filename))?;
    }

    #[cfg(windows)]
    const LAUNCHER_NAME: &str = "noperson.exe";
    #[cfg(not(windows))]
    const LAUNCHER_NAME: &str = "noperson";
    let staged_executable = launch.join(LAUNCHER_NAME);
    materialize_atomically(executable, &staged_executable)?;
    Ok(staged_executable)
}

fn materialize_atomically(source: &Path, destination: &Path) -> io::Result<()> {
    if source == destination {
        return Ok(());
    }
    let temporary = destination.with_extension(format!("tmp-{}", std::process::id()));
    remove_file_if_exists(&temporary)?;
    if std::fs::hard_link(source, &temporary).is_err() {
        std::fs::copy(source, &temporary)?;
    }
    #[cfg(windows)]
    remove_file_if_exists(destination)?;
    std::fs::rename(temporary, destination)
}

fn remove_file_if_exists(path: &Path) -> io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn detect_compute_capability() -> anyhow::Result<ComputeCapability> {
    let context = CudaContext::new(0)?;
    let (major, minor) = context.compute_capability()?;
    Ok(ComputeCapability { major, minor })
}

fn local_runtime_candidates() -> anyhow::Result<Vec<PathBuf>> {
    let mut roots = Vec::new();
    if let Some(override_root) = env::var_os("NOPERSON_RUNTIME_DIR") {
        push_unique(&mut roots, PathBuf::from(override_root));
    }
    push_unique(&mut roots, env::current_dir()?.join("libs"));
    if let Some(parent) = env::current_exe()?.parent() {
        push_unique(&mut roots, parent.join("libs"));
    }
    Ok(roots)
}

fn push_unique(paths: &mut Vec<PathBuf>, candidate: PathBuf) {
    if !paths.contains(&candidate) {
        paths.push(candidate);
    }
}

fn persistent_runtime_root() -> anyhow::Result<PathBuf> {
    if let Some(root) = env::var_os("NOPERSON_RUNTIME_DIR") {
        return Ok(PathBuf::from(root));
    }
    #[cfg(target_os = "windows")]
    let base = required_env_path("LOCALAPPDATA")?;
    #[cfg(not(target_os = "windows"))]
    let base = match env::var_os("XDG_DATA_HOME") {
        Some(path) => PathBuf::from(path),
        None => required_env_path("HOME")?.join(".local/share"),
    };
    Ok(base.join("noperson/runtime"))
}

fn required_env_path(name: &str) -> anyhow::Result<PathBuf> {
    env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("{name} is not set"))
}

#[allow(dead_code)]
fn prepend_library_path(layout: &RuntimeLayout, existing: Option<&Path>) -> OsString {
    let mut paths = layout.library_paths();
    if let Some(existing) = existing {
        paths.push(existing.to_path_buf());
    }
    env::join_paths(paths).expect("runtime paths do not contain a path separator")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::TensorRtShard;

    #[test]
    fn launch_directory_places_the_binary_beside_every_ort_provider() {
        let fixture = tempfile::tempdir().unwrap();
        let layout = RuntimeLayout::new(fixture.path().join("runtime"), TensorRtShard::Sm86);
        for provider in [
            layout.ort_shared_provider(),
            layout.ort_cuda_provider(),
            layout.ort_tensorrt_provider(),
        ] {
            std::fs::create_dir_all(provider.parent().unwrap()).unwrap();
            std::fs::write(provider, b"provider").unwrap();
        }
        let executable = fixture.path().join("source-noperson");
        std::fs::write(&executable, b"executable").unwrap();

        let staged = stage_launch_directory(&layout, &executable).unwrap();
        assert_eq!(std::fs::read(staged).unwrap(), b"executable");
        for provider in [
            layout.ort_shared_provider(),
            layout.ort_cuda_provider(),
            layout.ort_tensorrt_provider(),
        ] {
            assert_eq!(
                std::fs::read(layout.launch_dir().join(provider.file_name().unwrap())).unwrap(),
                b"provider"
            );
        }
    }
}
