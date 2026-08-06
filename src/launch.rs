//! Process-level frontend selection.

use std::ffi::OsString;
use std::path::PathBuf;

use thiserror::Error;

/// The one frontend selected before any GUI state is constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchMode {
    Realtime,
    ExtraGui,
    RuntimeCheck,
    Headless,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PredecodeMode {
    #[default]
    Auto,
    Full,
    Off,
}

impl LaunchMode {
    pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self, LaunchModeError> {
        Ok(LaunchOptions::parse(args)?.mode)
    }

    pub const fn flag(self) -> &'static str {
        match self {
            Self::Realtime => "--realtime",
            Self::ExtraGui => "--extra-gui",
            Self::RuntimeCheck => "--runtime-check",
            Self::Headless => "headless-run",
        }
    }
}

/// File-processing controls shared by image and video headless workflows.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadlessOptions {
    pub source_path: PathBuf,
    pub target_path: PathBuf,
    pub output_path: PathBuf,
    pub execution_provider: String,
    pub device_id: i32,
    pub swap_resolution: u32,
    pub face_detector_score: f32,
    pub max_faces: usize,
    pub worker_threads: usize,
    pub predecode: PredecodeMode,
}

impl Default for HeadlessOptions {
    fn default() -> Self {
        Self {
            source_path: PathBuf::new(),
            target_path: PathBuf::new(),
            output_path: PathBuf::new(),
            execution_provider: "cuda".to_owned(),
            device_id: 0,
            swap_resolution: 128,
            face_detector_score: 0.5,
            max_faces: 20,
            worker_threads: std::thread::available_parallelism().map_or(1, usize::from),
            predecode: PredecodeMode::Auto,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LaunchOptions {
    pub mode: LaunchMode,
    pub models_dir: Option<PathBuf>,
    pub headless: Option<HeadlessOptions>,
    pub help: bool,
}

impl LaunchOptions {
    pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self, LaunchModeError> {
        let mut arguments = args.into_iter();
        let mut mode: Option<LaunchMode> = None;
        let mut models_dir = None;
        let mut headless = None;
        let mut help = false;
        while let Some(argument) = arguments.next() {
            if argument == "--help" || argument == "-h" {
                help = true;
                continue;
            }
            if argument == "headless-run" {
                if let Some(previous) = mode
                    && previous != LaunchMode::Headless
                {
                    return Err(LaunchModeError::ConflictingModes {
                        first: previous.flag(),
                        second: LaunchMode::Headless.flag(),
                    });
                }
                mode = Some(LaunchMode::Headless);
                headless.get_or_insert_with(HeadlessOptions::default);
                continue;
            }
            if argument == "--models-dir" {
                let path = arguments
                    .next()
                    .ok_or(LaunchModeError::MissingModelsDirectory)?;
                if models_dir.replace(PathBuf::from(path)).is_some() {
                    return Err(LaunchModeError::DuplicateModelsDirectory);
                }
                continue;
            }
            if mode == Some(LaunchMode::Headless) {
                let option = argument
                    .into_string()
                    .map_err(|_| LaunchModeError::NonUnicodeArgument)?;
                let mut value = || {
                    arguments
                        .next()
                        .ok_or_else(|| LaunchModeError::MissingOptionValue(option.clone()))
                };
                let options = headless
                    .as_mut()
                    .expect("headless options exist after selecting headless-run");
                match option.as_str() {
                    "-s" | "--source-path" | "--source-paths" => {
                        options.source_path = PathBuf::from(value()?);
                    }
                    "-t" | "--target-path" => options.target_path = PathBuf::from(value()?),
                    "-o" | "--output-path" => options.output_path = PathBuf::from(value()?),
                    "--execution-provider" => {
                        let provider = os_value(&option, value()?)?;
                        if !matches!(provider.as_str(), "cuda" | "tensorrt") {
                            return Err(LaunchModeError::InvalidOptionValue {
                                option,
                                value: provider,
                            });
                        }
                        options.execution_provider = provider;
                    }
                    "--device-id" => options.device_id = parse_value(&option, value()?)?,
                    "--swap-resolution" => {
                        let resolution = parse_value(&option, value()?)?;
                        if !matches!(resolution, 128 | 256 | 384 | 512) {
                            return Err(LaunchModeError::InvalidOptionValue {
                                option,
                                value: resolution.to_string(),
                            });
                        }
                        options.swap_resolution = resolution;
                    }
                    "--face-detector-score" => {
                        let score: f32 = parse_value(&option, value()?)?;
                        if !(0.0..=1.0).contains(&score) || !score.is_finite() {
                            return Err(LaunchModeError::InvalidOptionValue {
                                option,
                                value: score.to_string(),
                            });
                        }
                        options.face_detector_score = score;
                    }
                    "--max-faces" => {
                        let count: usize = parse_value(&option, value()?)?;
                        if !(1..=50).contains(&count) {
                            return Err(LaunchModeError::InvalidOptionValue {
                                option,
                                value: count.to_string(),
                            });
                        }
                        options.max_faces = count;
                    }
                    "--worker-threads" => {
                        let count: usize = parse_value(&option, value()?)?;
                        if !(1..=32).contains(&count) {
                            return Err(LaunchModeError::InvalidOptionValue {
                                option,
                                value: count.to_string(),
                            });
                        }
                        options.worker_threads = count;
                    }
                    "--predecode" => {
                        let mode = os_value(&option, value()?)?;
                        options.predecode = match mode.as_str() {
                            "auto" => PredecodeMode::Auto,
                            "full" => PredecodeMode::Full,
                            "off" => PredecodeMode::Off,
                            _ => {
                                return Err(LaunchModeError::InvalidOptionValue {
                                    option,
                                    value: mode,
                                });
                            }
                        };
                    }
                    _ => return Err(LaunchModeError::UnknownArgument(option)),
                }
                continue;
            }
            let argument = argument
                .into_string()
                .map_err(|_| LaunchModeError::NonUnicodeArgument)?;
            let requested = match argument.as_str() {
                "--realtime" => LaunchMode::Realtime,
                "--extra-gui" => LaunchMode::ExtraGui,
                "--runtime-check" => LaunchMode::RuntimeCheck,
                _ => return Err(LaunchModeError::UnknownArgument(argument)),
            };
            if let Some(previous) = mode
                && previous != requested
            {
                return Err(LaunchModeError::ConflictingModes {
                    first: previous.flag(),
                    second: requested.flag(),
                });
            }
            mode = Some(requested);
        }
        let mode = mode.unwrap_or(LaunchMode::Realtime);
        if let Some(options) = &headless {
            for (path, option) in [
                (&options.source_path, "--source-path"),
                (&options.target_path, "--target-path"),
                (&options.output_path, "--output-path"),
            ] {
                if path.as_os_str().is_empty() && !help {
                    return Err(LaunchModeError::MissingHeadlessPath(option));
                }
            }
        }
        Ok(Self {
            mode,
            models_dir,
            headless,
            help,
        })
    }
}

fn os_value(option: &str, value: OsString) -> Result<String, LaunchModeError> {
    value
        .into_string()
        .map_err(|_| LaunchModeError::InvalidOptionValue {
            option: option.to_owned(),
            value: "<non-unicode>".to_owned(),
        })
}

fn parse_value<T>(option: &str, value: OsString) -> Result<T, LaunchModeError>
where
    T: std::str::FromStr,
{
    let value = os_value(option, value)?;
    value
        .parse()
        .map_err(|_| LaunchModeError::InvalidOptionValue {
            option: option.to_owned(),
            value,
        })
}

pub const fn help_text() -> &'static str {
    concat!(
        "noperson — GPU-accelerated face swap\n",
        "\n",
        "Usage: noperson [OPTIONS]\n",
        "\n",
        "Commands:\n",
        "  headless-run              Process an image or video without a GUI\n",
        "\n",
        "Options:\n",
        "      --realtime            Launch the realtime UI (default)\n",
        "      --extra-gui           Launch the advanced editor UI\n",
        "      --runtime-check       Validate the CUDA and TensorRT runtime\n",
        "      --models-dir <PATH>   Store and load models from PATH\n",
        "  -h, --help                Print help\n",
        "\n",
        "Environment:\n",
        "  NOPERSON_MODELS_DIR       Default model directory when --models-dir is absent\n",
    )
}

pub const fn headless_help_text() -> &'static str {
    concat!(
        "noperson — GPU-accelerated face swap\n",
        "\n",
        "Usage: noperson headless-run [OPTIONS]\n",
        "\n",
        "Paths:\n",
        "  -s, --source-path <PATH>          Source face image\n",
        "  -t, --target-path <PATH>          Target image or video\n",
        "  -o, --output-path <PATH>          Output image or video\n",
        "      --models-dir <PATH>           Store and load models from PATH\n",
        "\n",
        "Execution:\n",
        "      --execution-provider <NAME>  cuda (default) or tensorrt\n",
        "      --device-id <ID>              CUDA device index [default: 0]\n",
        "      --worker-threads <COUNT>      Decode/encode workers [1..32]\n",
        "      --predecode <MODE>            auto (default), full, or off\n",
        "\n",
        "Quality:\n",
        "      --swap-resolution <PX>       128, 256, 384, or 512 [default: 128]\n",
        "      --face-detector-score <N>    Detection threshold [default: 0.5]\n",
        "      --max-faces <COUNT>           Faces per frame [default: 20]\n",
        "  -h, --help                        Print help\n",
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LaunchModeError {
    #[error(
        "unknown argument {0}; expected --realtime, --extra-gui, --runtime-check, or --models-dir PATH"
    )]
    UnknownArgument(String),
    #[error("frontend modes conflict: {first} and {second}")]
    ConflictingModes {
        first: &'static str,
        second: &'static str,
    },
    #[error("command-line arguments must be valid Unicode")]
    NonUnicodeArgument,
    #[error("--models-dir requires a directory path")]
    MissingModelsDirectory,
    #[error("--models-dir may only be specified once")]
    DuplicateModelsDirectory,
    #[error("{0} is required for headless-run")]
    MissingHeadlessPath(&'static str),
    #[error("{0} requires a value")]
    MissingOptionValue(String),
    #[error("invalid value {value} for {option}")]
    InvalidOptionValue { option: String, value: String },
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{HeadlessOptions, LaunchMode, LaunchModeError, LaunchOptions, PredecodeMode};

    #[test]
    fn headless_run_accepts_facefusion_path_contract() {
        let options = LaunchOptions::parse([
            "headless-run".into(),
            "-s".into(),
            "face.jpg".into(),
            "-t".into(),
            "input.mp4".into(),
            "-o".into(),
            "output.mp4".into(),
            "--models-dir".into(),
            "/portable/models".into(),
        ])
        .unwrap();

        assert_eq!(options.mode, LaunchMode::Headless);
        assert_eq!(
            options.headless,
            Some(HeadlessOptions {
                source_path: PathBuf::from("face.jpg"),
                target_path: PathBuf::from("input.mp4"),
                output_path: PathBuf::from("output.mp4"),
                ..HeadlessOptions::default()
            })
        );
        assert_eq!(options.models_dir, Some(PathBuf::from("/portable/models")));
    }

    #[test]
    fn headless_run_requires_every_media_path() {
        assert_eq!(
            LaunchOptions::parse([
                "headless-run".into(),
                "--source-path".into(),
                "face.jpg".into(),
                "--target-path".into(),
                "input.png".into(),
            ])
            .unwrap_err(),
            LaunchModeError::MissingHeadlessPath("--output-path")
        );
    }

    #[test]
    fn headless_run_parses_quality_and_execution_controls() {
        let options = LaunchOptions::parse([
            "headless-run".into(),
            "-s".into(),
            "face.jpg".into(),
            "-t".into(),
            "input.png".into(),
            "-o".into(),
            "output.png".into(),
            "--execution-provider".into(),
            "tensorrt".into(),
            "--device-id".into(),
            "1".into(),
            "--swap-resolution".into(),
            "512".into(),
            "--face-detector-score".into(),
            "0.72".into(),
            "--max-faces".into(),
            "3".into(),
            "--worker-threads".into(),
            "8".into(),
            "--predecode".into(),
            "full".into(),
        ])
        .unwrap();
        let headless = options.headless.unwrap();

        assert_eq!(headless.execution_provider, "tensorrt");
        assert_eq!(headless.device_id, 1);
        assert_eq!(headless.swap_resolution, 512);
        assert_eq!(headless.face_detector_score, 0.72);
        assert_eq!(headless.max_faces, 3);
        assert_eq!(headless.worker_threads, 8);
        assert_eq!(headless.predecode, PredecodeMode::Full);
    }

    #[test]
    fn launch_options_accept_an_explicit_models_directory_for_every_frontend() {
        let options = LaunchOptions::parse([
            "--extra-gui".into(),
            "--models-dir".into(),
            "/portable/models".into(),
        ])
        .unwrap();

        assert_eq!(options.mode, LaunchMode::ExtraGui);
        assert_eq!(options.models_dir, Some(PathBuf::from("/portable/models")));
    }

    #[test]
    fn launch_options_preserve_the_realtime_and_models_defaults() {
        let options = LaunchOptions::parse(Vec::<std::ffi::OsString>::new()).unwrap();
        assert_eq!(options.mode, LaunchMode::Realtime);
        assert_eq!(options.models_dir, None);
        assert_eq!(options.headless, None);
        assert!(!options.help);
    }

    #[test]
    fn launch_options_reject_a_missing_models_directory_value() {
        assert_eq!(
            LaunchOptions::parse(["--models-dir".into()]).unwrap_err(),
            LaunchModeError::MissingModelsDirectory
        );
    }
}
