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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchOptions {
    pub mode: LaunchMode,
    pub models_dir: Option<PathBuf>,
    pub help: bool,
}

impl LaunchOptions {
    pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Self, LaunchModeError> {
        let mut arguments = args.into_iter();
        let mut mode: Option<LaunchMode> = None;
        let mut models_dir = None;
        let mut help = false;
        while let Some(argument) = arguments.next() {
            if argument == "--help" || argument == "-h" {
                help = true;
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
        Ok(Self {
            mode: mode.unwrap_or(LaunchMode::Realtime),
            models_dir,
            help,
        })
    }
}

pub const fn help_text() -> &'static str {
    concat!(
        "noperson — GPU-accelerated face swap\n",
        "\n",
        "Usage: noperson [OPTIONS]\n",
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
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{LaunchMode, LaunchModeError, LaunchOptions};

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
