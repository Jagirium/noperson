//! Process-level frontend selection.

use std::ffi::OsString;

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
        let mut selected: Option<Self> = None;
        for argument in args {
            let argument = argument
                .into_string()
                .map_err(|_| LaunchModeError::NonUnicodeArgument)?;
            let mode = match argument.as_str() {
                "--realtime" => Self::Realtime,
                "--extra-gui" => Self::ExtraGui,
                "--runtime-check" => Self::RuntimeCheck,
                _ => return Err(LaunchModeError::UnknownArgument(argument)),
            };
            if let Some(previous) = selected
                && previous != mode
            {
                return Err(LaunchModeError::ConflictingModes {
                    first: previous.flag(),
                    second: mode.flag(),
                });
            }
            selected = Some(mode);
        }
        Ok(selected.unwrap_or(Self::Realtime))
    }

    pub const fn flag(self) -> &'static str {
        match self {
            Self::Realtime => "--realtime",
            Self::ExtraGui => "--extra-gui",
            Self::RuntimeCheck => "--runtime-check",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LaunchModeError {
    #[error("unknown argument {0}; expected --realtime, --extra-gui, or --runtime-check")]
    UnknownArgument(String),
    #[error("frontend modes conflict: {first} and {second}")]
    ConflictingModes {
        first: &'static str,
        second: &'static str,
    },
    #[error("command-line arguments must be valid Unicode")]
    NonUnicodeArgument,
}
