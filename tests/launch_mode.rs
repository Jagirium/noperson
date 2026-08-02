use std::ffi::OsString;

use noperson::launch::{LaunchMode, LaunchModeError};

fn parse(args: &[&str]) -> Result<LaunchMode, LaunchModeError> {
    LaunchMode::parse(args.iter().map(OsString::from))
}

#[test]
fn realtime_is_the_default_and_has_an_explicit_alias() {
    assert_eq!(parse(&[]), Ok(LaunchMode::Realtime));
    assert_eq!(parse(&["--realtime"]), Ok(LaunchMode::Realtime));
}

#[test]
fn extra_gui_is_selected_only_at_process_start() {
    assert_eq!(parse(&["--extra-gui"]), Ok(LaunchMode::ExtraGui));
}

#[test]
fn runtime_check_remains_a_separate_non_gui_mode() {
    assert_eq!(parse(&["--runtime-check"]), Ok(LaunchMode::RuntimeCheck));
}

#[test]
fn conflicting_or_unknown_modes_are_rejected() {
    assert!(matches!(
        parse(&["--realtime", "--extra-gui"]),
        Err(LaunchModeError::ConflictingModes { .. })
    ));
    assert_eq!(
        parse(&["--editor"]),
        Err(LaunchModeError::UnknownArgument("--editor".to_owned()))
    );
}
