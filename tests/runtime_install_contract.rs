use std::fs;

#[test]
fn runtime_download_uses_a_private_tmp_directory_and_a_terminal_progress_bar() {
    let source = fs::read_to_string("src/runtime/install.rs").expect("runtime installer exists");
    let bootstrap =
        fs::read_to_string("src/runtime/bootstrap.rs").expect("runtime bootstrap exists");

    assert!(
        source.contains("runtime_root.join(\"tmp\")"),
        "runtime downloads must stay in noperson's private tmp directory"
    );
    assert!(
        source.contains("ProgressBar") && source.contains("ProgressStyle"),
        "interactive runtime downloads must render a real progress bar"
    );
    assert!(
        !source.contains("Runtime download: {artifact_name}"),
        "progress must not emit one tracing record per second"
    );
    assert!(
        bootstrap.contains("portable_runtime_root(&env::current_exe()?)"),
        "the default runtime root must travel beside the executable"
    );
    for global_root in ["XDG_DATA_HOME", "LOCALAPPDATA", ".local/share"] {
        assert!(
            !bootstrap.contains(global_root),
            "portable bootstrap must not write into {global_root}"
        );
    }
}
