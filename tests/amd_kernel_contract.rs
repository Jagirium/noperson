use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn amd_codegen_plan_pins_rocm_and_covers_native_wave_widths() {
    let output = Command::new("bash")
        .arg(repo_root().join("scripts/kernels/check-amd-codegen.sh"))
        .arg("--print-plan")
        .output()
        .expect("run AMD codegen plan");

    assert!(
        output.status.success(),
        "AMD codegen plan failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).expect("UTF-8 AMD codegen plan"),
        concat!(
            "image=rocm/dev-ubuntu-24.04:7.2.4\n",
            "rocm=7.2.4\n",
            "target=gfx90a wave=64\n",
            "target=gfx942 wave=64\n",
            "target=gfx1100 wave=32\n",
        )
    );
}
