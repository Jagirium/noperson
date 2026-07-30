//! Compare a reference image with a generated self-swap.
//!
//! Usage:
//!   cargo run --example quality_report -- face.jpg swapped_output.png

use image::GenericImageView;
use noperson::quality::compare_rgb;

fn main() -> anyhow::Result<()> {
    let reference_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "face.jpg".to_owned());
    let candidate_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "swapped_output.png".to_owned());
    let reference = image::open(&reference_path)?;
    let candidate = image::open(&candidate_path)?;
    anyhow::ensure!(
        reference.dimensions() == candidate.dimensions(),
        "image dimensions differ: {:?} vs {:?}",
        reference.dimensions(),
        candidate.dimensions()
    );
    let (width, height) = reference.dimensions();
    let metrics = compare_rgb(
        reference.to_rgb8().as_raw(),
        candidate.to_rgb8().as_raw(),
        width,
        height,
    )?;
    println!("reference={reference_path}");
    println!("candidate={candidate_path}");
    println!("size={width}x{height}");
    println!("mae={:.6}", metrics.mae);
    println!("psnr_db={:.6}", metrics.psnr);
    println!("seam_p99={:.6}", metrics.seam_p99);
    println!("changed_fraction={:.6}", metrics.changed_fraction);
    Ok(())
}
