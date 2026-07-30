//! Objective image-quality diagnostics for reproducible self-swap checks.

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityMetrics {
    /// Mean absolute error across all RGB channels.
    pub mae: f64,
    /// Peak signal-to-noise ratio in dB.
    pub psnr: f64,
    /// 99th percentile of spatial gradients in the per-pixel change map.
    /// Hard paste rectangles score high; feathered transitions score lower.
    pub seam_p99: f64,
    /// Fraction of channel values changed by more than one level.
    pub changed_fraction: f64,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum QualityError {
    #[error("invalid RGB dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },
    #[error("RGB buffer length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
}

pub fn compare_rgb(
    reference: &[u8],
    candidate: &[u8],
    width: u32,
    height: u32,
) -> Result<QualityMetrics, QualityError> {
    if width == 0 || height == 0 {
        return Err(QualityError::InvalidDimensions { width, height });
    }
    let expected = width as usize * height as usize * 3;
    for data in [reference, candidate] {
        if data.len() != expected {
            return Err(QualityError::LengthMismatch {
                expected,
                actual: data.len(),
            });
        }
    }

    let mut absolute_sum = 0.0_f64;
    let mut squared_sum = 0.0_f64;
    let mut changed = 0_usize;
    let mut change_map = vec![0.0_f64; width as usize * height as usize];
    for (pixel, (reference_rgb, candidate_rgb)) in reference
        .chunks_exact(3)
        .zip(candidate.chunks_exact(3))
        .enumerate()
    {
        let mut pixel_change = 0.0;
        for channel in 0..3 {
            let difference = reference_rgb[channel] as f64 - candidate_rgb[channel] as f64;
            let absolute = difference.abs();
            absolute_sum += absolute;
            squared_sum += difference * difference;
            changed += usize::from(absolute > 1.0);
            pixel_change += absolute;
        }
        change_map[pixel] = pixel_change / 3.0;
    }

    let width = width as usize;
    let height = height as usize;
    let mut gradients = Vec::with_capacity((width - 1) * height + (height - 1) * width);
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if x + 1 < width {
                gradients.push((change_map[index] - change_map[index + 1]).abs());
            }
            if y + 1 < height {
                gradients.push((change_map[index] - change_map[index + width]).abs());
            }
        }
    }
    gradients.sort_by(f64::total_cmp);
    let percentile_index = ((gradients.len() as f64 * 0.99).ceil() as usize)
        .saturating_sub(1)
        .min(gradients.len().saturating_sub(1));
    let seam_p99 = gradients.get(percentile_index).copied().unwrap_or(0.0);

    let mse = squared_sum / expected as f64;
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0_f64 * 255.0 / mse).log10()
    };
    Ok(QualityMetrics {
        mae: absolute_sum / expected as f64,
        psnr,
        seam_p99,
        changed_fraction: changed as f64 / expected as f64,
    })
}
