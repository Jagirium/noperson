//! CrossSwap-compatible face color transfer.

use crate::pipeline::dfm::{lab_to_rgb, rgb_to_lab};
use crate::{
    config::parameters::{AutoColorMode, FaceSwapParams},
    gpu::ops::GpuOps,
    pipeline::workspace::GpuWorkspace,
};

pub fn apply_auto_color_gpu(
    gpu: &GpuOps,
    workspace: &mut GpuWorkspace,
    params: &FaceSwapParams,
    has_face_mask: bool,
) -> anyhow::Result<()> {
    if !params.auto_color.enabled {
        return Ok(());
    }
    let masked_mode = matches!(
        params.auto_color.mode,
        AutoColorMode::HistogramMasked | AutoColorMode::DflMasked
    );
    let use_mask = masked_mode && has_face_mask;
    if use_mask {
        gpu.mask_resize(
            &workspace.mask_learned_128,
            &mut workspace.parser_attribute_512,
            128,
            128,
            512,
            512,
        )?;
    }
    match params.auto_color.mode {
        AutoColorMode::Dfl | AutoColorMode::DflMasked => gpu.auto_color_dfl(
            &workspace.face_512_original,
            &mut workspace.face_512,
            &workspace.parser_attribute_512,
            &mut workspace.auto_color_stats,
            512 * 512,
            use_mask,
            params.auto_color.blend,
        )?,
        AutoColorMode::Histogram | AutoColorMode::HistogramMasked => {
            apply_histogram_gpu_fallback(gpu, workspace, use_mask, params.auto_color.blend)?;
        }
    }
    Ok(())
}

fn apply_histogram_gpu_fallback(
    gpu: &GpuOps,
    workspace: &mut GpuWorkspace,
    use_mask: bool,
    blend: f32,
) -> anyhow::Result<()> {
    const SIDE: usize = 512;
    const PIXELS: usize = SIDE * SIDE;
    gpu.download_into(
        &workspace.face_512_original,
        &mut workspace.host_color_original,
    )?;
    gpu.download_into(&workspace.face_512, &mut workspace.host_color_swapped)?;
    if use_mask {
        gpu.download_into(
            &workspace.parser_attribute_512,
            &mut workspace.host_color_mask,
        )?;
    }

    let original = chw_pixels(&workspace.host_color_original, PIXELS);
    let swapped = chw_pixels(&workspace.host_color_swapped, PIXELS);
    let mask = use_mask.then_some(workspace.host_color_mask.as_slice());
    let matched = histogram_transfer_reference(&original, &swapped, mask, blend);
    for (pixel, rgb) in matched.iter().enumerate() {
        for (channel, value) in rgb.iter().enumerate() {
            workspace.host_color_swapped[channel * PIXELS + pixel] = *value;
        }
    }
    gpu.upload_into(&workspace.host_color_swapped, &mut workspace.face_512)?;
    Ok(())
}

fn chw_pixels(chw: &[f32], pixels: usize) -> Vec<[f32; 3]> {
    (0..pixels)
        .map(|pixel| std::array::from_fn(|channel| chw[channel * pixels + pixel]))
        .collect()
}

/// CPU oracle for CrossSwap's DFL_Test and DFL_Orig color-transfer modes.
///
/// Images are interleaved RGB in the 0..255 range. A mask, when supplied,
/// selects pixels used for statistics at CrossSwap's fixed 0.2 cutoff; the
/// resulting transform is still applied to the complete swapped crop.
pub fn dfl_transfer_reference(
    original: &[[f32; 3]],
    swapped: &[[f32; 3]],
    mask: Option<&[f32]>,
    blend: f32,
) -> Vec<[f32; 3]> {
    assert_eq!(original.len(), swapped.len());
    if let Some(mask) = mask {
        assert_eq!(mask.len(), swapped.len());
    }
    if swapped.is_empty() {
        return Vec::new();
    }

    let selected: Vec<usize> = (0..swapped.len())
        .filter(|&pixel| mask.is_none_or(|values| values[pixel] >= 0.2))
        .collect();
    if selected.is_empty() {
        return swapped.to_vec();
    }

    let original_lab: Vec<_> = original
        .iter()
        .map(|rgb| rgb_to_lab(rgb.map(|value| value / 255.0)))
        .collect();
    let swapped_lab: Vec<_> = swapped
        .iter()
        .map(|rgb| rgb_to_lab(rgb.map(|value| value / 255.0)))
        .collect();
    let (original_mean, original_std) = selected_mean_std(&original_lab, &selected);
    let (swapped_mean, swapped_std) = selected_mean_std(&swapped_lab, &selected);
    let alpha = blend.clamp(0.0, 1.0);

    swapped_lab
        .into_iter()
        .zip(swapped)
        .map(|(mut lab, target)| {
            for channel in 0..3 {
                lab[channel] = (lab[channel] - swapped_mean[channel])
                    * ((original_std[channel] + 1e-6) / (swapped_std[channel] + 1e-6))
                    + original_mean[channel];
            }
            lab[0] = lab[0].clamp(0.0, 100.0);
            lab[1] = lab[1].clamp(-127.0, 127.0);
            lab[2] = lab[2].clamp(-127.0, 127.0);
            let matched = lab_to_rgb(lab).map(|value| value * 255.0);
            std::array::from_fn(|channel| {
                ((1.0 - alpha) * target[channel] + alpha * matched[channel]).clamp(0.0, 255.0)
            })
        })
        .collect()
}

/// CPU oracle for Kornia contrib's empirical-CDF histogram matching.
///
/// CrossSwap flattens all RGB channels into one distribution. Its masked mode
/// still computes that global transform, then blends the transformed result
/// through the supplied spatial mask.
pub fn histogram_transfer_reference(
    original: &[[f32; 3]],
    swapped: &[[f32; 3]],
    mask: Option<&[f32]>,
    blend: f32,
) -> Vec<[f32; 3]> {
    assert_eq!(original.len(), swapped.len());
    if let Some(mask) = mask {
        assert_eq!(mask.len(), swapped.len());
    }
    if swapped.is_empty() {
        return Vec::new();
    }

    let source_values: Vec<f32> = swapped
        .iter()
        .flat_map(|pixel| pixel.iter().copied())
        .collect();
    let template_values: Vec<f32> = original
        .iter()
        .flat_map(|pixel| pixel.iter().copied())
        .collect();
    let source_distribution = empirical_distribution(&source_values);
    let template_distribution = empirical_distribution(&template_values);
    let alpha = blend.clamp(0.0, 1.0);

    swapped
        .iter()
        .enumerate()
        .map(|(pixel, rgb)| {
            std::array::from_fn(|channel| {
                let source_index = source_distribution
                    .binary_search_by(|entry| entry.value.total_cmp(&rgb[channel]))
                    .expect("source value belongs to its empirical distribution");
                let matched = interpolate_quantile(
                    source_distribution[source_index].quantile,
                    &template_distribution,
                );
                let transferred = (1.0 - alpha) * rgb[channel] + alpha * matched;
                let spatial_alpha = mask.map_or(1.0, |values| values[pixel].clamp(0.0, 1.0));
                spatial_alpha * transferred + (1.0 - spatial_alpha) * rgb[channel]
            })
        })
        .collect()
}

#[derive(Clone, Copy)]
struct DistributionEntry {
    value: f32,
    quantile: f32,
}

fn empirical_distribution(values: &[f32]) -> Vec<DistributionEntry> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let mut distribution = Vec::new();
    let mut index = 0;
    while index < sorted.len() {
        let value = sorted[index];
        let mut end = index + 1;
        while end < sorted.len() && sorted[end] == value {
            end += 1;
        }
        distribution.push(DistributionEntry {
            value,
            quantile: end as f32 / sorted.len() as f32,
        });
        index = end;
    }
    distribution
}

fn interpolate_quantile(quantile: f32, template: &[DistributionEntry]) -> f32 {
    if template.len() == 1 {
        return template[0].value;
    }
    let upper = template
        .partition_point(|entry| entry.quantile <= quantile)
        .clamp(1, template.len() - 1);
    let lower = upper - 1;
    let span = template[upper].quantile - template[lower].quantile;
    let position = (quantile - template[lower].quantile) / span;
    template[lower].value * (1.0 - position) + template[upper].value * position
}

fn selected_mean_std(values: &[[f32; 3]], selected: &[usize]) -> ([f32; 3], [f32; 3]) {
    let count = selected.len() as f32;
    let mut mean = [0.0; 3];
    for &pixel in selected {
        for channel in 0..3 {
            mean[channel] += values[pixel][channel];
        }
    }
    for value in &mut mean {
        *value /= count;
    }

    let mut std = [0.0; 3];
    for &pixel in selected {
        for channel in 0..3 {
            std[channel] += (values[pixel][channel] - mean[channel]).powi(2);
        }
    }
    let correction = (count - 1.0).max(1.0);
    for value in &mut std {
        *value = (*value / correction).sqrt();
    }
    (mean, std)
}
