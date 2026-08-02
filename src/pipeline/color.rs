//! CrossSwap-compatible face color transfer.

use crate::pipeline::dfm::{lab_to_rgb, rgb_to_lab};
use crate::{
    config::parameters::{AutoColorMode, ColorAdjustParams, FaceSwapParams, MakeupParams},
    gpu::ops::GpuOps,
    pipeline::workspace::GpuWorkspace,
};

pub fn apply_makeup_reference(
    image: &[[f32; 3]],
    classes: &[u8],
    hair: &MakeupParams,
    lips: &MakeupParams,
) -> Vec<[f32; 3]> {
    assert_eq!(image.len(), classes.len());
    image
        .iter()
        .zip(classes)
        .map(|(pixel, class)| {
            let makeup = if *class == 17 && hair.enabled {
                Some(hair)
            } else if matches!(*class, 12 | 13) && lips.enabled {
                Some(lips)
            } else {
                None
            };
            makeup.map_or(*pixel, |makeup| {
                let blend = makeup.blend.clamp(0.0, 1.0);
                std::array::from_fn(|channel| {
                    pixel[channel] * (1.0 - blend) + makeup.color[channel] * blend
                })
            })
        })
        .collect()
}

pub fn apply_makeup_gpu(
    gpu: &GpuOps,
    workspace: &mut GpuWorkspace,
    params: &FaceSwapParams,
) -> anyhow::Result<()> {
    let parser = &params.faceparser;
    if !params.faceparser_enabled || (!parser.hair_makeup.enabled && !parser.lips_makeup.enabled) {
        return Ok(());
    }
    gpu.parser_makeup(
        &mut workspace.face_512,
        &workspace.parser_classes,
        parser.hair_makeup.enabled,
        parser.hair_makeup.color,
        parser.hair_makeup.blend,
        parser.lips_makeup.enabled,
        parser.lips_makeup.color,
        parser.lips_makeup.blend,
    )?;
    Ok(())
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn noperson_jpeg_roundtrip(
        rgb: *const u8,
        width: i32,
        height: i32,
        quality: i32,
        output: *mut u8,
    ) -> i32;
}

pub fn jpeg_roundtrip_reference(
    rgb: &[[u8; 3]],
    width: usize,
    height: usize,
    quality: u8,
) -> anyhow::Result<Vec<[u8; 3]>> {
    anyhow::ensure!(
        rgb.len() == width * height,
        "JPEG dimensions do not match data"
    );
    anyhow::ensure!((1..=100).contains(&quality), "JPEG quality must be 1..=100");
    let input: Vec<u8> = rgb.iter().flatten().copied().collect();
    let mut output = vec![0u8; input.len()];

    #[cfg(target_os = "linux")]
    {
        let status = unsafe {
            noperson_jpeg_roundtrip(
                input.as_ptr(),
                width as i32,
                height as i32,
                quality as i32,
                output.as_mut_ptr(),
            )
        };
        anyhow::ensure!(status == 0, "libjpeg roundtrip failed with status {status}");
    }
    #[cfg(not(target_os = "linux"))]
    {
        use image::{ImageEncoder, codecs::jpeg::JpegEncoder};
        let mut encoded = Vec::new();
        JpegEncoder::new_with_quality(&mut encoded, quality).write_image(
            &input,
            width as u32,
            height as u32,
            image::ExtendedColorType::Rgb8,
        )?;
        output = image::load_from_memory_with_format(&encoded, image::ImageFormat::Jpeg)?
            .to_rgb8()
            .into_raw();
    }
    Ok(output
        .chunks_exact(3)
        .map(|pixel| [pixel[0], pixel[1], pixel[2]])
        .collect())
}

/// CPU oracle for CrossSwap's ordered torchvision-v2 color adjustment stack.
pub fn adjust_color_reference(
    image: &[[f32; 3]],
    width: usize,
    height: usize,
    params: &ColorAdjustParams,
) -> Vec<[f32; 3]> {
    assert_eq!(image.len(), width * height);
    let offsets = [params.red, params.green, params.blue];
    let mut output: Vec<[f32; 3]> = image
        .iter()
        .map(|rgb| {
            std::array::from_fn(|channel| {
                (rgb[channel].powf(params.gamma) + offsets[channel])
                    .clamp(0.0, 255.0)
                    .trunc()
            })
        })
        .collect();

    map_u8_stage(&mut output, |value| value * params.brightness);
    let contrast_mean = output
        .iter()
        .map(|rgb| grayscale(*rgb).floor())
        .sum::<f32>()
        / output.len() as f32;
    map_u8_stage(&mut output, |value| {
        value * params.contrast + contrast_mean * (1.0 - params.contrast)
    });
    for rgb in &mut output {
        let gray = grayscale(*rgb).floor();
        for value in rgb {
            *value = (*value * params.saturation + gray * (1.0 - params.saturation))
                .clamp(0.0, 255.0)
                .trunc();
        }
    }

    if width > 2 && height > 2 {
        let input = output.clone();
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let pixel = y * width + x;
                for channel in 0..3 {
                    let mut weighted = 0.0;
                    for dy in 0..3 {
                        for dx in 0..3 {
                            let weight = if dx == 1 && dy == 1 { 5.0 } else { 1.0 };
                            weighted += input[(y + dy - 1) * width + x + dx - 1][channel] * weight;
                        }
                    }
                    let blurred = (weighted / 13.0).round_ties_even();
                    output[pixel][channel] = (input[pixel][channel] * params.sharpness
                        + blurred * (1.0 - params.sharpness))
                        .clamp(0.0, 255.0)
                        .trunc();
                }
            }
        }
    }

    for rgb in &mut output {
        let mut hsv = rgb_to_hsv(rgb.map(|value| value / 255.0));
        hsv[0] = (hsv[0] + params.hue).rem_euclid(1.0);
        let adjusted = hsv_to_rgb(hsv);
        for channel in 0..3 {
            rgb[channel] = (adjusted[channel].clamp(0.0, 1.0) * (256.0 - 1e-3)).trunc();
        }
    }
    output
}

fn map_u8_stage(image: &mut [[f32; 3]], transform: impl Fn(f32) -> f32) {
    for rgb in image {
        for value in rgb {
            *value = transform(*value).clamp(0.0, 255.0).trunc();
        }
    }
}

fn grayscale(rgb: [f32; 3]) -> f32 {
    rgb[0] * 0.2989 + rgb[1] * 0.587 + rgb[2] * 0.114
}

fn rgb_to_hsv(rgb: [f32; 3]) -> [f32; 3] {
    let max = rgb.into_iter().fold(f32::NEG_INFINITY, f32::max);
    let min = rgb.into_iter().fold(f32::INFINITY, f32::min);
    if max == min {
        return [0.0, 0.0, max];
    }
    let range = max - min;
    let saturation = range / max;
    let rc = (max - rgb[0]) / range;
    let gc = (max - rgb[1]) / range;
    let bc = (max - rgb[2]) / range;
    let hue = if max == rgb[0] {
        bc - gc
    } else if max == rgb[1] {
        2.0 + rc - bc
    } else {
        4.0 + gc - rc
    };
    [(hue / 6.0 + 1.0) % 1.0, saturation, max]
}

fn hsv_to_rgb(hsv: [f32; 3]) -> [f32; 3] {
    let h6 = hsv[0] * 6.0;
    let sector = h6.floor() as usize % 6;
    let fraction = h6 - h6.floor();
    let p = (1.0 - hsv[1]) * hsv[2];
    let q = (1.0 - hsv[1] * fraction) * hsv[2];
    let t = (1.0 - hsv[1] * (1.0 - fraction)) * hsv[2];
    match sector {
        0 => [hsv[2], t, p],
        1 => [q, hsv[2], p],
        2 => [p, hsv[2], t],
        3 => [p, q, hsv[2]],
        4 => [t, p, hsv[2]],
        _ => [hsv[2], p, q],
    }
}

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

pub fn apply_color_adjust_gpu(
    gpu: &GpuOps,
    workspace: &mut GpuWorkspace,
    params: &FaceSwapParams,
) -> anyhow::Result<()> {
    if !params.color_adjust.enabled && !params.color_correction {
        return Ok(());
    }
    let controls = &params.color_adjust;
    let seed = workspace.color_noise_nonce;
    workspace.color_noise_nonce = workspace.color_noise_nonce.wrapping_add(1);
    gpu.adjust_color(
        &mut workspace.face_512,
        &mut workspace.face_512_scratch,
        &mut workspace.color_gray_sum,
        512,
        512,
        controls.gamma,
        [controls.red, controls.green, controls.blue],
        controls.brightness,
        controls.contrast,
        controls.saturation,
        controls.sharpness,
        controls.hue,
        controls.noise,
        seed,
    )?;
    Ok(())
}

pub fn apply_final_blur_gpu(
    gpu: &GpuOps,
    workspace: &mut GpuWorkspace,
    params: &FaceSwapParams,
) -> anyhow::Result<()> {
    if !params.final_blur_enabled {
        return Ok(());
    }
    let kernel_size = params.final_blur * 2 + 1;
    let sigma = params.final_blur as f32 * 0.1;
    let ks = crate::pipeline::face_mask::prepare_blur_kernel(gpu, workspace, kernel_size, sigma)?;
    gpu.gaussian_blur_chw(
        &mut workspace.face_512,
        &mut workspace.face_512_scratch,
        512,
        512,
        &workspace.blur_kernel,
        ks,
    )?;
    Ok(())
}

pub fn apply_jpeg_compression(
    gpu: &GpuOps,
    workspace: &mut GpuWorkspace,
    params: &FaceSwapParams,
) -> anyhow::Result<()> {
    if !params.jpeg_compression_enabled {
        return Ok(());
    }
    const PIXELS: usize = 512 * 512;
    gpu.download_into(&workspace.face_512, &mut workspace.host_color_swapped)?;
    let rgb: Vec<[u8; 3]> = (0..PIXELS)
        .map(|pixel| {
            std::array::from_fn(|channel| {
                workspace.host_color_swapped[channel * PIXELS + pixel].clamp(0.0, 255.0) as u8
            })
        })
        .collect();
    let decoded = jpeg_roundtrip_reference(&rgb, 512, 512, params.jpeg_quality)?;
    for (pixel, rgb) in decoded.iter().enumerate() {
        for (channel, value) in rgb.iter().enumerate() {
            workspace.host_color_swapped[channel * PIXELS + pixel] = f32::from(*value);
        }
    }
    gpu.upload_into(&workspace.host_color_swapped, &mut workspace.face_512)?;
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
