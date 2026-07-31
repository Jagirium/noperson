//! Face masks: Occluder, border mask, gaussian blur, soft paste-back.
//!
//! Port of crosswap/app/processors/face_masks.py + frame_worker.py mask logic.
//!
//! Mask convention: 1.0 = keep swapped face, 0.0 = keep original background.

use crate::config::parameters::{
    FaceParserMaskParams, FaceSwapParams, RestoreEyesParams, RestoreMouthParams,
};
use crate::gpu::ops::GpuOps;
use crate::models::manager::ModelManager;
use crate::pipeline::workspace::{GpuWorkspace, MAX_BLUR_KS};

pub fn postprocess_occluder_mask(mask: &mut [f32]) {
    for value in mask {
        *value = if *value > 0.0 { 1.0 } else { 0.0 };
    }
}

pub fn postprocess_xseg_mask(mask: &mut [f32]) {
    for value in mask {
        *value = value.clamp(0.0, 1.0);
        if *value < 0.1 {
            *value = 0.0;
        }
    }
}

pub fn fake_diff_mask(
    swapped_chw: &[f32],
    original_chw: &[f32],
    pixels: usize,
    amount: u32,
) -> Vec<f32> {
    assert_eq!(swapped_chw.len(), pixels * 3);
    assert_eq!(original_chw.len(), pixels * 3);
    let threshold = amount as f32 * 2.55;
    (0..pixels)
        .map(|pixel| {
            let changed = (0..3).any(|channel| {
                let index = channel * pixels + pixel;
                (swapped_chw[index] - original_chw[index]).abs() >= threshold
            });
            f32::from(changed)
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticRegion {
    Eyes,
    Mouth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreEllipse {
    pub center_x: i32,
    pub center_y: i32,
    pub radius_x: i32,
    pub radius_y: i32,
}

pub fn mouth_restore_ellipse(
    landmarks: &[[f64; 2]; 5],
    params: &RestoreMouthParams,
) -> RestoreEllipse {
    let [left_x, left_y] = landmarks[3].map(|value| value as i32);
    let [right_x, right_y] = landmarks[4].map(|value| value as i32);
    let delta_x = left_x - right_x;
    let delta_y = left_y - right_y;
    let base_radius = (((delta_x * delta_x + delta_y * delta_y) as f64).sqrt()
        * params.size_factor as f64) as i32;
    RestoreEllipse {
        center_x: (left_x + right_x) / 2 + params.offset_x,
        center_y: (left_y + right_y) / 2 + params.offset_y,
        radius_x: (base_radius as f64 * params.radius_x as f64) as i32,
        radius_y: (base_radius as f64 * params.radius_y as f64) as i32,
    }
}

pub fn eye_restore_ellipses(
    landmarks: &[[f64; 2]; 5],
    params: &RestoreEyesParams,
) -> [RestoreEllipse; 2] {
    let [mut left_x, left_y] = landmarks[0].map(|value| value as i32);
    let [mut right_x, right_y] = landmarks[1].map(|value| value as i32);
    left_x += params.offset_x;
    right_x += params.offset_x;
    let left_y = left_y + params.offset_y;
    let right_y = right_y + params.offset_y;
    let delta_x = left_x - right_x;
    let delta_y = left_y - right_y;
    let base_radius = (((delta_x * delta_x + delta_y * delta_y) as f64).sqrt()
        / params.size_factor as f64) as i32;
    let radius_x = (base_radius as f64 * params.radius_x as f64) as i32;
    let radius_y = (base_radius as f64 * params.radius_y as f64) as i32;
    left_x += params.spacing_offset;
    right_x -= params.spacing_offset;
    [
        RestoreEllipse {
            center_x: left_x,
            center_y: left_y,
            radius_x,
            radius_y,
        },
        RestoreEllipse {
            center_x: right_x,
            center_y: right_y,
            radius_x,
            radius_y,
        },
    ]
}

pub fn apply_restore_ellipse_mask_reference(
    mask: &mut [f32],
    width: u32,
    height: u32,
    ellipse: RestoreEllipse,
    blend: f32,
    feather: u32,
) {
    if ellipse.radius_x <= 0 || ellipse.radius_y <= 0 || feather == 0 {
        return;
    }
    let width = width as i32;
    let height = height as i32;
    let min_x = (ellipse.center_x - ellipse.radius_x).max(0);
    let max_x = (ellipse.center_x + ellipse.radius_x).min(width);
    let min_y = (ellipse.center_y - ellipse.radius_y).max(0);
    let max_y = (ellipse.center_y + ellipse.radius_y).min(height);
    for y in min_y..max_y {
        for x in min_x..max_x {
            let dx = (x - ellipse.center_x) as f32 / ellipse.radius_x as f32;
            let dy = (y - ellipse.center_y) as f32 / ellipse.radius_y as f32;
            let distance = (dx * dx + dy * dy).sqrt();
            let soft =
                ((1.0 - distance) * ellipse.radius_x as f32 / feather as f32).clamp(0.0, 1.0);
            let index = y as usize * width as usize + x as usize;
            mask[index] *= 1.0 - soft * (1.0 - blend);
        }
    }
}

pub fn semantic_region_mask(classes: &[u8], region: SemanticRegion) -> Vec<f32> {
    classes
        .iter()
        .map(|class| {
            let selected = match region {
                SemanticRegion::Eyes => matches!(*class, 4 | 5),
                SemanticRegion::Mouth => matches!(*class, 11..=13),
            };
            f32::from(selected)
        })
        .collect()
}

/// CPU oracle for CrossSwap's FaceParser class-mask composition.
pub fn compose_face_parser_mask(
    classes: &[u8],
    width: u32,
    height: u32,
    params: &FaceParserMaskParams,
) -> anyhow::Result<Vec<f32>> {
    let pixels = width as usize * height as usize;
    anyhow::ensure!(width > 0 && height > 0, "parser mask dimensions are empty");
    anyhow::ensure!(
        classes.len() == pixels,
        "parser class map has {} values, expected {pixels}",
        classes.len()
    );

    let mut output = if params.background == 0 {
        vec![1.0; pixels]
    } else {
        let mut background: Vec<f32> = classes
            .iter()
            .map(|class| {
                if matches!(*class, 0 | 14 | 15 | 16 | 17 | 18) {
                    0.0
                } else {
                    1.0
                }
            })
            .collect();
        morphology_reference(&mut background, width, height, params.background);
        blur_reference(&mut background, width, height, params.background_blur);
        background
    };

    for (class, amount) in [
        (1, params.face),
        (2, params.left_eyebrow),
        (3, params.right_eyebrow),
        (4, params.left_eye),
        (5, params.right_eye),
        (6, params.eyeglasses),
        (10, params.nose),
        (11, params.mouth),
        (12, params.upper_lip),
        (13, params.lower_lip),
        (14, params.neck),
        (17, params.hair),
    ] {
        if amount == 0 {
            continue;
        }
        let mut attribute: Vec<f32> = classes
            .iter()
            .map(|actual| f32::from(*actual == class))
            .collect();
        morphology_reference(&mut attribute, width, height, amount as i32);
        for value in &mut attribute {
            *value = 1.0 - *value;
        }
        blur_reference(&mut attribute, width, height, params.face_blur);
        for (mask, attribute) in output.iter_mut().zip(attribute) {
            *mask = (*mask * attribute).clamp(0.0, 1.0);
        }
    }
    Ok(output)
}

fn morphology_reference(mask: &mut [f32], width: u32, height: u32, amount: i32) {
    let erode = amount < 0;
    if erode {
        for value in mask.iter_mut() {
            *value = 1.0 - *value;
        }
    }
    let mut scratch = vec![0.0; mask.len()];
    for _ in 0..amount.unsigned_abs() {
        for y in 0..height {
            for x in 0..width {
                let mut maximum = 0.0f32;
                for offset_y in -1..=1 {
                    for offset_x in -1..=1 {
                        let sample_y = y as i32 + offset_y;
                        let sample_x = x as i32 + offset_x;
                        if sample_y >= 0
                            && sample_y < height as i32
                            && sample_x >= 0
                            && sample_x < width as i32
                        {
                            maximum = maximum
                                .max(mask[sample_y as usize * width as usize + sample_x as usize]);
                        }
                    }
                }
                scratch[y as usize * width as usize + x as usize] = maximum;
            }
        }
        mask.copy_from_slice(&scratch);
    }
    if erode {
        for value in mask {
            *value = 1.0 - *value;
        }
    }
}

fn blur_reference(mask: &mut [f32], width: u32, height: u32, amount: u32) {
    if amount > 0 {
        gaussian_blur(
            mask,
            width,
            height,
            amount.saturating_mul(2).saturating_add(1),
            (amount as f32 + 1.0) * 0.2,
        );
    }
}

/// Occluder mask inference.
///
/// Input: face [3, 256, 256] in [0, 255].
/// ONNX: input="img" [1,3,256,256] normalized /255, output="output" [1,1,256,256].
/// Output: binary mask [256, 256] where 1.0 = face, 0.0 = occluded.
pub fn apply_occluder(
    manager: &mut ModelManager,
    face_chw_256: &[f32], // [3, 256, 256] in [0, 255]
) -> anyhow::Result<Vec<f32>> {
    let session = manager
        .get_mut("Occluder")
        .ok_or_else(|| anyhow::anyhow!("Occluder not loaded"))?;

    // Normalize: / 255.0
    let normalized: Vec<f32> = face_chw_256.iter().map(|&v| v / 255.0).collect();

    let input_tensor = ort::value::Tensor::from_array(([1usize, 3, 256, 256], normalized))?;

    let outputs = session.run(ort::inputs!["img" => input_tensor])?;
    let (_shape, data) = outputs["output"].try_extract_tensor::<f32>()?;

    // Threshold at 0 → binary, then float
    let mask: Vec<f32> = data
        .iter()
        .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
        .collect();
    Ok(mask) // [256*256]
}

/// Generate border mask [size, size] with zeroed borders.
///
/// `top/bottom/left/right` are the number of pixels to zero from each edge (0-based).
/// Result: 1.0 inside the border, 0.0 outside.
pub fn generate_border_mask(size: u32, top: u32, bottom: u32, left: u32, right: u32) -> Vec<f32> {
    let s = size as usize;
    let mut mask = vec![1.0f32; s * s];

    let bottom_start = size - bottom;
    let right_start = size - right;

    for y in 0..size {
        for x in 0..size {
            if y < top || y >= bottom_start || x < left || x >= right_start {
                mask[(y * size + x) as usize] = 0.0;
            }
        }
    }
    mask
}

/// Generate a soft oval mask for face boundary.
///
/// Returns [size, size] mask with 1.0 inside oval, smooth falloff at edges.
pub fn soft_oval_mask(
    width: u32,
    height: u32,
    center_x: f32,
    center_y: f32,
    radius_x: f32,
    radius_y: f32,
    feather: f32,
) -> Vec<f32> {
    let mut mask = vec![0.0f32; (height * width) as usize];
    let scale = radius_x / feather.max(1.0);

    for y in 0..height {
        for x in 0..width {
            let dx = (x as f32 - center_x) / radius_x;
            let dy = (y as f32 - center_y) / radius_y;
            let dist = (dx * dx + dy * dy).sqrt();
            let v = ((1.0 - dist) * scale).clamp(0.0, 1.0);
            mask[(y * width + x) as usize] = v;
        }
    }
    mask
}

/// Apply Gaussian blur to a single-channel [H, W] mask (CPU).
///
/// Separable 2-pass (horizontal + vertical) Gaussian filter.
pub fn gaussian_blur(mask: &mut [f32], width: u32, height: u32, kernel_size: u32, sigma: f32) {
    gaussian_blur_with_border(
        mask,
        width,
        height,
        kernel_size,
        sigma,
        GaussianBorder::Reflect,
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaussianBorder {
    /// torchvision.transforms.GaussianBlur reflection without repeating the edge.
    Reflect,
    /// torch.nn.functional.conv2d implicit zero padding.
    Zero,
}

pub fn gaussian_blur_with_border(
    mask: &mut [f32],
    width: u32,
    height: u32,
    kernel_size: u32,
    sigma: f32,
    border: GaussianBorder,
) {
    if kernel_size <= 1 {
        return;
    }

    let ks = kernel_size as i32;
    let half = ks / 2;

    // Build 1D Gaussian kernel
    let mut kernel = vec![0.0f32; ks as usize];
    let mut sum = 0.0f32;
    for i in 0..ks {
        let x = (i - half) as f32;
        let v = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel[i as usize] = v;
        sum += v;
    }
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    let w = width as usize;
    let h = height as usize;

    // Horizontal pass
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut val = 0.0f32;
            for (k, weight) in kernel.iter().enumerate() {
                let sx = x as i32 + k as i32 - half;
                if let Some(sx) = gaussian_border_index(sx, width as i32, border) {
                    val += mask[y * w + sx] * weight;
                }
            }
            tmp[y * w + x] = val;
        }
    }

    // Vertical pass
    for y in 0..h {
        for x in 0..w {
            let mut val = 0.0f32;
            for (k, weight) in kernel.iter().enumerate() {
                let sy = y as i32 + k as i32 - half;
                if let Some(sy) = gaussian_border_index(sy, height as i32, border) {
                    val += tmp[sy * w + x] * weight;
                }
            }
            mask[y * w + x] = val;
        }
    }
}

fn gaussian_border_index(mut index: i32, length: i32, border: GaussianBorder) -> Option<usize> {
    if (0..length).contains(&index) {
        return Some(index as usize);
    }
    if border == GaussianBorder::Zero || length <= 1 {
        return (border == GaussianBorder::Reflect && length == 1).then_some(0);
    }
    while index < 0 || index >= length {
        index = if index < 0 {
            -index
        } else {
            2 * length - 2 - index
        };
    }
    Some(index as usize)
}

/// Resize a single-channel mask [src_h, src_w] → [dst_h, dst_w] via bilinear.
pub fn resize_mask(src: &[f32], src_h: u32, src_w: u32, dst_h: u32, dst_w: u32) -> Vec<f32> {
    let mut dst = vec![0.0f32; (dst_h * dst_w) as usize];
    let sy = src_h as f32 / dst_h as f32;
    let sx = src_w as f32 / dst_w as f32;

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let fy = dy as f32 * sy;
            let fx = dx as f32 * sx;
            let y0 = (fy as u32).min(src_h - 1);
            let y1 = (y0 + 1).min(src_h - 1);
            let x0 = (fx as u32).min(src_w - 1);
            let x1 = (x0 + 1).min(src_w - 1);
            let wy = fy - y0 as f32;
            let wx = fx - x0 as f32;

            let v = src[(y0 * src_w + x0) as usize] * (1.0 - wx) * (1.0 - wy)
                + src[(y0 * src_w + x1) as usize] * wx * (1.0 - wy)
                + src[(y1 * src_w + x0) as usize] * (1.0 - wx) * wy
                + src[(y1 * src_w + x1) as usize] * wx * wy;
            dst[(dy * dst_w + dx) as usize] = v;
        }
    }
    dst
}

/// Compose final mask from border mask and optional occluder.
///
/// Both masks should be same size. Result = elementwise multiply.
pub fn compose_masks(border: &[f32], occluder: Option<&[f32]>) -> Vec<f32> {
    match occluder {
        Some(occ) => border
            .iter()
            .zip(occ.iter())
            .map(|(&b, &o)| b * o)
            .collect(),
        None => border.to_vec(),
    }
}

/// GPU-native mask generation: border + optional restorer oval + gaussian blur.
///
/// Writes the blurred mask into `ws.mask_128`, then resizes to `ws.mask_512`.
/// The blur kernel is cached in the workspace and regenerated only when
/// kernel_size/sigma change.
pub fn gpu_generate_mask_512(
    gpu: &GpuOps,
    ws: &mut GpuWorkspace,
    border_top: u32,
    border_bottom: u32,
    border_left: u32,
    border_right: u32,
    blur_ks: u32,
    blur_sigma: f32,
    use_restorer_oval: bool,
    use_learned_mask: bool,
    overall_blur: u32,
) -> anyhow::Result<()> {
    // Crosswap only applies the oval fallback when restoration is active and
    // no explicit parser/XSeg mask is available.
    gpu.border_oval_mask(
        &mut ws.mask_128,
        128,
        border_top,
        border_bottom,
        border_left,
        border_right,
        use_restorer_oval,
    )?;

    // 2. Optional gaussian blur → mask_128 (uses mask_128_tmp scratch)
    if blur_ks > 1 {
        let ks = prepare_blur_kernel(gpu, ws, blur_ks, blur_sigma)?;
        gpu.gaussian_blur_mask(
            &mut ws.mask_128,
            &mut ws.mask_128_tmp,
            128,
            128,
            &ws.blur_kernel,
            ks,
        )?;
    }

    if use_learned_mask {
        if overall_blur > 0 {
            let kernel_size = overall_blur * 2 + 1;
            let sigma = (overall_blur as f32 + 1.0) * 0.2;
            let ks = prepare_blur_kernel(gpu, ws, kernel_size, sigma)?;
            gpu.gaussian_blur_mask(
                &mut ws.mask_learned_128,
                &mut ws.mask_128_tmp,
                128,
                128,
                &ws.blur_kernel,
                ks,
            )?;
        }
        gpu.mask_mul(&mut ws.mask_128, &ws.mask_learned_128)?;
    }

    // 3. Resize 128 → 512 into mask_512
    gpu.mask_resize(&ws.mask_128, &mut ws.mask_512, 128, 128, 512, 512)?;
    Ok(())
}

/// Run enabled learned masks and compose them in CrossSwap's 128-space.
pub fn gpu_generate_learned_mask_128(
    gpu: &GpuOps,
    manager: &mut ModelManager,
    ws: &mut GpuWorkspace,
    params: &FaceSwapParams,
) -> anyhow::Result<bool> {
    let landmark_restore =
        !params.faceparser_enabled && (params.restore_mouth || params.restore_eyes);
    if !params.occluder_enabled
        && !params.xseg_enabled
        && !params.faceparser_enabled
        && !landmark_restore
    {
        return Ok(false);
    }

    gpu.border_oval_mask(&mut ws.mask_learned_128, 128, 0, 0, 0, 0, false)?;

    if params.occluder_enabled {
        prepare_learned_mask_input(gpu, ws)?;
        crate::pipeline::ort_binding::run_bound_f32(
            manager,
            &gpu.stream,
            "Occluder",
            "img",
            &ws.face_256,
            &[1, 3, 256, 256],
            "output",
            &mut ws.mask_256,
            &[1, 1, 256, 256],
        )?;
        gpu.occluder_threshold(&mut ws.mask_256)?;
        gpu.morphology_mask(
            &mut ws.mask_256,
            &mut ws.mask_256_tmp,
            256,
            256,
            params.occluder_size,
        )?;
        compose_learned_mask(gpu, ws, params.occluder_xseg_blur)?;
    }

    if params.xseg_enabled {
        prepare_learned_mask_input(gpu, ws)?;
        crate::pipeline::ort_binding::run_bound_f32(
            manager,
            &gpu.stream,
            "XSeg",
            "in_face:0",
            &ws.face_256,
            &[1, 3, 256, 256],
            "out_mask:0",
            &mut ws.mask_256,
            &[1, 1, 256, 256],
        )?;
        gpu.xseg_postprocess(&mut ws.mask_256)?;
        gpu.morphology_mask(
            &mut ws.mask_256,
            &mut ws.mask_256_tmp,
            256,
            256,
            params.xseg_size,
        )?;
        compose_learned_mask(gpu, ws, params.occluder_xseg_blur)?;
    }

    if params.faceparser_enabled {
        compose_faceparser_gpu(gpu, manager, ws, &params.faceparser)?;
    }

    Ok(true)
}

pub fn gpu_apply_landmark_restore_mask(
    gpu: &GpuOps,
    ws: &mut GpuWorkspace,
    params: &FaceSwapParams,
    aligned_landmarks: &[[f64; 2]; 5],
) -> anyhow::Result<()> {
    if params.faceparser_enabled || (!params.restore_mouth && !params.restore_eyes) {
        return Ok(());
    }

    gpu.border_oval_mask(&mut ws.parser_mask_512, 512, 0, 0, 0, 0, false)?;
    if params.restore_mouth {
        apply_restore_ellipse_gpu(
            gpu,
            &mut ws.parser_mask_512,
            mouth_restore_ellipse(aligned_landmarks, &params.restore_mouth_params),
            params.restore_mouth_params.blend,
            params.restore_mouth_params.feather,
        )?;
    }
    if params.restore_eyes {
        for ellipse in eye_restore_ellipses(aligned_landmarks, &params.restore_eyes_params) {
            apply_restore_ellipse_gpu(
                gpu,
                &mut ws.parser_mask_512,
                ellipse,
                params.restore_eyes_params.blend,
                params.restore_eyes_params.feather,
            )?;
        }
    }

    let blur = params.restore_eyes_mouth_blur;
    if blur > 0 {
        let kernel_size = blur * 2 + 1;
        let sigma = (blur as f32 + 1.0) * 0.2;
        let ks = prepare_blur_kernel(gpu, ws, kernel_size, sigma)?;
        gpu.gaussian_blur_mask(
            &mut ws.parser_mask_512,
            &mut ws.parser_tmp_512,
            512,
            512,
            &ws.blur_kernel,
            ks,
        )?;
    }
    gpu.mask_resize(&ws.parser_mask_512, &mut ws.mask_128, 512, 512, 128, 128)?;
    gpu.mask_mul(&mut ws.mask_learned_128, &ws.mask_128)?;
    Ok(())
}

fn apply_restore_ellipse_gpu(
    gpu: &GpuOps,
    mask: &mut cudarc::driver::CudaSlice<f32>,
    ellipse: RestoreEllipse,
    blend: f32,
    feather: u32,
) -> anyhow::Result<()> {
    gpu.restore_ellipse_mask(
        mask,
        512,
        512,
        ellipse.center_x,
        ellipse.center_y,
        ellipse.radius_x,
        ellipse.radius_y,
        blend,
        feather,
    )?;
    Ok(())
}

pub fn gpu_restore_semantic_regions(
    gpu: &GpuOps,
    ws: &mut GpuWorkspace,
    params: &FaceSwapParams,
) -> anyhow::Result<()> {
    if !params.faceparser_enabled {
        return Ok(());
    }
    if params.restore_mouth {
        restore_semantic_region(gpu, ws, params, SemanticRegion::Mouth)?;
    }
    if params.restore_eyes {
        restore_semantic_region(gpu, ws, params, SemanticRegion::Eyes)?;
    }
    Ok(())
}

pub fn gpu_apply_fake_diff(
    gpu: &GpuOps,
    ws: &mut GpuWorkspace,
    params: &FaceSwapParams,
) -> anyhow::Result<()> {
    if !params.differencing_enabled {
        return Ok(());
    }
    let pixels = 512 * 512;
    gpu.fake_diff_mask(
        &ws.face_512,
        &ws.face_512_original,
        &mut ws.parser_mask_512,
        pixels,
        params.differencing_amount,
    )?;
    if params.differencing_blur > 0 {
        let kernel_size = params.differencing_blur * 2 + 1;
        let sigma = (params.differencing_blur as f32 + 1.0) * 0.2;
        let ks = prepare_blur_kernel(gpu, ws, kernel_size, sigma)?;
        gpu.gaussian_blur_mask(
            &mut ws.parser_mask_512,
            &mut ws.parser_tmp_512,
            512,
            512,
            &ws.blur_kernel,
            ks,
        )?;
    }
    gpu.fake_diff_composite(
        &mut ws.face_512,
        &ws.face_512_original,
        &ws.parser_mask_512,
        pixels,
    )?;
    Ok(())
}

fn restore_semantic_region(
    gpu: &GpuOps,
    ws: &mut GpuWorkspace,
    params: &FaceSwapParams,
    region: SemanticRegion,
) -> anyhow::Result<()> {
    let (region_code, dilation, feather, temporal_alpha, luminance_factor, blend) = match region {
        SemanticRegion::Eyes => (
            0,
            2,
            params.restore_eyes_params.feather,
            0.4,
            0.6,
            params.restore_eyes_params.blend,
        ),
        SemanticRegion::Mouth => (
            1,
            3,
            params.restore_mouth_params.feather,
            0.5,
            0.5,
            params.restore_mouth_params.blend,
        ),
    };
    gpu.semantic_region_mask(
        &ws.parser_classes,
        &mut ws.parser_attribute_512,
        &mut ws.semantic_count,
        region_code,
    )?;
    gpu.morphology_mask(
        &mut ws.parser_attribute_512,
        &mut ws.parser_tmp_512,
        512,
        512,
        dilation,
    )?;

    let minimum_kernel = match region {
        SemanticRegion::Eyes => 3,
        SemanticRegion::Mouth => 5,
    };
    let kernel_size = feather
        .saturating_mul(2)
        .saturating_add(1)
        .max(minimum_kernel);
    let sigma = (kernel_size as f32 / 3.0).max(match region {
        SemanticRegion::Eyes => 1.0,
        SemanticRegion::Mouth => 1.5,
    });
    let ks = prepare_blur_kernel(gpu, ws, kernel_size, sigma)?;
    gpu.gaussian_blur_mask_with_border(
        &mut ws.parser_attribute_512,
        &mut ws.parser_tmp_512,
        512,
        512,
        &ws.blur_kernel,
        ks,
        1,
    )?;

    match region {
        SemanticRegion::Eyes => gpu.semantic_temporal_mask(
            &mut ws.parser_attribute_512,
            &mut ws.semantic_previous_eyes,
            &ws.semantic_count,
            &mut ws.semantic_eyes_valid,
            temporal_alpha,
        )?,
        SemanticRegion::Mouth => gpu.semantic_temporal_mask(
            &mut ws.parser_attribute_512,
            &mut ws.semantic_previous_mouth,
            &ws.semantic_count,
            &mut ws.semantic_mouth_valid,
            temporal_alpha,
        )?,
    }
    gpu.semantic_region_stats(
        &ws.face_512,
        &ws.face_512_original,
        &ws.parser_attribute_512,
        &mut ws.semantic_stats,
    )?;
    gpu.semantic_composite(
        &mut ws.face_512,
        &ws.face_512_original,
        &ws.parser_attribute_512,
        &ws.semantic_stats,
        &ws.semantic_count,
        blend,
        luminance_factor,
    )?;
    Ok(())
}

fn compose_faceparser_gpu(
    gpu: &GpuOps,
    manager: &mut ModelManager,
    ws: &mut GpuWorkspace,
    params: &FaceParserMaskParams,
) -> anyhow::Result<()> {
    gpu.imagenet_normalize_copy_512(&ws.face_512_pre_restorer, &mut ws.face_512_scratch)?;
    crate::pipeline::ort_binding::run_bound_f32(
        manager,
        &gpu.stream,
        "FaceParser",
        "input",
        &ws.face_512_scratch,
        &[1, 3, 512, 512],
        "output",
        &mut ws.parser_logits,
        &[1, 19, 512, 512],
    )?;
    gpu.parser_argmax(&ws.parser_logits, &mut ws.parser_classes)?;

    if params.background == 0 {
        gpu.border_oval_mask(&mut ws.parser_mask_512, 512, 0, 0, 0, 0, false)?;
    } else {
        gpu.parser_class_mask(&ws.parser_classes, &mut ws.parser_mask_512, 0, true)?;
        gpu.morphology_mask(
            &mut ws.parser_mask_512,
            &mut ws.parser_tmp_512,
            512,
            512,
            params.background,
        )?;
        blur_parser_mask(gpu, ws, ParserMaskBuffer::Output, params.background_blur)?;
    }

    for (class_id, amount) in parser_attributes(params) {
        if amount == 0 {
            continue;
        }
        gpu.parser_class_mask(
            &ws.parser_classes,
            &mut ws.parser_attribute_512,
            class_id,
            false,
        )?;
        gpu.morphology_mask(
            &mut ws.parser_attribute_512,
            &mut ws.parser_tmp_512,
            512,
            512,
            amount as i32,
        )?;
        gpu.mask_invert(&mut ws.parser_attribute_512)?;
        blur_parser_mask(gpu, ws, ParserMaskBuffer::Attribute, params.face_blur)?;
        gpu.mask_mul(&mut ws.parser_mask_512, &ws.parser_attribute_512)?;
    }

    gpu.mask_resize(
        &ws.parser_mask_512,
        &mut ws.mask_128_tmp,
        512,
        512,
        128,
        128,
    )?;
    gpu.mask_mul(&mut ws.mask_learned_128, &ws.mask_128_tmp)?;
    Ok(())
}

#[derive(Clone, Copy)]
enum ParserMaskBuffer {
    Output,
    Attribute,
}

fn blur_parser_mask(
    gpu: &GpuOps,
    ws: &mut GpuWorkspace,
    target: ParserMaskBuffer,
    amount: u32,
) -> anyhow::Result<()> {
    if amount == 0 {
        return Ok(());
    }
    let kernel_size = amount.saturating_mul(2).saturating_add(1);
    let sigma = (amount as f32 + 1.0) * 0.2;
    let ks = prepare_blur_kernel(gpu, ws, kernel_size, sigma)?;
    match target {
        ParserMaskBuffer::Output => gpu.gaussian_blur_mask(
            &mut ws.parser_mask_512,
            &mut ws.parser_tmp_512,
            512,
            512,
            &ws.blur_kernel,
            ks,
        )?,
        ParserMaskBuffer::Attribute => gpu.gaussian_blur_mask(
            &mut ws.parser_attribute_512,
            &mut ws.parser_tmp_512,
            512,
            512,
            &ws.blur_kernel,
            ks,
        )?,
    }
    Ok(())
}

fn parser_attributes(params: &FaceParserMaskParams) -> [(u32, u32); 12] {
    [
        (1, params.face),
        (2, params.left_eyebrow),
        (3, params.right_eyebrow),
        (4, params.left_eye),
        (5, params.right_eye),
        (6, params.eyeglasses),
        (10, params.nose),
        (11, params.mouth),
        (12, params.upper_lip),
        (13, params.lower_lip),
        (14, params.neck),
        (17, params.hair),
    ]
}

fn prepare_learned_mask_input(gpu: &GpuOps, ws: &mut GpuWorkspace) -> anyhow::Result<()> {
    gpu.resize_npp(&ws.face_512_original, &mut ws.face_256, 512, 512, 256, 256)?;
    gpu.normalize_prefix(&mut ws.face_256, 3 * 256 * 256)?;
    Ok(())
}

fn compose_learned_mask(
    gpu: &GpuOps,
    ws: &mut GpuWorkspace,
    blur_amount: u32,
) -> anyhow::Result<()> {
    gpu.mask_resize(&ws.mask_256, &mut ws.mask_128_tmp, 256, 256, 128, 128)?;
    gpu.mask_mul(&mut ws.mask_learned_128, &ws.mask_128_tmp)?;
    if blur_amount > 0 {
        let blur_ks = blur_amount.saturating_mul(2).saturating_add(1);
        let sigma = (blur_amount as f32 + 1.0) * 0.2;
        let ks = prepare_blur_kernel(gpu, ws, blur_ks, sigma)?;
        gpu.gaussian_blur_mask(
            &mut ws.mask_learned_128,
            &mut ws.mask_128_tmp,
            128,
            128,
            &ws.blur_kernel,
            ks,
        )?;
    }
    Ok(())
}

pub(crate) fn prepare_blur_kernel(
    gpu: &GpuOps,
    ws: &mut GpuWorkspace,
    kernel_size: u32,
    sigma: f32,
) -> anyhow::Result<u32> {
    let ks = kernel_size.min(MAX_BLUR_KS as u32);
    if ks != ws.blur_ks_current || (sigma - ws.blur_sigma_current).abs() > 1e-6 {
        let half = (ks as i32) / 2;
        let mut weights = vec![0.0f32; ks as usize];
        let mut sum = 0.0f32;
        for i in 0..ks as i32 {
            let x = (i - half) as f32;
            let value = (-x * x / (2.0 * sigma * sigma)).exp();
            weights[i as usize] = value;
            sum += value;
        }
        for weight in &mut weights {
            *weight /= sum;
        }
        gpu.upload_into(&weights, &mut ws.blur_kernel)?;
        ws.blur_ks_current = ks;
        ws.blur_sigma_current = sigma;
    }
    Ok(ks)
}

/// Paste swapped face back into frame using mask.
///
/// `frame_chw` — full frame [3, H, W] in [0, 255], MODIFIED in place.
/// `swap_chw` — swapped face [3, face_size, face_size] in [0, 255].
/// `mask` — face mask [face_size, face_size] where 1.0 = swap, 0.0 = keep original.
/// `affine` — 2×3 affine matrix (face coords → frame coords).
pub fn paste_back(
    frame_chw: &mut [f32],
    frame_h: u32,
    frame_w: u32,
    swap_chw: &[f32],
    mask: &[f32],
    face_size: u32,
    affine: &[[f64; 3]; 2],
) {
    // affine maps frame_kps → face_template (src→dst).
    // To paste back: iterate frame pixels, use FORWARD affine to find face pixel.
    // affine(frame_pixel) → face_pixel (sample from swap).
    let a = affine;
    let fs = face_size as f32;

    // Compute bounding box of the face in frame space using inverse transform
    let inv = crate::math::affine::invert_2x3(affine);
    let corners = [
        (0.0, 0.0),
        (fs as f64, 0.0),
        (0.0, fs as f64),
        (fs as f64, fs as f64),
    ];
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;
    for (cx, cy) in corners {
        let fx = inv[0][0] * cx + inv[0][1] * cy + inv[0][2];
        let fy = inv[1][0] * cx + inv[1][1] * cy + inv[1][2];
        min_x = min_x.min(fx);
        min_y = min_y.min(fy);
        max_x = max_x.max(fx);
        max_y = max_y.max(fy);
    }

    let left = (min_x.floor() as i32).max(0) as u32;
    let top = (min_y.floor() as i32).max(0) as u32;
    let right = (max_x.ceil() as i32).min(frame_w as i32) as u32;
    let bottom = (max_y.ceil() as i32).min(frame_h as i32) as u32;

    // Inverse mapping: for each frame pixel in bbox, find face pixel via forward affine
    for fy in top..bottom {
        for fx in left..right {
            // Forward affine: frame → face
            let sx = (a[0][0] * fx as f64 + a[0][1] * fy as f64 + a[0][2]) as f32;
            let sy = (a[1][0] * fx as f64 + a[1][1] * fy as f64 + a[1][2]) as f32;

            // Check bounds in face space
            if sx < 0.0 || sy < 0.0 || sx >= fs - 1.0 || sy >= fs - 1.0 {
                continue;
            }

            // Bilinear sample from swap face and mask
            let x0 = sx as u32;
            let y0 = sy as u32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;
            let wx = sx - x0 as f32;
            let wy = sy - y0 as f32;

            let face_idx = |yy: u32, xx: u32| (yy * face_size + xx) as usize;

            // Sample mask
            let m = mask[face_idx(y0, x0)] * (1.0 - wx) * (1.0 - wy)
                + mask[face_idx(y0, x1)] * wx * (1.0 - wy)
                + mask[face_idx(y1, x0)] * (1.0 - wx) * wy
                + mask[face_idx(y1, x1)] * wx * wy;

            if m < 0.001 {
                continue;
            }

            for c in 0..3u32 {
                let cidx =
                    |yy: u32, xx: u32| (c * face_size * face_size + yy * face_size + xx) as usize;

                // Bilinear sample from swap
                let sv = swap_chw[cidx(y0, x0)] * (1.0 - wx) * (1.0 - wy)
                    + swap_chw[cidx(y0, x1)] * wx * (1.0 - wy)
                    + swap_chw[cidx(y1, x0)] * (1.0 - wx) * wy
                    + swap_chw[cidx(y1, x1)] * wx * wy;

                let frame_idx = (c * frame_h * frame_w + fy * frame_w + fx) as usize;
                let orig = frame_chw[frame_idx];
                frame_chw[frame_idx] = sv * m + orig * (1.0 - m);
            }
        }
    }
}
