//! Face masks: Occluder, border mask, gaussian blur, soft paste-back.
//!
//! Port of crosswap/app/processors/face_masks.py + frame_worker.py mask logic.
//!
//! Mask convention: 1.0 = keep swapped face, 0.0 = keep original background.

use crate::config::parameters::{FaceParserMaskParams, FaceSwapParams};
use crate::gpu::{ops::GpuOps, unified};
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
                let sx = (x as i32 + k as i32 - half).clamp(0, width as i32 - 1) as usize;
                val += mask[y * w + sx] * weight;
            }
            tmp[y * w + x] = val;
        }
    }

    // Vertical pass
    for y in 0..h {
        for x in 0..w {
            let mut val = 0.0f32;
            for (k, weight) in kernel.iter().enumerate() {
                let sy = (y as i32 + k as i32 - half).clamp(0, height as i32 - 1) as usize;
                val += tmp[sy * w + x] * weight;
            }
            mask[y * w + x] = val;
        }
    }
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
    if !params.occluder_enabled && !params.xseg_enabled && !params.faceparser_enabled {
        return Ok(false);
    }

    gpu.border_oval_mask(&mut ws.mask_learned_128, 128, 0, 0, 0, 0, false)?;

    if params.occluder_enabled {
        prepare_learned_mask_input(gpu, ws)?;
        let session = manager
            .get_mut("Occluder")
            .ok_or_else(|| anyhow::anyhow!("Occluder is not loaded"))?;
        unified::run_occluder(session, &gpu.stream, &mut ws.face_256, &mut ws.mask_256)?;
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
        let session = manager
            .get_mut("XSeg")
            .ok_or_else(|| anyhow::anyhow!("XSeg is not loaded"))?;
        unified::run_xseg(session, &gpu.stream, &mut ws.face_256, &mut ws.mask_256)?;
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

fn compose_faceparser_gpu(
    gpu: &GpuOps,
    manager: &mut ModelManager,
    ws: &mut GpuWorkspace,
    params: &FaceParserMaskParams,
) -> anyhow::Result<()> {
    gpu.stream
        .memcpy_dtod(&ws.face_512_pre_restorer, &mut ws.face_512_scratch)?;
    gpu.imagenet_normalize_512(&mut ws.face_512_scratch)?;
    let session = manager
        .get_mut("FaceParser")
        .ok_or_else(|| anyhow::anyhow!("FaceParser is not loaded"))?;
    unified::run_faceparser(
        session,
        &gpu.stream,
        &mut ws.face_512_scratch,
        &mut ws.parser_logits,
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

fn prepare_blur_kernel(
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
