#[derive(Clone, Copy, Debug)]
pub enum Matrix {
    Bt601,
    Bt709,
    Bt2020Ncl,
    Unspecified,
}

#[derive(Clone, Copy, Debug)]
pub enum Range {
    Limited,
    Full,
}

#[derive(Clone, Copy, Debug)]
pub enum Format {
    Nv12,
    P010,
}

fn coefficients(matrix: Matrix, range: Range, dst_w: usize) -> [f32; 11] {
    let matrix = match matrix {
        Matrix::Unspecified if dst_w < 1280 => Matrix::Bt601,
        Matrix::Unspecified => Matrix::Bt709,
        matrix => matrix,
    };
    match (matrix, range) {
        (Matrix::Bt601, Range::Limited) => [
            0.257, 0.504, 0.098, 16.0, -0.148, -0.291, 0.439, 0.439, -0.368, -0.071, 128.0,
        ],
        (Matrix::Bt709, Range::Limited) => [
            0.183, 0.614, 0.062, 16.0, -0.101, -0.339, 0.439, 0.439, -0.399, -0.040, 128.0,
        ],
        (Matrix::Bt2020Ncl, Range::Limited) => [
            0.2256, 0.5823, 0.0509, 16.0, -0.1227, -0.3166, 0.4392, 0.4392, -0.4039, -0.0353, 128.0,
        ],
        (Matrix::Bt601, Range::Full) => [
            0.299, 0.587, 0.114, 0.0, -0.168_736, -0.331_264, 0.5, 0.5, -0.418_688, -0.081_312,
            128.0,
        ],
        (Matrix::Bt709, Range::Full) => [
            0.2126, 0.7152, 0.0722, 0.0, -0.114_572, -0.385_428, 0.5, 0.5, -0.454_153, -0.045_847,
            128.0,
        ],
        (Matrix::Bt2020Ncl, Range::Full) => [
            0.2627, 0.678, 0.0593, 0.0, -0.139_63, -0.360_37, 0.5, 0.5, -0.459_786, -0.040_214,
            128.0,
        ],
        (Matrix::Unspecified, _) => unreachable!("resolution selects an explicit matrix"),
    }
}

fn bilinear(
    chw: &[f32],
    channel: usize,
    src_h: usize,
    src_w: usize,
    dst_y: usize,
    dst_x: usize,
    dst_h: usize,
    dst_w: usize,
) -> f32 {
    let sx = (((dst_x as f32) + 0.5) * (src_w as f32 / dst_w as f32) - 0.5)
        .clamp(0.0, (src_w - 1) as f32);
    let sy = (((dst_y as f32) + 0.5) * (src_h as f32 / dst_h as f32) - 0.5)
        .clamp(0.0, (src_h - 1) as f32);
    let x0 = sx as usize;
    let y0 = sy as usize;
    let x1 = (x0 + 1).min(src_w - 1);
    let y1 = (y0 + 1).min(src_h - 1);
    let fx = sx - x0 as f32;
    let fy = sy - y0 as f32;
    let plane = src_h * src_w;
    let source = &chw[channel * plane..][..plane];
    let top = source[y0 * src_w + x0] * (1.0 - fx) + source[y0 * src_w + x1] * fx;
    let bottom = source[y1 * src_w + x0] * (1.0 - fx) + source[y1 * src_w + x1] * fx;
    top * (1.0 - fy) + bottom * fy
}

fn write_code(output: &mut [u8], offset: usize, code: f32, format: Format) {
    match format {
        Format::Nv12 => output[offset] = code.clamp(0.0, 255.0).round_ties_even() as u8,
        Format::P010 => {
            let sample = ((code * 4.0).clamp(0.0, 1023.0) + 0.5) as u16;
            output[offset..offset + 2].copy_from_slice(&(sample << 6).to_le_bytes());
        }
    }
}

pub fn encode_scalar(
    chw: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    pitch: usize,
    matrix: Matrix,
    range: Range,
    format: Format,
) -> Vec<u8> {
    let bytes_per_sample = match format {
        Format::Nv12 => 1,
        Format::P010 => 2,
    };
    assert!(dst_h > 0 && dst_w > 0 && dst_h % 2 == 0 && dst_w % 2 == 0);
    assert!(pitch >= dst_w * bytes_per_sample);
    assert_eq!(chw.len(), 3 * src_h * src_w);
    let coefficients = coefficients(matrix, range, dst_w);
    let mut output = vec![0xa5; pitch * (dst_h + dst_h / 2)];
    for y in (0..dst_h).step_by(2) {
        for x in (0..dst_w).step_by(2) {
            let samples = [(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)];
            let mut rgb = [[0.0; 3]; 4];
            let mut sum_r = 0.0;
            let mut sum_g = 0.0;
            let mut sum_b = 0.0;
            for (index, (sample_y, sample_x)) in samples.into_iter().enumerate() {
                let r = bilinear(chw, 0, src_h, src_w, sample_y, sample_x, dst_h, dst_w);
                let g = bilinear(chw, 1, src_h, src_w, sample_y, sample_x, dst_h, dst_w);
                let b = bilinear(chw, 2, src_h, src_w, sample_y, sample_x, dst_h, dst_w);
                rgb[index] = [r, g, b];
                sum_r += r;
                sum_g += g;
                sum_b += b;
            }
            for (index, (sample_y, sample_x)) in samples.into_iter().enumerate() {
                let [r, g, b] = rgb[index];
                write_code(
                    &mut output,
                    sample_y * pitch + sample_x * bytes_per_sample,
                    coefficients[0] * r
                        + coefficients[1] * g
                        + coefficients[2] * b
                        + coefficients[3],
                    format,
                );
            }
            let r = sum_r * 0.25;
            let g = sum_g * 0.25;
            let b = sum_b * 0.25;
            let uv = pitch * dst_h + (y / 2) * pitch + x * bytes_per_sample;
            write_code(
                &mut output,
                uv,
                coefficients[4] * r + coefficients[5] * g + coefficients[6] * b + coefficients[10],
                format,
            );
            write_code(
                &mut output,
                uv + bytes_per_sample,
                coefficients[7] * r + coefficients[8] * g + coefficients[9] * b + coefficients[10],
                format,
            );
        }
    }
    output
}

pub fn nonuniform_chw(height: usize, width: usize) -> Vec<f32> {
    (0..3)
        .flat_map(|channel| {
            (0..height * width).map(move |index| {
                let x = index % width;
                let y = index / width;
                (channel * 47 + x * 19 + y * 31 + (x * y) * 7) as f32
            })
        })
        .collect()
}
