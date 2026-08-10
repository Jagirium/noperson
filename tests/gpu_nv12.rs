#[cfg(not(noperson_static_test))]
use std::sync::Arc;

#[cfg(not(noperson_static_test))]
use cudarc::driver::CudaContext;
#[cfg(not(noperson_static_test))]
use noperson::gpu::ops::GpuOps;

#[derive(Clone, Copy, Debug)]
enum Matrix {
    Bt601,
    Bt709,
    Bt2020Ncl,
    Unspecified,
}

#[derive(Clone, Copy, Debug)]
enum Range {
    Limited,
    Full,
}

#[derive(Clone, Copy, Debug)]
enum Format {
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

fn rounded_u8(value: f32) -> u8 {
    value.clamp(0.0, 255.0).round_ties_even() as u8
}

fn write_code(output: &mut [u8], offset: usize, code: f32, format: Format) {
    match format {
        Format::Nv12 => output[offset] = rounded_u8(code),
        Format::P010 => {
            let sample = ((code * 4.0).clamp(0.0, 1023.0) + 0.5) as u16;
            output[offset..offset + 2].copy_from_slice(&(sample << 6).to_le_bytes());
        }
    }
}

/// Independent scalar byte oracle. It deliberately does not use GPU wrappers
/// or CUDA helpers, and initializes every pitch byte with a sentinel so the
/// parity check also proves that padding is untouched.
fn encode_scalar(
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
                let y_code = coefficients[0] * r
                    + coefficients[1] * g
                    + coefficients[2] * b
                    + coefficients[3];
                write_code(
                    &mut output,
                    sample_y * pitch + sample_x * bytes_per_sample,
                    y_code,
                    format,
                );
            }
            let r = sum_r * 0.25;
            let g = sum_g * 0.25;
            let b = sum_b * 0.25;
            let u_code =
                coefficients[4] * r + coefficients[5] * g + coefficients[6] * b + coefficients[10];
            let v_code =
                coefficients[7] * r + coefficients[8] * g + coefficients[9] * b + coefficients[10];
            let uv = pitch * dst_h + (y / 2) * pitch + x * bytes_per_sample;
            write_code(&mut output, uv, u_code, format);
            write_code(&mut output, uv + bytes_per_sample, v_code, format);
        }
    }
    output
}

fn nonuniform_chw(height: usize, width: usize) -> Vec<f32> {
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

#[test]
fn scalar_nv12_p010_reference_covers_nonuniform_resize_matrix_range_and_pitch() {
    assert_eq!(rounded_u8(-1.0), 0);
    assert_eq!(rounded_u8(2.5), 2);
    assert_eq!(rounded_u8(3.5), 4);
    assert_eq!(rounded_u8(256.0), 255);

    let fixtures = [(2, 2, 2, 2), (4, 4, 4, 4), (3, 5, 4, 4)];
    let combinations = [
        (Matrix::Bt601, Range::Limited),
        (Matrix::Bt709, Range::Limited),
        (Matrix::Bt2020Ncl, Range::Limited),
        (Matrix::Bt601, Range::Full),
        (Matrix::Bt709, Range::Full),
        (Matrix::Bt2020Ncl, Range::Full),
    ];
    for (src_h, src_w, dst_h, dst_w) in fixtures {
        let chw = nonuniform_chw(src_h, src_w);
        for (matrix, range) in combinations {
            for format in [Format::Nv12, Format::P010] {
                let bytes_per_sample = match format {
                    Format::Nv12 => 1,
                    Format::P010 => 2,
                };
                let pitch = dst_w * bytes_per_sample + 7;
                let actual = encode_scalar(
                    &chw, src_h, src_w, dst_h, dst_w, pitch, matrix, range, format,
                );
                for row in 0..dst_h + dst_h / 2 {
                    assert!(
                        actual[row * pitch + dst_w * bytes_per_sample..(row + 1) * pitch]
                            .iter()
                            .all(|byte| *byte == 0xa5)
                    );
                }
                assert!(actual.iter().any(|byte| *byte != 0xa5));
            }
        }
    }

    for range in [Range::Limited, Range::Full] {
        assert_eq!(
            coefficients(Matrix::Unspecified, range, 1278),
            coefficients(Matrix::Bt601, range, 1278),
            "Unspecified below 1280 must select BT.601"
        );
        assert_eq!(
            coefficients(Matrix::Unspecified, range, 1280),
            coefficients(Matrix::Bt709, range, 1280),
            "Unspecified at or above 1280 must select BT.709"
        );
    }
}

#[test]
fn scalar_encoder_matches_literal_nonuniform_nv12_and_p010_bytes() {
    // Planar 2x2 RGB. Source equals destination size, so each output sample is
    // one explicit source pixel; the expected bytes are literal BT.601-limited
    // results and do not come from another implementation of the oracle.
    let chw = [
        0.0, 40.0, 80.0, 120.0, // R
        10.0, 50.0, 90.0, 130.0, // G
        20.0, 60.0, 100.0, 140.0, // B
    ];

    assert_eq!(
        encode_scalar(
            &chw,
            2,
            2,
            2,
            2,
            4,
            Matrix::Bt601,
            Range::Limited,
            Format::Nv12,
        ),
        vec![
            23, 57, 0xa5, 0xa5, 92, 126, 0xa5, 0xa5, 134, 123, 0xa5, 0xa5
        ]
    );
    assert_eq!(
        encode_scalar(
            &chw,
            2,
            2,
            2,
            2,
            6,
            Matrix::Bt601,
            Range::Limited,
            Format::P010,
        ),
        vec![
            0, 23, 64, 57, 0xa5, 0xa5, 192, 91, 0, 126, 0xa5, 0xa5, 192, 133, 0, 123, 0xa5, 0xa5,
        ]
    );
}

#[test]
fn nv12_encoder_static_contract_launches_one_thread_per_macroblock() {
    let kernel_source = include_str!("../gpu_kernels/frame_convert.cu");
    let kernel = kernel_source
        .split_once("void chw_f32_to_nv12_scaled_kernel(")
        .expect("NV12 encoder entry point must exist")
        .1
        .split_once("// Letterbox resize + normalize")
        .expect("NV12 encoder must end before letterbox encoder")
        .0;
    let launches = include_str!("../src/gpu/ops.rs");
    assert!(kernel.contains("unsigned int total = (dst_h / 2) * (dst_w / 2);"));
    assert!(kernel.contains("unsigned int macro_index = idx;"));
    assert!(kernel.contains("unsigned int x = (macro_index % macro_w) * 2;"));
    assert!(kernel.contains("unsigned int y = (macro_index / macro_w) * 2;"));
    assert_eq!(
        kernel.matches("sample_chw_bilinear(").count(),
        12,
        "the macroblock thread must sample four RGB triplets exactly once"
    );
    let expected_sample_order = [
        "float r00 = sample_chw_bilinear(",
        "float g00 = sample_chw_bilinear(",
        "float b00 = sample_chw_bilinear(",
        "float r01 = sample_chw_bilinear(",
        "float g01 = sample_chw_bilinear(",
        "float b01 = sample_chw_bilinear(",
        "float r10 = sample_chw_bilinear(",
        "float g10 = sample_chw_bilinear(",
        "float b10 = sample_chw_bilinear(",
        "float r11 = sample_chw_bilinear(",
        "float g11 = sample_chw_bilinear(",
        "float b11 = sample_chw_bilinear(",
    ];
    let mut previous = None;
    for sample in expected_sample_order {
        let position = kernel
            .find(sample)
            .expect("required retained sample is missing");
        if let Some(previous) = previous {
            assert!(
                position > previous,
                "RGB samples must stay ordered 00/01/10/11"
            );
        }
        previous = Some(position);
    }
    let required_y_writes = [
        "store_p010_sample(p010_output, x, y00_10);",
        "store_p010_sample(p010_output, x + 1, y01_10);",
        "store_p010_sample(p010_next_row, x, y10_10);",
        "store_p010_sample(p010_next_row, x + 1, y11_10);",
        "y_row[x] = round_clamp_u8(y00);",
        "y_row[x + 1] = round_clamp_u8(y01);",
        "y_next_row[x] = round_clamp_u8(y10);",
        "y_next_row[x + 1] = round_clamp_u8(y11);",
    ];
    for write in required_y_writes {
        assert!(kernel.contains(write), "missing Y association: {write}");
    }
    for expression in [
        "float y00 = yr * r00 + yg * g00 + yb * b00 + y_offset;",
        "float y01 = yr * r01 + yg * g01 + yb * b01 + y_offset;",
        "float y10 = yr * r10 + yg * g10 + yb * b10 + y_offset;",
        "float y11 = yr * r11 + yg * g11 + yb * b11 + y_offset;",
    ] {
        assert!(
            kernel.contains(expression),
            "missing exact Y expression: {expression}"
        );
    }
    for intermediate in [
        "float y00_10 = fminf(fmaxf(y00 * 4.0f, 0.0f), 1023.0f);",
        "float y01_10 = fminf(fmaxf(y01 * 4.0f, 0.0f), 1023.0f);",
        "float y10_10 = fminf(fmaxf(y10 * 4.0f, 0.0f), 1023.0f);",
        "float y11_10 = fminf(fmaxf(y11 * 4.0f, 0.0f), 1023.0f);",
        "float u10 = fminf(fmaxf(u_code * 4.0f, 0.0f), 1023.0f);",
        "float v10 = fminf(fmaxf(v_code * 4.0f, 0.0f), 1023.0f);",
    ] {
        assert!(
            kernel.contains(intermediate),
            "missing exact P010 scale/clamp: {intermediate}"
        );
    }
    let ordered_sums = [
        "sum_r += r00;",
        "sum_g += g00;",
        "sum_b += b00;",
        "sum_r += r01;",
        "sum_g += g01;",
        "sum_b += b01;",
        "sum_r += r10;",
        "sum_g += g10;",
        "sum_b += b10;",
        "sum_r += r11;",
        "sum_g += g11;",
        "sum_b += b11;",
    ];
    let mut previous = None;
    for sum in ordered_sums {
        let position = kernel.find(sum).expect("ordered RGB sum is missing");
        if let Some(previous) = previous {
            assert!(
                position > previous,
                "RGB sums must stay ordered 00/01/10/11"
            );
        }
        previous = Some(position);
    }
    for formula in [
        "float r = sum_r * 0.25f;",
        "float g = sum_g * 0.25f;",
        "float b = sum_b * 0.25f;",
        "float u_code = ur * r + ug * g + ub * b + uv_offset;",
        "float v_code = vr * r + vg * g + vb * b + uv_offset;",
        "store_p010_sample(p010_output, x, u10);",
        "store_p010_sample(p010_output, x + 1, v10);",
        "uv_row[x] = round_clamp_u8(u_code);",
        "uv_row[x + 1] = round_clamp_u8(v_code);",
    ] {
        assert!(
            kernel.contains(formula),
            "missing chroma formula: {formula}"
        );
    }
    assert!(
        !kernel.contains("if ((x & 1u) != 0u || (y & 1u) != 0u) return;"),
        "the old per-luma-thread early return must not reappear"
    );
    assert!(
        !kernel.contains("for ("),
        "the retained RGB values must not be resampled in a chroma loop"
    );
    assert_eq!(kernel.matches("round_clamp_u8(").count(), 6);
    assert_eq!(kernel.matches("store_p010_sample(").count(), 6);
    assert!(!kernel.contains("unsigned short* p010_output"));
    assert!(kernel_source.contains("const unsigned int byte_offset = sample_index * 2;"));
    assert!(kernel_source.contains("row[byte_offset] = (unsigned char)(packed & 0xffu);"));
    assert!(kernel_source.contains("row[byte_offset + 1] = (unsigned char)(packed >> 8);"));
    assert!(
        kernel_source
            .contains("return (unsigned char)__float2int_rn(fminf(fmaxf(value, 0.0f), 255.0f));")
    );
    assert!(launches.contains("let total = (dst_h / 2) * (dst_w / 2);"));
    assert!(launches.contains("ColorMatrix::Unspecified if dst_w < 1280 => ColorMatrix::Bt601"));
    assert!(launches.contains("ColorMatrix::Unspecified => ColorMatrix::Bt709"));
}

#[test]
#[cfg(not(noperson_static_test))]
#[ignore = "requires CUDA"]
fn converts_and_scales_solid_green_to_nv12_on_gpu() -> anyhow::Result<()> {
    let ctx = Arc::new(CudaContext::new(0)?);
    let stream = ctx.new_stream()?;
    let gpu = GpuOps::new(&ctx, stream)?;

    // 2x2 planar RGB: R plane, G plane, B plane.
    let chw = [0.0; 4]
        .into_iter()
        .chain([255.0; 4])
        .chain([0.0; 4])
        .collect::<Vec<_>>();
    let source = gpu.upload(&chw)?;
    let mut nv12 = gpu.alloc_zeros_u8(4 * 4 * 3 / 2)?;

    gpu.chw_f32_to_nv12_scaled(&source, &mut nv12, 2, 2, 4, 4)?;
    let output = gpu.download_u8(&nv12)?;

    assert!(output[..16].iter().all(|&value| value == 145));
    assert!(output[16..].chunks_exact(2).all(|uv| uv == [54, 34]));
    Ok(())
}
