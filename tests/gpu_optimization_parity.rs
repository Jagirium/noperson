use std::path::Path;
use std::sync::Arc;

use noperson::backend::ComputeOps;
use noperson::backend::{ComputeContext, DevicePtrMut};
use noperson::pipeline::frame_processor::{AssignmentBackend, GenerationGpuState, SourceFace};
use noperson::{
    config::parameters::ColorAdjustParams,
    pipeline::{
        color::{adjust_color_reference, dfl_transfer_reference},
        dfm::rct_reference,
    },
};

#[path = "support/morphology.rs"]
mod morphology;
#[path = "support/nv12.rs"]
mod nv12;

fn basis(index: usize) -> Vec<f32> {
    let mut embedding = vec![0.0; 512];
    embedding[index] = 1.0;
    embedding
}

fn source(target_embedding: Option<Vec<f32>>, threshold: f32, marker: f32) -> SourceFace {
    SourceFace {
        target_embedding,
        backend: AssignmentBackend::Inswapper {
            latent: vec![marker; 512],
        },
        threshold,
        params: None,
    }
}

fn assert_close(actual: f32, expected: f32, context: &str) {
    assert!(
        (actual - expected).abs() < 1e-3,
        "{context}: actual={actual}, expected={expected}"
    );
}

fn chw(pixels: &[[f32; 3]]) -> Vec<f32> {
    (0..3)
        .flat_map(|channel| pixels.iter().map(move |pixel| pixel[channel]))
        .collect()
}

#[test]
#[ignore = "single consolidated CUDA verification gate"]
fn optimized_gpu_paths_match_references() -> anyhow::Result<()> {
    noperson::backend::cuda::npp::initialize_runtime(Path::new("libs/base"))?;
    let context = Arc::new(ComputeContext::new(0)?);
    let stream = context.new_stream()?;
    let gpu = ComputeOps::new(&context, stream)?;

    let sources = vec![
        source(Some(basis(1)), 0.75, 1.0),
        source(None, 1.0, 2.0),
        source(Some(basis(0)), 1.0, 3.0),
    ];
    let mut state = GenerationGpuState::new(&gpu, &sources)?;
    let query = gpu.upload(&basis(0))?;
    assert_eq!(state.select_first_source(&gpu, &query)?, Some(1));

    let inclusive_sources = vec![
        source(Some(basis(0)), 1.000_001, 4.0),
        source(Some(basis(0)), 1.0, 5.0),
        source(Some(basis(0)), 1.0, 6.0),
    ];
    let mut inclusive_state = GenerationGpuState::new(&gpu, &inclusive_sources)?;
    assert_eq!(inclusive_state.select_first_source(&gpu, &query)?, Some(1));

    let no_match_sources = vec![source(Some(basis(1)), 0.500_001, 7.0)];
    let mut no_match_state = GenerationGpuState::new(&gpu, &no_match_sources)?;
    assert_eq!(no_match_state.select_first_source(&gpu, &query)?, None);

    let zero_query = gpu.upload(&vec![0.0; 512])?;
    let zero_sources = vec![source(Some(vec![0.0; 512]), 0.5, 8.0)];
    let mut zero_state = GenerationGpuState::new(&gpu, &zero_sources)?;
    assert_eq!(zero_state.select_first_source(&gpu, &zero_query)?, Some(0));

    let latent = state
        .resident_latent(2)?
        .expect("Inswapper source must have a resident latent");
    let latent = gpu.download(latent)?;
    assert_eq!(latent.len(), 16 * 512);
    assert!(latent.iter().all(|value| *value == 3.0));

    for fixture in morphology::FIXTURES {
        for amount in morphology::AMOUNTS {
            let mut mask = gpu.upload(fixture.input)?;
            let mut tmp = gpu.alloc_zeros(fixture.input.len())?;
            gpu.morphology_mask(
                &mut mask,
                &mut tmp,
                fixture.width as u32,
                fixture.height as u32,
                amount,
            )?;
            let expected =
                morphology::scalar_reference(fixture.input, fixture.width, fixture.height, amount);
            if amount > 0 {
                assert_eq!(
                    expected,
                    morphology::repeated_clamped_max(
                        fixture.input,
                        fixture.width,
                        fixture.height,
                        amount as usize,
                    ),
                    "positive morphology oracle for {} at amount {amount}",
                    fixture.name,
                );
            }
            assert_eq!(
                gpu.download(&mask)?,
                expected,
                "morphology fixture {} at amount {amount}",
                fixture.name,
            );
        }
    }

    let rct_source = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.2, 0.3, 0.4],
    ];
    let rct_like = [
        [0.9, 0.8, 0.7],
        [0.6, 0.5, 0.4],
        [0.3, 0.2, 0.1],
        [0.8, 0.7, 0.6],
    ];
    let rct_mask = [1.0, 0.0, 1.0, 0.0];
    let mut rct_actual = gpu.upload(&rct_source.concat())?;
    let rct_like_gpu = gpu.upload(&rct_like.concat())?;
    let rct_mask_gpu = gpu.upload(&rct_mask)?;
    let mut rct_stats = gpu.alloc_zeros(12)?;
    let mut float_partials = gpu.alloc_zeros(1024 * 13)?;
    gpu.dfm_rct(
        &mut rct_actual,
        &rct_like_gpu,
        &rct_mask_gpu,
        &mut rct_stats,
        &mut float_partials,
        4,
        0.2,
    )?;
    let rct_actual = gpu.download(&rct_actual)?;
    for (pixel, expected) in rct_reference(&rct_source, &rct_like, &rct_mask, 0.2)
        .iter()
        .enumerate()
    {
        for channel in 0..3 {
            assert_close(
                rct_actual[pixel * 3 + channel],
                expected[channel],
                "DFM RCT reduction",
            );
        }
    }

    let dfl_original = [
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
        [70.0, 80.0, 90.0],
        [100.0, 110.0, 120.0],
    ];
    let dfl_swapped = [
        [120.0, 110.0, 100.0],
        [90.0, 80.0, 70.0],
        [60.0, 50.0, 40.0],
        [30.0, 20.0, 10.0],
    ];
    let dfl_mask = [0.2, 0.199_999, 1.0, 0.0];
    let dfl_original_gpu = gpu.upload(&chw(&dfl_original))?;
    let mut dfl_actual = gpu.upload(&chw(&dfl_swapped))?;
    let dfl_mask_gpu = gpu.upload(&dfl_mask)?;
    let mut dfl_stats = gpu.alloc_zeros(13)?;
    gpu.auto_color_dfl(
        &dfl_original_gpu,
        &mut dfl_actual,
        &dfl_mask_gpu,
        &mut dfl_stats,
        &mut float_partials,
        4,
        true,
        0.8,
    )?;
    let dfl_actual = gpu.download(&dfl_actual)?;
    for (pixel, expected) in
        dfl_transfer_reference(&dfl_original, &dfl_swapped, Some(&dfl_mask), 0.8)
            .iter()
            .enumerate()
    {
        for channel in 0..3 {
            assert_close(
                dfl_actual[channel * 4 + pixel],
                expected[channel],
                "DFL auto-color reduction",
            );
        }
    }

    let color_pixels = [
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
        [70.0, 80.0, 90.0],
        [100.0, 110.0, 120.0],
    ];
    let mut color_image = gpu.upload(&chw(&color_pixels))?;
    let mut color_scratch = gpu.alloc_zeros(12)?;
    let mut gray_sum = gpu.stream.alloc_zeros::<u32>(1)?;
    let mut integer_partials = gpu.stream.alloc_zeros::<u32>(1024)?;
    let controls = ColorAdjustParams {
        enabled: true,
        gamma: 1.0,
        brightness: 1.0,
        contrast: 1.0,
        saturation: 1.0,
        sharpness: 1.0,
        hue: 0.0,
        noise: 0.0,
        ..ColorAdjustParams::default()
    };
    gpu.adjust_color(
        &mut color_image,
        &mut color_scratch,
        &mut gray_sum,
        &mut integer_partials,
        2,
        2,
        controls.gamma,
        [controls.red, controls.green, controls.blue],
        controls.brightness,
        controls.contrast,
        controls.saturation,
        controls.sharpness,
        controls.hue,
        controls.noise,
        0,
    )?;
    let mut gray_total = [0u32];
    gpu.stream.memcpy_dtoh(&gray_sum, &mut gray_total)?;
    assert_eq!(gray_total, [252]);
    let color_actual = gpu.download(&color_image)?;
    for (pixel, expected) in adjust_color_reference(&color_pixels, 2, 2, &controls)
        .iter()
        .enumerate()
    {
        for channel in 0..3 {
            assert_close(
                color_actual[channel * 4 + pixel],
                expected[channel],
                "color gray-sum reduction",
            );
        }
    }

    const SEMANTIC_PIXELS: usize = 512 * 512;
    let mut classes = vec![0u8; SEMANTIC_PIXELS];
    classes[..7].copy_from_slice(&[0, 4, 5, 11, 12, 13, 18]);
    let classes = gpu.upload_u8(&classes)?;
    let mut semantic_mask = gpu.alloc_zeros(SEMANTIC_PIXELS)?;
    let mut semantic_count = gpu.stream.alloc_zeros::<u32>(1)?;
    gpu.semantic_region_mask(
        &classes,
        &mut semantic_mask,
        &mut semantic_count,
        &mut integer_partials,
        0,
    )?;
    let mut count = [0u32];
    gpu.stream.memcpy_dtoh(&semantic_count, &mut count)?;
    assert_eq!(count, [2]);

    let swapped = gpu.upload(&vec![10.0; SEMANTIC_PIXELS * 3])?;
    let original = gpu.upload(&vec![20.0; SEMANTIC_PIXELS * 3])?;
    let mut semantic_stats = gpu.alloc_zeros(8)?;
    gpu.semantic_region_stats(
        &swapped,
        &original,
        &semantic_mask,
        &mut semantic_stats,
        &mut float_partials,
    )?;
    let semantic_stats = gpu.download(&semantic_stats)?;
    assert_eq!(
        semantic_stats,
        vec![
            262_142.0,
            2.0,
            2_621_420.0,
            2_621_420.0,
            2_621_420.0,
            40.0,
            40.0,
            40.0
        ]
    );

    let nv12_source = nv12::nonuniform_chw(3, 5);
    let nv12_source_gpu = gpu.upload(&nv12_source)?;
    let matrices = [
        (
            nv12::Matrix::Bt601,
            noperson::io::native_video::ColorMatrix::Bt601,
        ),
        (
            nv12::Matrix::Bt709,
            noperson::io::native_video::ColorMatrix::Bt709,
        ),
        (
            nv12::Matrix::Bt2020Ncl,
            noperson::io::native_video::ColorMatrix::Bt2020NonConstantLuminance,
        ),
        (
            nv12::Matrix::Unspecified,
            noperson::io::native_video::ColorMatrix::Unspecified,
        ),
    ];
    for (reference_matrix, gpu_matrix) in matrices {
        for (reference_range, gpu_range) in [
            (
                nv12::Range::Limited,
                noperson::io::native_video::ColorRange::Limited,
            ),
            (
                nv12::Range::Full,
                noperson::io::native_video::ColorRange::Full,
            ),
        ] {
            for (reference_format, gpu_format) in [
                (
                    nv12::Format::Nv12,
                    noperson::io::native_video::PixelFormat::Nv12,
                ),
                (
                    nv12::Format::P010,
                    noperson::io::native_video::PixelFormat::P010,
                ),
            ] {
                let bytes_per_sample = match reference_format {
                    nv12::Format::Nv12 => 1,
                    nv12::Format::P010 => 2,
                };
                let pitch = 4 * bytes_per_sample + 7;
                let expected = nv12::encode_scalar(
                    &nv12_source,
                    3,
                    5,
                    4,
                    4,
                    pitch,
                    reference_matrix,
                    reference_range,
                    reference_format,
                );
                let mut output = gpu.upload_u8(&vec![0xa5; expected.len()])?;
                {
                    let (device_ptr, _guard) = output.device_ptr_mut(&gpu.stream);
                    unsafe {
                        gpu.chw_f32_to_pitched_nv12_scaled_color(
                            &nv12_source_gpu,
                            device_ptr,
                            pitch as u32,
                            3,
                            5,
                            4,
                            4,
                            gpu_matrix,
                            gpu_range,
                            gpu_format,
                        )?;
                    }
                }
                assert_eq!(
                    gpu.download_u8(&output)?,
                    expected,
                    "NV12/P010 macroblock parity for {reference_matrix:?}/{reference_range:?}/{reference_format:?}"
                );
            }
        }
    }

    // Exercise the other Unspecified dispatch branch at the 1280-pixel
    // boundary. The compact odd-sized source keeps this parity case cheap
    // while still covering resize, both ranges, both formats, and padding.
    for (reference_range, gpu_range) in [
        (
            nv12::Range::Limited,
            noperson::io::native_video::ColorRange::Limited,
        ),
        (
            nv12::Range::Full,
            noperson::io::native_video::ColorRange::Full,
        ),
    ] {
        for (reference_format, gpu_format) in [
            (
                nv12::Format::Nv12,
                noperson::io::native_video::PixelFormat::Nv12,
            ),
            (
                nv12::Format::P010,
                noperson::io::native_video::PixelFormat::P010,
            ),
        ] {
            let bytes_per_sample = match reference_format {
                nv12::Format::Nv12 => 1,
                nv12::Format::P010 => 2,
            };
            let dst_h = 2;
            let dst_w = 1280;
            let pitch = dst_w * bytes_per_sample + 7;
            let expected = nv12::encode_scalar(
                &nv12_source,
                3,
                5,
                dst_h,
                dst_w,
                pitch,
                nv12::Matrix::Unspecified,
                reference_range,
                reference_format,
            );
            let mut output = gpu.upload_u8(&vec![0xa5; expected.len()])?;
            {
                let (device_ptr, _guard) = output.device_ptr_mut(&gpu.stream);
                unsafe {
                    gpu.chw_f32_to_pitched_nv12_scaled_color(
                        &nv12_source_gpu,
                        device_ptr,
                        pitch as u32,
                        3,
                        5,
                        dst_h as u32,
                        dst_w as u32,
                        noperson::io::native_video::ColorMatrix::Unspecified,
                        gpu_range,
                        gpu_format,
                    )?;
                }
            }
            assert_eq!(
                gpu.download_u8(&output)?,
                expected,
                "Unspecified >=1280 must dispatch BT.709 for {reference_range:?}/{reference_format:?}"
            );
        }
    }

    Ok(())
}
