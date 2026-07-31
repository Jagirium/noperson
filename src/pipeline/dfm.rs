use cudarc::driver::{DevicePtr, DevicePtrMut};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::ValueType;

use crate::gpu::ops::GpuOps;
use crate::models::manager::ModelManager;
use crate::pipeline::ort_binding::{bind_input_raw, bind_output_raw, create_cuda_tensor_f32};
use crate::pipeline::workspace::GpuWorkspace;

pub fn rgb_to_lab(rgb: [f32; 3]) -> [f32; 3] {
    let linear = rgb.map(|value| {
        if value > 0.04045 {
            ((value + 0.055) / 1.055).powf(2.4)
        } else {
            value / 12.92
        }
    });
    let normalized = [
        (0.412453 * linear[0] + 0.357580 * linear[1] + 0.180423 * linear[2]) / 0.95047,
        0.212671 * linear[0] + 0.715160 * linear[1] + 0.072169 * linear[2],
        (0.019334 * linear[0] + 0.119193 * linear[1] + 0.950227 * linear[2]) / 1.08883,
    ];
    let f = normalized.map(|value| {
        if value > 0.008856 {
            value.max(0.008856).powf(1.0 / 3.0)
        } else {
            7.787 * value + 4.0 / 29.0
        }
    });
    [
        116.0 * f[1] - 16.0,
        500.0 * (f[0] - f[1]),
        200.0 * (f[1] - f[2]),
    ]
}

#[allow(clippy::excessive_precision)] // Kornia matrix coefficients; f32 rounds at compile time.
pub fn lab_to_rgb(lab: [f32; 3]) -> [f32; 3] {
    let fy = (lab[0] + 16.0) / 116.0;
    let f = [lab[1] / 500.0 + fy, fy, (fy - lab[2] / 200.0).max(0.0)];
    let xyz_normalized = f.map(|value| {
        if value > 0.2068966 {
            value.powi(3)
        } else {
            (value - 4.0 / 29.0) / 7.787
        }
    });
    let xyz = [
        xyz_normalized[0] * 0.95047,
        xyz_normalized[1],
        xyz_normalized[2] * 1.08883,
    ];
    let linear = [
        3.2404813432005266 * xyz[0] - 1.5371515162713185 * xyz[1] - 0.4985363261688878 * xyz[2],
        -0.9692549499965682 * xyz[0] + 1.8759900014898907 * xyz[1] + 0.0415559265582928 * xyz[2],
        0.0556466391351772 * xyz[0] - 0.2040413383665112 * xyz[1] + 1.0573110696453443 * xyz[2],
    ];
    linear.map(|value| {
        let srgb = if value > 0.0031308 {
            1.055 * value.max(0.0031308).powf(1.0 / 2.4) - 0.055
        } else {
            12.92 * value
        };
        srgb.clamp(0.0, 1.0)
    })
}

pub fn rct_reference(
    source: &[[f32; 3]],
    like: &[[f32; 3]],
    mask: &[f32],
    cutoff: f32,
) -> Vec<[f32; 3]> {
    assert_eq!(source.len(), like.len());
    assert_eq!(source.len(), mask.len());
    let source_lab: Vec<_> = source.iter().copied().map(rgb_to_lab).collect();
    let like_lab: Vec<_> = like.iter().copied().map(rgb_to_lab).collect();
    let count = source.len() as f32;
    let mut source_mean = [0.0; 3];
    let mut like_mean = [0.0; 3];
    for pixel in 0..source.len() {
        if mask[pixel] >= cutoff {
            for channel in 0..3 {
                source_mean[channel] += source_lab[pixel][channel];
                like_mean[channel] += like_lab[pixel][channel];
            }
        }
    }
    for channel in 0..3 {
        source_mean[channel] /= count;
        like_mean[channel] /= count;
    }
    let mut source_std = [0.0; 3];
    let mut like_std = [0.0; 3];
    for pixel in 0..source.len() {
        for channel in 0..3 {
            let source_value = if mask[pixel] >= cutoff {
                source_lab[pixel][channel]
            } else {
                0.0
            };
            let like_value = if mask[pixel] >= cutoff {
                like_lab[pixel][channel]
            } else {
                0.0
            };
            source_std[channel] += (source_value - source_mean[channel]).powi(2);
            like_std[channel] += (like_value - like_mean[channel]).powi(2);
        }
    }
    let correction = (count - 1.0).max(1.0);
    for channel in 0..3 {
        source_std[channel] = (source_std[channel] / correction).sqrt();
        like_std[channel] = (like_std[channel] / correction).sqrt();
    }

    source_lab
        .into_iter()
        .map(|mut lab| {
            for channel in 0..3 {
                lab[channel] = (lab[channel] - source_mean[channel])
                    * (like_std[channel] / (source_std[channel] + 1e-6))
                    + like_mean[channel];
            }
            lab[0] = lab[0].clamp(0.0, 100.0);
            lab[1] = lab[1].clamp(-127.0, 127.0);
            lab[2] = lab[2].clamp(-127.0, 127.0);
            lab_to_rgb(lab)
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DfmTensor {
    pub name: String,
    pub shape: Vec<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DfmContract {
    width: usize,
    height: usize,
    has_morph_value: bool,
}

impl DfmContract {
    pub fn from_session(session: &Session) -> anyhow::Result<Self> {
        let inputs = session
            .inputs()
            .iter()
            .map(session_tensor)
            .collect::<anyhow::Result<Vec<_>>>()?;
        let outputs = session
            .outputs()
            .iter()
            .map(session_tensor)
            .collect::<anyhow::Result<Vec<_>>>()?;
        Self::from_io(&inputs, &outputs)
    }

    pub fn from_io(inputs: &[DfmTensor], outputs: &[DfmTensor]) -> anyhow::Result<Self> {
        anyhow::ensure!(
            matches!(inputs.len(), 1 | 2),
            "DFM requires one or two inputs"
        );
        let input = &inputs[0];
        anyhow::ensure!(input.name == "in_face:0", "DFM input must be in_face:0");
        anyhow::ensure!(input.shape.len() == 4, "DFM input must be NHWC");
        let shape = Self::convert_shape(&input.shape);
        anyhow::ensure!(shape[3] == 3, "DFM input must have three channels");
        let height = shape[1];
        let width = shape[2];
        anyhow::ensure!(
            width > 0 && height > 0 && width <= 512 && height <= 512,
            "DFM input dimensions must fit the 512 workspace"
        );

        let has_morph_value = inputs.len() == 2;
        if let Some(morph) = inputs.get(1) {
            anyhow::ensure!(morph.name == "morph_value:0", "invalid DFM morph input");
        }

        for (name, channels) in [
            ("out_face_mask:0", 1),
            ("out_celeb_face:0", 3),
            ("out_celeb_face_mask:0", 1),
        ] {
            let output = outputs
                .iter()
                .find(|output| output.name == name)
                .ok_or_else(|| anyhow::anyhow!("DFM output {name} is missing"))?;
            anyhow::ensure!(output.shape.len() == 4, "DFM output {name} must be NHWC");
            let output_shape = Self::convert_shape(&output.shape);
            anyhow::ensure!(
                output_shape[1] == height
                    && output_shape[2] == width
                    && output_shape[3] == channels,
                "DFM output {name} has incompatible shape"
            );
        }

        Ok(Self {
            width,
            height,
            has_morph_value,
        })
    }

    pub fn input_size(self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn has_morph_value(self) -> bool {
        self.has_morph_value
    }

    pub fn convert_shape(shape: &[i64]) -> Vec<usize> {
        shape
            .iter()
            .map(|dimension| usize::try_from(*dimension).unwrap_or(1).max(1))
            .collect()
    }

    pub fn convert_gpu(
        self,
        manager: &mut ModelManager,
        gpu: &GpuOps,
        workspace: &mut GpuWorkspace,
        morph: f32,
        rct: bool,
    ) -> anyhow::Result<()> {
        let width = self.width as u32;
        let height = self.height as u32;
        let pixels = width * height;
        gpu.resize_npp(
            &workspace.face_512_original,
            &mut workspace.face_256,
            512,
            512,
            height,
            width,
        )?;
        gpu.chw_rgb_to_nhwc_bgr_unit(&workspace.face_256, &mut workspace.face_512_scratch, pixels)?;
        gpu.stream
            .memcpy_htod(&[morph.clamp(0.01, 1.0)], &mut workspace.dfm_morph)?;

        let memory = MemoryInfo::new(
            AllocationDevice::CUDA,
            manager.device_id(),
            AllocatorType::Device,
            MemoryType::Default,
        )?;
        let input_shape = [1, height as i64, width as i64, 3];
        let mask_shape = [1, height as i64, width as i64, 1];
        let face_shape = [1, height as i64, width as i64, 3];
        {
            let (input_ptr, _input_guard) = workspace.face_512_scratch.device_ptr(&gpu.stream);
            let (morph_ptr, _morph_guard) = workspace.dfm_morph.device_ptr(&gpu.stream);
            let (face_mask_ptr, _face_mask_guard) =
                workspace.parser_mask_512.device_ptr_mut(&gpu.stream);
            let (face_ptr, _face_guard) = workspace.face_256.device_ptr_mut(&gpu.stream);
            let (celeb_mask_ptr, _celeb_mask_guard) =
                workspace.parser_attribute_512.device_ptr_mut(&gpu.stream);

            let input = unsafe { create_cuda_tensor_f32(&memory, input_ptr, &input_shape)? };
            let morph_value = unsafe { create_cuda_tensor_f32(&memory, morph_ptr, &[1])? };
            let face_mask = unsafe { create_cuda_tensor_f32(&memory, face_mask_ptr, &mask_shape)? };
            let face = unsafe { create_cuda_tensor_f32(&memory, face_ptr, &face_shape)? };
            let celeb_mask =
                unsafe { create_cuda_tensor_f32(&memory, celeb_mask_ptr, &mask_shape)? };

            let (session, binding) = manager.session_and_binding("DFM")?;
            unsafe {
                bind_input_raw(binding, "in_face:0", &input)?;
                if self.has_morph_value {
                    bind_input_raw(binding, "morph_value:0", &morph_value)?;
                }
                bind_output_raw(binding, "out_face_mask:0", &face_mask)?;
                bind_output_raw(binding, "out_celeb_face:0", &face)?;
                bind_output_raw(binding, "out_celeb_face_mask:0", &celeb_mask)?;
            }
            binding.synchronize_inputs()?;
            let _ = session.run_binding(binding)?;
            binding.synchronize_outputs()?;
            binding.clear();
        }

        if rct {
            gpu.dfm_rct(
                &mut workspace.face_256,
                &workspace.face_512_scratch,
                &workspace.parser_attribute_512,
                &mut workspace.dfm_rct_stats,
                pixels,
                0.3,
            )?;
        }

        gpu.nhwc_bgr_unit_to_chw_rgb(&workspace.face_256, &mut workspace.face_512_scratch, pixels)?;
        gpu.resize_npp(
            &workspace.face_512_scratch,
            &mut workspace.face_512,
            height,
            width,
            512,
            512,
        )?;
        Ok(())
    }
}

fn session_tensor(value: &ort::value::Outlet) -> anyhow::Result<DfmTensor> {
    let ValueType::Tensor { shape, .. } = value.dtype() else {
        anyhow::bail!("DFM {} is not a tensor", value.name());
    };
    Ok(DfmTensor {
        name: value.name().to_owned(),
        shape: shape.as_ref().to_vec(),
    })
}
