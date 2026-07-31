use cudarc::driver::{DevicePtr, DevicePtrMut};
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::ValueType;

use crate::gpu::ops::GpuOps;
use crate::models::manager::ModelManager;
use crate::pipeline::ort_binding::{bind_input_raw, bind_output_raw, create_cuda_tensor_f32};
use crate::pipeline::workspace::GpuWorkspace;

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
