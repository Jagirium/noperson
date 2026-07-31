use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use thiserror::Error;

use crate::config::settings::ExecutionProvider;
use crate::gpu::{ops::GpuOps, unified};
use crate::models::{manager::ModelManager, registry::find_model};

/// Tile size selected by CrossSwap's production `FrameWorker.enhance_core`.
pub const CROSSSWAP_TILE_SIZE: u32 = 512;

/// Frame-enhancer models exposed by the CrossSwap backend contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnhancerModel {
    RealEsrganX2Plus,
    RealEsrganX4Plus,
    RealEsrGeneralX4V3,
    BsrganX2,
    BsrganX4,
    UltraSharpX4,
    UltraMixX4,
}

impl EnhancerModel {
    pub fn from_crosswap_name(name: &str) -> Result<Self, FrameEnhancerError> {
        match name {
            "RealEsrgan-x2-Plus" => Ok(Self::RealEsrganX2Plus),
            "RealEsrgan-x4-Plus" => Ok(Self::RealEsrganX4Plus),
            "RealEsr-General-x4v3" => Ok(Self::RealEsrGeneralX4V3),
            "BSRGan-x2" => Ok(Self::BsrganX2),
            "BSRGan-x4" => Ok(Self::BsrganX4),
            "UltraSharp-x4" => Ok(Self::UltraSharpX4),
            "UltraMix-x4" => Ok(Self::UltraMixX4),
            _ => Err(FrameEnhancerError::UnknownModel(name.to_owned())),
        }
    }

    pub const fn crosswap_name(self) -> &'static str {
        match self {
            Self::RealEsrganX2Plus => "RealEsrgan-x2-Plus",
            Self::RealEsrganX4Plus => "RealEsrgan-x4-Plus",
            Self::RealEsrGeneralX4V3 => "RealEsr-General-x4v3",
            Self::BsrganX2 => "BSRGan-x2",
            Self::BsrganX4 => "BSRGan-x4",
            Self::UltraSharpX4 => "UltraSharp-x4",
            Self::UltraMixX4 => "UltraMix-x4",
        }
    }

    pub const fn registry_name(self) -> &'static str {
        match self {
            Self::RealEsrganX2Plus => "RealEsrganx2Plus",
            Self::RealEsrganX4Plus => "RealEsrganx4Plus",
            Self::RealEsrGeneralX4V3 => "RealEsrx4v3",
            Self::BsrganX2 => "BSRGANx2",
            Self::BsrganX4 => "BSRGANx4",
            Self::UltraSharpX4 => "UltraSharpx4",
            Self::UltraMixX4 => "UltraMixx4",
        }
    }

    pub const fn scale(self) -> u32 {
        match self {
            Self::RealEsrganX2Plus | Self::BsrganX2 => 2,
            Self::RealEsrganX4Plus
            | Self::RealEsrGeneralX4V3
            | Self::BsrganX4
            | Self::UltraSharpX4
            | Self::UltraMixX4 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnhancerTile {
    pub batch_index: usize,
    pub input_x: u32,
    pub input_y: u32,
    pub output_x: u32,
    pub output_y: u32,
}

/// Complete geometry for CrossSwap's single-batch tile enhancer pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilePlan {
    pub input_width: u32,
    pub input_height: u32,
    pub tile_size: u32,
    pub output_tile_size: u32,
    pub scale: u32,
    pub padded_width: u32,
    pub padded_height: u32,
    pub output_width: u32,
    pub output_height: u32,
    pub tiles: Vec<EnhancerTile>,
}

impl TilePlan {
    pub fn new(
        width: u32,
        height: u32,
        tile_size: u32,
        scale: u32,
    ) -> Result<Self, FrameEnhancerError> {
        if width == 0 || height == 0 || tile_size == 0 || scale == 0 {
            return Err(FrameEnhancerError::ZeroDimension);
        }

        let tiles_x = width.div_ceil(tile_size);
        let tiles_y = height.div_ceil(tile_size);
        let padded_width = tiles_x
            .checked_mul(tile_size)
            .ok_or(FrameEnhancerError::DimensionOverflow)?;
        let padded_height = tiles_y
            .checked_mul(tile_size)
            .ok_or(FrameEnhancerError::DimensionOverflow)?;
        let output_width = width
            .checked_mul(scale)
            .ok_or(FrameEnhancerError::DimensionOverflow)?;
        let output_height = height
            .checked_mul(scale)
            .ok_or(FrameEnhancerError::DimensionOverflow)?;
        let output_tile_size = tile_size
            .checked_mul(scale)
            .ok_or(FrameEnhancerError::DimensionOverflow)?;
        let tile_count = tiles_x
            .checked_mul(tiles_y)
            .ok_or(FrameEnhancerError::DimensionOverflow)? as usize;
        let mut tiles = Vec::new();
        tiles
            .try_reserve_exact(tile_count)
            .map_err(|_| FrameEnhancerError::PlanTooLarge)?;

        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                let input_x = tile_x * tile_size;
                let input_y = tile_y * tile_size;
                tiles.push(EnhancerTile {
                    batch_index: tiles.len(),
                    input_x,
                    input_y,
                    output_x: input_x
                        .checked_mul(scale)
                        .ok_or(FrameEnhancerError::DimensionOverflow)?,
                    output_y: input_y
                        .checked_mul(scale)
                        .ok_or(FrameEnhancerError::DimensionOverflow)?,
                });
            }
        }

        Ok(Self {
            input_width: width,
            input_height: height,
            tile_size,
            output_tile_size,
            scale,
            padded_width,
            padded_height,
            output_width,
            output_height,
            tiles,
        })
    }

    pub fn input_shape(&self) -> [usize; 4] {
        [
            self.tiles.len(),
            3,
            self.tile_size as usize,
            self.tile_size as usize,
        ]
    }

    pub fn batched_input_elements(&self) -> Result<usize, FrameEnhancerError> {
        checked_elements(&self.input_shape())
    }

    pub fn batched_output_elements(&self) -> Result<usize, FrameEnhancerError> {
        checked_elements(&[
            self.tiles.len(),
            3,
            self.output_tile_size as usize,
            self.output_tile_size as usize,
        ])
    }

    pub fn output_elements(&self) -> Result<usize, FrameEnhancerError> {
        checked_elements(&[3, self.output_height as usize, self.output_width as usize])
    }
}

fn checked_elements(dimensions: &[usize]) -> Result<usize, FrameEnhancerError> {
    dimensions.iter().try_fold(1usize, |elements, dimension| {
        elements
            .checked_mul(*dimension)
            .ok_or(FrameEnhancerError::DimensionOverflow)
    })
}

struct EnhancerWorkspace {
    tiles_in: CudaSlice<f32>,
    tiles_out: CudaSlice<f32>,
    enhanced_frame: CudaSlice<f32>,
}

impl EnhancerWorkspace {
    fn matches(&self, plan: &TilePlan) -> Result<bool, FrameEnhancerError> {
        Ok(self.tiles_in.len() == plan.batched_input_elements()?
            && self.tiles_out.len() == plan.batched_output_elements()?
            && self.enhanced_frame.len() == plan.output_elements()?)
    }
}

/// GPU-resident CrossSwap frame enhancer with reusable shape-specific buffers.
pub struct FrameEnhancer {
    gpu: Arc<GpuOps>,
    manager: ModelManager,
    model: EnhancerModel,
    tile_size: u32,
    workspace: Option<EnhancerWorkspace>,
}

impl FrameEnhancer {
    pub fn new(
        gpu: Arc<GpuOps>,
        models_dir: impl AsRef<Path>,
        provider: ExecutionProvider,
        device_id: i32,
        model: EnhancerModel,
    ) -> anyhow::Result<Self> {
        let registry_name = model.registry_name();
        let artifact = find_model(registry_name)
            .ok_or_else(|| anyhow::anyhow!("enhancer model {registry_name} is not registered"))?;
        let mut manager = ModelManager::with_execution(models_dir, provider, device_id);
        manager.set_compute_stream(gpu.stream.cu_stream() as *mut ());
        manager.load(registry_name, artifact.filename)?;
        Ok(Self {
            gpu,
            manager,
            model,
            tile_size: CROSSSWAP_TILE_SIZE,
            workspace: None,
        })
    }

    pub const fn model(&self) -> EnhancerModel {
        self.model
    }

    pub fn plan(&self, width: u32, height: u32) -> Result<TilePlan, FrameEnhancerError> {
        TilePlan::new(width, height, self.tile_size, self.model.scale())
    }

    /// Enhance one raw [0,255] CHW frame into a caller-owned output buffer.
    ///
    /// The output dimensions are `width*scale × height*scale`. All inference,
    /// crop, resize, and blending stay on the shared CUDA stream.
    pub fn enhance_into(
        &mut self,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        width: u32,
        height: u32,
        blend: f32,
    ) -> anyhow::Result<TilePlan> {
        let plan = self.plan(width, height)?;
        let input_elements = checked_elements(&[3, height as usize, width as usize])?;
        let output_elements = plan.output_elements()?;
        anyhow::ensure!(
            input.len() >= input_elements,
            "enhancer input buffer has {} elements, needs {input_elements}",
            input.len()
        );
        anyhow::ensure!(
            output.len() >= output_elements,
            "enhancer output buffer has {} elements, needs {output_elements}",
            output.len()
        );
        self.ensure_workspace(&plan)?;

        let tiles_x = plan.padded_width / plan.tile_size;
        let tile_count = u32::try_from(plan.tiles.len())
            .map_err(|_| anyhow::anyhow!("enhancer tile count exceeds CUDA limits"))?;
        let workspace = self
            .workspace
            .as_mut()
            .expect("enhancer workspace was allocated above");

        self.gpu.enhancer_pack_tiles(
            input,
            &mut workspace.tiles_in,
            height,
            width,
            tiles_x,
            plan.tile_size,
            tile_count,
        )?;
        self.gpu.normalize(&mut workspace.tiles_in)?;

        let session = self
            .manager
            .get_mut(self.model.registry_name())
            .ok_or_else(|| anyhow::anyhow!("enhancer session disappeared"))?;
        unified::run_enhancer(
            session,
            &self.gpu.stream,
            &mut workspace.tiles_in,
            &plan.input_shape(),
            &mut workspace.tiles_out,
        )?;
        self.gpu.enhancer_scatter_tiles(
            &workspace.tiles_out,
            &mut workspace.enhanced_frame,
            plan.output_height,
            plan.output_width,
            tiles_x,
            plan.output_tile_size,
        )?;
        self.gpu.denormalize(&mut workspace.enhanced_frame)?;

        self.gpu.resize(
            input,
            output,
            height,
            width,
            plan.output_height,
            plan.output_width,
            3,
        )?;
        self.gpu
            .scalar_blend_inplace(&workspace.enhanced_frame, output, output_elements, blend)?;
        Ok(plan)
    }

    fn ensure_workspace(&mut self, plan: &TilePlan) -> anyhow::Result<()> {
        if self
            .workspace
            .as_ref()
            .map(|workspace| workspace.matches(plan))
            .transpose()?
            .unwrap_or(false)
        {
            return Ok(());
        }

        let next = EnhancerWorkspace {
            tiles_in: self.gpu.alloc_zeros(plan.batched_input_elements()?)?,
            tiles_out: self.gpu.alloc_zeros(plan.batched_output_elements()?)?,
            enhanced_frame: self.gpu.alloc_zeros(plan.output_elements()?)?,
        };
        self.workspace = Some(next);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FrameEnhancerError {
    #[error("unknown CrossSwap frame enhancer {0}")]
    UnknownModel(String),
    #[error("frame, tile, and scale dimensions must be non-zero")]
    ZeroDimension,
    #[error("frame enhancer dimensions overflow")]
    DimensionOverflow,
    #[error("frame enhancer tile plan is too large")]
    PlanTooLarge,
}
