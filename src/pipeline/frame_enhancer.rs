use thiserror::Error;

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
            scale,
            padded_width,
            padded_height,
            output_width,
            output_height,
            tiles,
        })
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
