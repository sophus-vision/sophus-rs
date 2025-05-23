use crate::{
    ImageSize,
    ImageView2U8,
    ImageView2U16,
    ImageView3F32,
    ImageView3U8,
    ImageView3U16,
    ImageView4F32,
    ImageView4U8,
    ImageView4U16,
    ImageViewF32,
    ImageViewU8,
    ImageViewU16,
    prelude::*,
};

/// Intensity image view of unsigned integer values.
pub trait IsIntensityViewImageU<'a> {
    /// Color type of the image
    const PNG_COLOR_TYPE: png::ColorType;
    /// Bit depth of the image
    const BIT_DEPTH: png::BitDepth;

    /// Size of the image
    fn size(&'a self) -> ImageSize;

    /// raw u8 slice of the image
    fn raw_u8_slice(&self) -> &[u8];
}

impl<'a> IsIntensityViewImageU<'a> for ImageViewU8<'a> {
    const PNG_COLOR_TYPE: png::ColorType = png::ColorType::Grayscale;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Eight;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        self.as_scalar_slice()
    }
}

impl<'a> IsIntensityViewImageU<'a> for ImageView2U8<'a> {
    const PNG_COLOR_TYPE: png::ColorType = png::ColorType::GrayscaleAlpha;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Eight;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        self.as_scalar_slice()
    }
}

impl<'a> IsIntensityViewImageU<'a> for ImageView3U8<'a> {
    const PNG_COLOR_TYPE: png::ColorType = png::ColorType::Rgb;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Eight;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        self.as_scalar_slice()
    }
}

impl<'a> IsIntensityViewImageU<'a> for ImageView4U8<'a> {
    const PNG_COLOR_TYPE: png::ColorType = png::ColorType::Rgba;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Eight;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        self.as_scalar_slice()
    }
}

impl<'a> IsIntensityViewImageU<'a> for ImageViewU16<'a> {
    const PNG_COLOR_TYPE: png::ColorType = png::ColorType::Grayscale;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Sixteen;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_scalar_slice())
    }
}

impl<'a> IsIntensityViewImageU<'a> for ImageView2U16<'a> {
    const PNG_COLOR_TYPE: png::ColorType = png::ColorType::GrayscaleAlpha;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Sixteen;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_scalar_slice())
    }
}

impl<'a> IsIntensityViewImageU<'a> for ImageView3U16<'a> {
    const PNG_COLOR_TYPE: png::ColorType = png::ColorType::Rgb;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Sixteen;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_scalar_slice())
    }
}

impl<'a> IsIntensityViewImageU<'a> for ImageView4U16<'a> {
    const PNG_COLOR_TYPE: png::ColorType = png::ColorType::Rgba;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Sixteen;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_scalar_slice())
    }
}

/// Intensity image view of unsigned integer values.
pub trait IsIntensityViewImageF32<'a> {
    /// Color type of the image
    type TiffColorType: tiff::encoder::colortype::ColorType;

    /// Size of the image
    fn size(&'a self) -> ImageSize;

    /// raw uf32 slice of the image
    fn raw_f32_slice(
        &self,
    ) -> &[<<Self as IsIntensityViewImageF32<'a>>::TiffColorType as tiff::encoder::colortype::ColorType>::Inner];
}

impl<'a> IsIntensityViewImageF32<'a> for ImageViewF32<'a> {
    type TiffColorType = tiff::encoder::colortype::Gray32Float;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_f32_slice(&self) -> &[f32] {
        self.as_scalar_slice()
    }
}

impl<'a> IsIntensityViewImageF32<'a> for ImageView3F32<'a> {
    type TiffColorType = tiff::encoder::colortype::RGB32Float;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_f32_slice(&self) -> &[f32] {
        self.as_scalar_slice()
    }
}

impl<'a> IsIntensityViewImageF32<'a> for ImageView4F32<'a> {
    type TiffColorType = tiff::encoder::colortype::RGBA32Float;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_f32_slice(&self) -> &[f32] {
        self.as_scalar_slice()
    }
}
