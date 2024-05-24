use crate::image_view::ImageView2U16;
use crate::image_view::ImageView2U8;
use crate::image_view::ImageView3U16;
use crate::image_view::ImageView3U8;
use crate::image_view::ImageView4U16;
use crate::image_view::ImageView4U8;
use crate::image_view::ImageViewU16;
use crate::image_view::ImageViewU8;
use crate::prelude::*;
use crate::ImageSize;

/// Intensity image view of unsigned integer values.
pub trait IsIntensityViewImageU<'a> {
    /// Color type of the image
    const COLOR_TYPE: png::ColorType;
    /// Bit depth of the image
    const BIT_DEPTH: png::BitDepth;

    /// Size of the image
    fn size(&'a self) -> ImageSize;

    /// raw u8 slice of the image
    fn raw_u8_slice(&self) -> &[u8];
}

impl<'a> IsIntensityViewImageU<'a> for ImageViewU8<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Grayscale;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Eight;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        self.as_scalar_slice()
    }
}

impl<'a> IsIntensityViewImageU<'a> for ImageView2U8<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::GrayscaleAlpha;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Eight;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        self.as_scalar_slice()
    }
}

impl<'a> IsIntensityViewImageU<'a> for ImageView3U8<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Rgb;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Eight;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        self.as_scalar_slice()
    }
}

impl<'a> IsIntensityViewImageU<'a> for ImageView4U8<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Rgba;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Eight;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        self.as_scalar_slice()
    }
}

impl<'a> IsIntensityViewImageU<'a> for ImageViewU16<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Grayscale;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Sixteen;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_scalar_slice())
    }
}

impl<'a> IsIntensityViewImageU<'a> for ImageView2U16<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::GrayscaleAlpha;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Sixteen;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_scalar_slice())
    }
}

impl<'a> IsIntensityViewImageU<'a> for ImageView3U16<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Rgb;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Sixteen;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_scalar_slice())
    }
}

impl<'a> IsIntensityViewImageU<'a> for ImageView4U16<'a> {
    const COLOR_TYPE: png::ColorType = png::ColorType::Rgba;
    const BIT_DEPTH: png::BitDepth = png::BitDepth::Sixteen;

    fn size(&'a self) -> ImageSize {
        self.image_size()
    }

    fn raw_u8_slice(&self) -> &[u8] {
        bytemuck::cast_slice(self.as_scalar_slice())
    }
}
