use super::data::DataChunkVec;
use super::dyn_mut_view::DynMutImageView;
use super::dyn_mut_view::DynMutImageViewTrait;
use super::dyn_view::DynImageView;
use super::dyn_view::DynImageViewTrait;

use super::mut_image::MutImage;

use super::mut_image::MutImageTrait;
use super::mut_image::MutIntensityImageEnum;
use super::mut_view::MutImageViewTrait;

use super::pixel::PixelFormat;
use super::pixel::PixelTag;
use super::pixel::P1F32;

use super::pixel::ScalarTrait;

use crate::image::layout::ImageLayout;
use crate::image::layout::ImageLayoutTrait;
use crate::image::layout::ImageSize;
use crate::image::layout::ImageSizeTrait;

pub struct MutAnyImage {
    pub buffer: DataChunkVec,
    pub layout: ImageLayout,
    pub pixel_format: PixelFormat,
}

impl ImageSizeTrait for MutAnyImage {
    fn size(&self) -> ImageSize {
        self.dyn_view().size()
    }
}

impl ImageLayoutTrait for MutAnyImage {
    fn layout(&self) -> ImageLayout {
        self.dyn_view().layout()
    }
}

impl<'a> DynImageViewTrait<'a> for MutAnyImage {
    fn dyn_view(&self) -> DynImageView<'_> {
        let layout = self.layout;
        let pixel_format = self.pixel_format;
        let byte_slice = self.buffer.slice::<u8>();

        DynImageView {
            layout,
            byte_slice,
            pixel_format,
        }
    }
}

impl<'a> DynMutImageViewTrait<'a> for MutAnyImage {
    fn dyn_mut_view(&mut self) -> DynMutImageView<'_> {
        let layout = self.layout;
        let pixel_format = self.pixel_format;

        let mut_byte_slice = self.buffer.mut_slice::<u8>();

        DynMutImageView {
            layout,
            mut_byte_slice,
            pixel_format,
        }
    }
}

impl MutAnyImage {
    pub fn from_mut_image<const NUM: usize, Scalar: ScalarTrait + 'static>(
        img: MutImage<NUM, Scalar>,
    ) -> Self {
        let layout = img.layout();
        let pixel_format = PixelFormat {
            number_category: Scalar::NUMBER_CATEGORY,
            num_scalars: NUM,
            num_bytes_per_scalar: std::mem::size_of::<Scalar>(),
        };

        Self {
            buffer: img.buffer,
            layout,
            pixel_format,
        }
    }
}

trait DynMutImageTrait {}

impl DynMutImageTrait for MutAnyImage {}

#[test]
fn from_mut_image() {
    let size_2_x_3 = ImageSize::from_width_and_height(2, 3);
    let img_f32 = MutImage::<1, f32>::with_size_and_val(size_2_x_3, P1F32::new(0.25));
    let _dyn_img = MutAnyImage::from_mut_image(img_f32);
}

#[derive(Debug, Clone)]
pub struct MutIntensityImage {
    pub buffer: MutIntensityImageEnum,
    pub layout: ImageLayout,
    pub pixel_format: PixelFormat,
}

impl ImageSizeTrait for MutIntensityImage {
    fn size(&self) -> ImageSize {
        self.dyn_view().size()
    }
}

impl ImageLayoutTrait for MutIntensityImage {
    fn layout(&self) -> ImageLayout {
        self.dyn_view().layout()
    }
}

impl<'a> DynImageViewTrait<'a> for MutIntensityImage {
    fn dyn_view(&self) -> DynImageView<'_> {
        self.buffer.dyn_view()
    }
}

impl<'a> DynMutImageViewTrait<'a> for MutIntensityImage {
    fn dyn_mut_view(&mut self) -> DynMutImageView<'_> {
        self.buffer.dyn_mut_view()
    }
}

pub trait MutIntensityImagelTrait {
    fn to_enum(self) -> MutIntensityImageEnum;
}

impl MutIntensityImagelTrait for MutImage<1, u8> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::PU8(self)
    }
}

impl MutIntensityImagelTrait for MutImage<1, u16> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::PU16(self)
    }
}

impl MutIntensityImagelTrait for MutImage<1, f32> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::PF32(self)
    }
}

impl MutIntensityImagelTrait for MutImage<3, u8> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::P3U8(self)
    }
}

impl MutIntensityImagelTrait for MutImage<3, u16> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::P3U16(self)
    }
}

impl MutIntensityImagelTrait for MutImage<3, f32> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::P3F32(self)
    }
}

impl MutIntensityImagelTrait for MutImage<4, u8> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::P4U8(self)
    }
}

impl MutIntensityImagelTrait for MutImage<4, u16> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::P4U16(self)
    }
}

impl MutIntensityImagelTrait for MutImage<4, f32> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::P4F32(self)
    }
}

impl<'a> MutIntensityImage {
    pub fn from_mut_image<
        const NUM: usize,
        Scalar: ScalarTrait + 'static,
        I: MutImageViewTrait<'a, NUM, Scalar> + MutIntensityImagelTrait + MutImageTrait<NUM, Scalar>,
    >(
        img: I,
    ) -> Self {
        let layout = img.layout();
        let pixel_format = PixelFormat {
            number_category: Scalar::NUMBER_CATEGORY,
            num_scalars: NUM,
            num_bytes_per_scalar: std::mem::size_of::<Scalar>(),
        };
        Self {
            buffer: img.to_enum(),
            layout,
            pixel_format,
        }
    }
}

impl MutIntensityImage {
    pub fn with_size_and_tag(size: ImageSize, tag: PixelTag) -> Self {
        match tag {
            PixelTag::PU8 => MutIntensityImage::from_mut_image(MutImage::<1, u8>::with_size(size)),
            PixelTag::PU16 => {
                MutIntensityImage::from_mut_image(MutImage::<1, u16>::with_size(size))
            }
            PixelTag::PF32 => {
                MutIntensityImage::from_mut_image(MutImage::<1, f32>::with_size(size))
            }
            PixelTag::P3U8 => MutIntensityImage::from_mut_image(MutImage::<3, u8>::with_size(size)),
            PixelTag::P3U16 => {
                MutIntensityImage::from_mut_image(MutImage::<3, u16>::with_size(size))
            }
            PixelTag::P3F32 => {
                MutIntensityImage::from_mut_image(MutImage::<3, f32>::with_size(size))
            }
            PixelTag::P4U8 => MutIntensityImage::from_mut_image(MutImage::<4, u8>::with_size(size)),
            PixelTag::P4U16 => {
                MutIntensityImage::from_mut_image(MutImage::<4, u16>::with_size(size))
            }
            PixelTag::P4F32 => {
                MutIntensityImage::from_mut_image(MutImage::<4, f32>::with_size(size))
            }
        }
    }
}
