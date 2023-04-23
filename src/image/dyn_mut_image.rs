use super::dyn_mut_view::DynMutImageView;
use super::dyn_mut_view::DynMutImageViewTrait;
use super::dyn_view::DynImageView;
use super::dyn_view::DynImageViewTrait;

use super::mut_image::MutImage;

use super::mut_image::MutImageTrait;
use super::mut_image::MutIntensityImageEnum;
use super::mut_view::MutImageViewTrait;
use super::pixel::IntensityPixelTrait;
use super::pixel::Pixel;
use super::pixel::PixelFormat;
use super::pixel::PixelTag;
use super::pixel::PixelTrait;
use super::pixel::RawDataChunk;
use crate::image::layout::ImageLayout;
use crate::image::layout::ImageLayoutTrait;
use crate::image::layout::ImageSize;
use crate::image::layout::ImageSizeTrait;

#[cfg(not(target_arch = "wasm32"))]
use pyo3::pyclass;

pub struct MutAnyImage {
    pub buffer: std::vec::Vec<RawDataChunk>,
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
        let byte_slice;
        let layout = self.layout;
        let pixel_format = self.pixel_format;
        unsafe {
            byte_slice = std::slice::from_raw_parts(
                self.buffer[0].u8_ptr(),
                layout.padded_area() * pixel_format.num_bytes(),
            );
        }
        DynImageView {
            layout,
            byte_slice,
            pixel_format,
        }
    }
}

impl<'a> DynMutImageViewTrait<'a> for MutAnyImage {
    fn dyn_mut_view(&mut self) -> DynMutImageView<'_> {
        let mut_byte_slice;
        let layout = self.layout;
        let pixel_format = self.pixel_format;
        unsafe {
            mut_byte_slice = std::slice::from_raw_parts_mut(
                self.buffer[0].mut_u8_ptr(),
                layout.padded_area() * pixel_format.num_bytes(),
            );
        }
        DynMutImageView {
            layout,
            mut_byte_slice,
            pixel_format,
        }
    }
}

impl MutAnyImage {
    pub fn from_mut_image<T: PixelTrait>(img: MutImage<T>) -> Self {
        let layout = img.layout();
        let pixel_format = PixelFormat {
            number_category: T::NUMBER_CATEGORY,
            num_scalars: T::NUM_CHANNELS,
            num_bytes_per_scalar: std::mem::size_of::<T::Scalar>(),
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
    let img_f32 = MutImage::<f32>::with_size_and_val(size_2_x_3, 0.25);
    let _dyn_img = MutAnyImage::from_mut_image(img_f32);
}

#[cfg_attr(not(target_arch = "wasm32"), pyclass)]
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

impl MutIntensityImagelTrait for MutImage<u8> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::PU8(self)
    }
}

impl MutIntensityImagelTrait for MutImage<u16> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::PU16(self)
    }
}

impl MutIntensityImagelTrait for MutImage<f32> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::PF32(self)
    }
}

impl MutIntensityImagelTrait for MutImage<Pixel<3, u8>> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::P3U8(self)
    }
}

impl MutIntensityImagelTrait for MutImage<Pixel<3, u16>> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::P3U16(self)
    }
}

impl MutIntensityImagelTrait for MutImage<Pixel<3, f32>> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::P3F32(self)
    }
}

impl MutIntensityImagelTrait for MutImage<Pixel<4, u8>> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::P4U8(self)
    }
}

impl MutIntensityImagelTrait for MutImage<Pixel<4, u16>> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::P4U16(self)
    }
}

impl MutIntensityImagelTrait for MutImage<Pixel<4, f32>> {
    fn to_enum(self) -> MutIntensityImageEnum {
        MutIntensityImageEnum::P4F32(self)
    }
}

impl<'a> MutIntensityImage {
    pub fn from_mut_image<
        T: IntensityPixelTrait + 'a,
        I: MutImageViewTrait<'a, T> + MutIntensityImagelTrait + MutImageTrait<T>,
    >(
        img: I,
    ) -> Self {
        let layout = img.layout();
        let pixel_format = PixelFormat {
            number_category: T::NUMBER_CATEGORY,
            num_scalars: T::NUM_CHANNELS,
            num_bytes_per_scalar: std::mem::size_of::<T::Scalar>(),
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
            PixelTag::PU8 => MutIntensityImage::from_mut_image(MutImage::<u8>::with_size(size)),
            PixelTag::PU16 => MutIntensityImage::from_mut_image(MutImage::<u16>::with_size(size)),
            PixelTag::PF32 => MutIntensityImage::from_mut_image(MutImage::<f32>::with_size(size)),
            PixelTag::P3U8 => {
                MutIntensityImage::from_mut_image(MutImage::<Pixel<3, u8>>::with_size(size))
            }
            PixelTag::P3U16 => {
                MutIntensityImage::from_mut_image(MutImage::<Pixel<3, u16>>::with_size(size))
            }
            PixelTag::P3F32 => {
                MutIntensityImage::from_mut_image(MutImage::<Pixel<3, f32>>::with_size(size))
            }
            PixelTag::P4U8 => {
                MutIntensityImage::from_mut_image(MutImage::<Pixel<4, u8>>::with_size(size))
            }
            PixelTag::P4U16 => {
                MutIntensityImage::from_mut_image(MutImage::<Pixel<4, u16>>::with_size(size))
            }
            PixelTag::P4F32 => {
                MutIntensityImage::from_mut_image(MutImage::<Pixel<4, f32>>::with_size(size))
            }
        }
    }
}
