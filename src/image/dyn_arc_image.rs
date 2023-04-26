#![allow(clippy::clone_double_ref)]

use std::sync::Arc;

use super::arc_image::ArcImage;
use super::arc_image::ImageTrait;
use super::arc_image::IntensityImageEnum;

use super::arc_image::IntensityImagelTrait;
use super::data::DataChunkVec;
use super::dyn_mut_image::MutIntensityImage;
use super::dyn_view::DynImageView;
use super::dyn_view::DynImageViewTrait;
use super::mut_image::MutImage;
use super::pixel::PixelFormat;
use super::pixel::ScalarTrait;
use super::view::ImageViewTrait;
use crate::image::layout::ImageLayout;
use crate::image::layout::ImageLayoutTrait;
use crate::image::layout::ImageSize;
use crate::image::layout::ImageSizeTrait;
use crate::image::pixel::P1F32;

pub struct AnyImage {
    pub buffer: Arc<DataChunkVec>,
    pub layout: ImageLayout,
    pub pixel_format: PixelFormat,
}

impl ImageSizeTrait for AnyImage {
    fn size(&self) -> ImageSize {
        self.dyn_view().size()
    }
}

impl ImageLayoutTrait for AnyImage {
    fn layout(&self) -> ImageLayout {
        self.dyn_view().layout()
    }
}

impl<'a> DynImageViewTrait<'a> for AnyImage {
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

impl AnyImage {
    pub fn from_image<const NUM: usize, Scalar: ScalarTrait + 'static>(
        img: &ArcImage<NUM, Scalar>,
    ) -> AnyImage {
        let buffer = img.buffer.clone();

        let layout = img.layout();
        let pixel_format = PixelFormat {
            number_category: Scalar::NUMBER_CATEGORY,
            num_scalars: NUM,
            num_bytes_per_scalar: std::mem::size_of::<Scalar>(),
        };
        Self {
            buffer,
            layout,
            pixel_format,
        }
    }
}

#[test]
fn from_mut_image() {
    let size_2_x_3 = ImageSize::from_width_and_height(2, 3);
    let img_f32 = ArcImage::<1, f32>::with_size_and_val(size_2_x_3, P1F32::new(0.25));
    let dyn_img = AnyImage::from_image(&img_f32);

    assert_eq!(Arc::strong_count(&img_f32.buffer), 2);
    assert_eq!(Arc::strong_count(&dyn_img.buffer), 2);
}

#[derive(Debug, Clone)]
pub struct IntensityImage {
    pub buffer: IntensityImageEnum,
    pub layout: ImageLayout,
    pub pixel_format: PixelFormat,
}

impl ImageSizeTrait for IntensityImage {
    fn size(&self) -> ImageSize {
        self.dyn_view().size()
    }
}

impl ImageLayoutTrait for IntensityImage {
    fn layout(&self) -> ImageLayout {
        self.dyn_view().layout()
    }
}

impl<'a> DynImageViewTrait<'a> for IntensityImage {
    fn dyn_view(&self) -> DynImageView<'_> {
        self.buffer.dyn_view()
    }
}

impl IntensityImage {
    pub fn from_dyn_mut(img: MutIntensityImage) -> Self {
        match img.buffer {
            super::mut_image::MutIntensityImageEnum::PU8(i) => IntensityImage::from_mut_image(i),
            super::mut_image::MutIntensityImageEnum::PU16(i) => IntensityImage::from_mut_image(i),
            super::mut_image::MutIntensityImageEnum::PF32(i) => IntensityImage::from_mut_image(i),
            super::mut_image::MutIntensityImageEnum::P3U8(i) => IntensityImage::from_mut_image(i),
            super::mut_image::MutIntensityImageEnum::P3U16(i) => IntensityImage::from_mut_image(i),
            super::mut_image::MutIntensityImageEnum::P3F32(i) => IntensityImage::from_mut_image(i),
            super::mut_image::MutIntensityImageEnum::P4U8(i) => IntensityImage::from_mut_image(i),
            super::mut_image::MutIntensityImageEnum::P4U16(i) => IntensityImage::from_mut_image(i),
            super::mut_image::MutIntensityImageEnum::P4F32(i) => IntensityImage::from_mut_image(i),
        }
    }
}

impl<'a> IntensityImage {
    pub fn from_image<
        const NUM: usize,
        Scalar: ScalarTrait + 'static,
        I: ImageViewTrait<'a, NUM, Scalar>
            + IntensityImagelTrait<NUM, Scalar>
            + ImageTrait<NUM, Scalar>,
    >(
        img: &I,
    ) -> IntensityImage {
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

impl IntensityImage {
    pub fn from_mut_image<const NUM: usize, Scalar: ScalarTrait + 'static>(
        img: MutImage<NUM, Scalar>,
    ) -> IntensityImage
    where
        ArcImage<NUM, Scalar>: IntensityImagelTrait<NUM, Scalar>,
    {
        IntensityImage::from_image(&ArcImage::from_mut_image(img))
    }
}
