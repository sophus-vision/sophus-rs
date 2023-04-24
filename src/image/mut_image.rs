use std::{marker::PhantomData, sync::Arc};
use num_traits::Zero;

use super::{
    arc_image::ArcImage,
    dyn_mut_view::{DynMutImageView, DynMutImageViewTrait},
    dyn_view::{DynImageView, DynImageViewTrait},
    layout::{ImageLayout, ImageLayoutTrait, ImageSize, ImageSizeTrait},
    mut_view::MutImageView,
    mut_view::MutImageViewTrait,
    pixel::PixelTrait,
    pixel::{RawDataChunk, ScalarTrait, P},
    view::{ImageView, ImageViewTrait},
};

#[derive(Debug, Default, Clone)]
pub struct MutImage<const NUM: usize, Scalar: ScalarTrait+'static> {
    pub buffer: std::vec::Vec<RawDataChunk>,
    pub layout: ImageLayout,
    phantom: PhantomData<Scalar>,
    // mut_flat_view: image::flat::FlatSamples<&'a mut [T::Scalar]>,
}

impl<'a, const NUM: usize, Scalar: ScalarTrait+'static> MutImage<NUM, Scalar> {
    pub fn with_size_and_val(size: ImageSize, val: P<NUM, Scalar>) -> Self {
        let layout = ImageLayout {
            size,
            stride: size.width,
        };

        let num_chunks = (layout.num_bytes_of_padded_area::<NUM, Scalar>() + 64 - 1) / 64;
        let mut buffer = Vec::with_capacity(num_chunks);
        buffer.resize(num_chunks, RawDataChunk::default());

        // let samples = mut_img.try_get_flat_mut_slice::<T::Scalar>().unwrap();

        // let mut_flat_view = image::flat::FlatSamples {
        //     samples,
        //     layout: image::flat::SampleLayout {
        //         channels: mut_img.pixel_format.num_channels as u8,
        //         channel_stride: 1,
        //         width: mut_img.layout.size.width as u32,
        //         width_stride: mut_img.stride(),
        //         height: mut_img.layout.size.height as u32,
        //         height_stride: mut_img.layout.size.height,
        //     },
        //     color_hint: None,
        // };

        let mut mut_img = MutImage {
            buffer,
            layout,
            phantom: PhantomData {},
        };
        mut_img.fill(val);
        mut_img
    }

    pub fn with_size(size: ImageSize) -> Self {
        Self::with_size_and_val(size, P::<NUM, Scalar>::zero())
    }

    pub fn make_copy_from<V: ImageViewTrait<'a, NUM, Scalar>>(view: &'a V) -> Self {
        let mut i = Self::with_size_and_val(view.size(), P::<NUM, Scalar>::zero());
        i.copy_data_from(view);
        i
    }

    pub fn make_from_transform<
        const NUM2: usize,
        Scalar2: ScalarTrait+'static,
        V: ImageViewTrait<'a, NUM2, Scalar2>,
        F: Fn(P<NUM2, Scalar2>) -> P<NUM, Scalar>,
    >(
        view: &'a V,
        op: F,
    ) -> Self {
        let mut i = Self::with_size_and_val(view.size(), P::<NUM, Scalar>::zero());
        i.transform_from(view, op);
        i
    }

    pub fn from_image(mut img: ArcImage<NUM, Scalar>) -> Self {
        let buffer: Vec<RawDataChunk> = Arc::make_mut(&mut img.buffer).clone();
        let layout = img.layout();
        Self {
            buffer,
            layout,
            phantom: PhantomData {},
        }
    }
}

impl<const NUM: usize, Scalar: ScalarTrait+'static> ImageSizeTrait for MutImage<NUM, Scalar> {
    fn size(&self) -> ImageSize {
        self.view().size()
    }
}

impl<const NUM: usize, Scalar: ScalarTrait+'static> ImageLayoutTrait for MutImage<NUM, Scalar> {
    fn layout(&self) -> ImageLayout {
        self.view().layout()
    }
}

impl<'a, const NUM: usize, Scalar: ScalarTrait + 'static> ImageViewTrait<'a, NUM, Scalar>
    for MutImage<NUM, Scalar>
{
    fn view(&self) -> ImageView<'_, NUM, Scalar> {
        let slice;
        unsafe {
            slice = std::slice::from_raw_parts(
                P::<NUM, Scalar>::cast_from_raw(self.buffer.as_ptr()),
                self.layout.padded_area(),
            );
        }
        let scalar_slice;
        unsafe {
            scalar_slice = std::slice::from_raw_parts(
                P::<NUM, Scalar>::scalar_cast(self.buffer.as_ptr()),
                self.layout.padded_area() * NUM,
            );
        }
        ImageView::<'_, NUM, Scalar> {
            layout: self.layout,
            slice,
            scalar_slice,
        }
    }

    fn scalar_slice(&self) -> &[Scalar] {
        self.view().scalar_slice
    }
}

impl<'a, const NUM: usize, Scalar: ScalarTrait + 'static> MutImageViewTrait<'a, NUM, Scalar>
    for MutImage<NUM, Scalar>
{
    fn mut_view(&mut self) -> MutImageView<'_, NUM, Scalar> {
        let mut_slice;
        unsafe {
            mut_slice = std::slice::from_raw_parts_mut(
                P::<NUM, Scalar>::mut_cast(self.buffer.as_mut_ptr()),
                self.layout.padded_area(),
            );
        }
        let mut_scalar_slice;
        unsafe {
            mut_scalar_slice = std::slice::from_raw_parts_mut(
                P::<NUM, Scalar>::mut_scalar_cast(self.buffer.as_mut_ptr()),
                self.layout.padded_area() * NUM,
            );
        }
        MutImageView::<'_, NUM, Scalar> {
            layout: self.layout,
            mut_slice,
            mut_scalar_slice,
        }
    }
}

pub trait MutImageTrait<const NUM: usize, Scalar: ScalarTrait+'static> {
    fn buffer(self) -> std::vec::Vec<RawDataChunk>;
}

impl<const NUM: usize, Scalar: ScalarTrait+'static> MutImageTrait<NUM, Scalar> for MutImage<NUM, Scalar> {
    fn buffer(self) -> std::vec::Vec<RawDataChunk> {
        self.buffer
    }
}

#[cfg(test)]
mod tests {


    use crate::image::pixel::P1F32;

    use super::*;

    #[test]
    fn empty_image() {
        let img = MutImage::<1, u8>::default();
        assert!(img.is_empty());

        let size_2_x_3 = ImageSize::from_width_and_height(2, 3);
        let img_f32 = MutImage::<1, f32>::with_size_and_val(size_2_x_3, P1F32::zero());
        assert!(!img_f32.is_empty());
        assert_eq!(img_f32.size(), size_2_x_3);
    }

    #[test]
    fn create_copy_access() {
        // 1. create new mut image.
        let size_2_x_3 = ImageSize::from_width_and_height(2, 3);

        let mut img_f32 = MutImage::<1, f32>::with_size_and_val(size_2_x_3, P1F32::new(0.25));

        // create a copy of it.
        let img_f32_copy = MutImage::make_copy_from(&img_f32);

        // test that copy contains the data expected.
        assert_eq!(img_f32.slice(), img_f32_copy.slice());
        img_f32.fill(P::<1,f32>::new(0.23));
        assert_ne!(img_f32.slice(), img_f32_copy.slice());
    }

    #[test]
    pub fn transform() {
        let size_2_x_3 = ImageSize::from_width_and_height(2, 3);

        let img_f32 = MutImage::<1, f32>::with_size_and_val(size_2_x_3,  P1F32::new(1.0));

        // let op = |v: f32| {
        //     let mut pixel = P::default();
        //     pixel.0[0] = v;
        //     pixel.0[1] = 0.2 * v;
        //     pixel
        // };

        // let pattern = MutImage::make_from_transform(&img_f32, op);
        // assert_eq!(
        //     pattern.slice(),
        //     MutImage::<3, f32>::with_size_and_val(size_2_x_3, op(1.0)).slice()
        // );
    }
}

#[derive(Debug, Clone)]
pub enum MutIntensityImageEnum {
    PU8(MutImage<1, u8>),
    PU16(MutImage<1, u16>),
    PF32(MutImage<1, f32>),
    P3U8(MutImage<3, u8>),
    P3U16(MutImage<3, u16>),
    P3F32(MutImage<3, f32>),
    P4U8(MutImage<4, u8>),
    P4U16(MutImage<4,u16>),
    P4F32(MutImage<4, f32>),
}

impl MutIntensityImageEnum {
    fn u8_ptr(&self) -> *const u8 {
        match self {
            MutIntensityImageEnum::PU8(i) => i.buffer.as_ptr() as *const u8,
            MutIntensityImageEnum::PU16(i) => i.buffer.as_ptr() as *const u8,
            MutIntensityImageEnum::PF32(i) => i.buffer.as_ptr() as *const u8,
            MutIntensityImageEnum::P3U8(i) => i.buffer.as_ptr() as *const u8,
            MutIntensityImageEnum::P3U16(i) => i.buffer.as_ptr() as *const u8,
            MutIntensityImageEnum::P3F32(i) => i.buffer.as_ptr() as *const u8,
            MutIntensityImageEnum::P4U8(i) => i.buffer.as_ptr() as *const u8,
            MutIntensityImageEnum::P4U16(i) => i.buffer.as_ptr() as *const u8,
            MutIntensityImageEnum::P4F32(i) => i.buffer.as_ptr() as *const u8,
        }
    }

    fn mut_u8_ptr(&mut self) -> *mut u8 {
        match self {
            MutIntensityImageEnum::PU8(i) => i.buffer.as_mut_ptr() as *mut u8,
            MutIntensityImageEnum::PU16(i) => i.buffer.as_mut_ptr() as *mut u8,
            MutIntensityImageEnum::PF32(i) => i.buffer.as_mut_ptr() as *mut u8,
            MutIntensityImageEnum::P3U8(i) => i.buffer.as_mut_ptr() as *mut u8,
            MutIntensityImageEnum::P3U16(i) => i.buffer.as_mut_ptr() as *mut u8,
            MutIntensityImageEnum::P3F32(i) => i.buffer.as_mut_ptr() as *mut u8,
            MutIntensityImageEnum::P4U8(i) => i.buffer.as_mut_ptr() as *mut u8,
            MutIntensityImageEnum::P4U16(i) => i.buffer.as_mut_ptr() as *mut u8,
            MutIntensityImageEnum::P4F32(i) => i.buffer.as_mut_ptr() as *mut u8,
        }
    }
}

impl ImageSizeTrait for MutIntensityImageEnum {
    fn size(&self) -> ImageSize {
        self.layout().size
    }
}

impl ImageLayoutTrait for MutIntensityImageEnum {
    fn layout(&self) -> ImageLayout {
        match self {
            MutIntensityImageEnum::PU8(i) => i.layout,
            MutIntensityImageEnum::PU16(i) => i.layout,
            MutIntensityImageEnum::PF32(i) => i.layout,
            MutIntensityImageEnum::P3U8(i) => i.layout,
            MutIntensityImageEnum::P3U16(i) => i.layout,
            MutIntensityImageEnum::P3F32(i) => i.layout,
            MutIntensityImageEnum::P4U8(i) => i.layout,
            MutIntensityImageEnum::P4U16(i) => i.layout,
            MutIntensityImageEnum::P4F32(i) => i.layout,
        }
    }
}

impl<'a> DynImageViewTrait<'a> for MutIntensityImageEnum {
    fn dyn_view(&self) -> DynImageView<'_> {
        let byte_slice;
        let layout = self.layout();
        let pixel_format = self.pixel_format();
        unsafe {
            byte_slice = std::slice::from_raw_parts(
                self.u8_ptr(),
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

impl<'a> DynMutImageViewTrait<'a> for MutIntensityImageEnum {
    fn dyn_mut_view(&mut self) -> DynMutImageView<'_> {
        let mut_byte_slice;
        let layout = self.layout();
        let pixel_format = self.pixel_format();
        unsafe {
            mut_byte_slice = std::slice::from_raw_parts_mut(
                self.mut_u8_ptr(),
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
