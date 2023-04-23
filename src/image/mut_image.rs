use std::{marker::PhantomData, sync::Arc};

use super::{
    arc_image::ArcImage,
    dyn_mut_view::{DynMutImageView, DynMutImageViewTrait},
    dyn_view::{DynImageView, DynImageViewTrait},
    layout::{ImageLayout, ImageLayoutTrait, ImageSize, ImageSizeTrait},
    mut_view::MutImageView,
    mut_view::MutImageViewTrait,
    pixel::PixelTrait,
    pixel::{Pixel, RawDataChunk},
    view::{ImageView, ImageViewTrait},
};

#[derive(Debug, Default, Clone)]
pub struct MutImage<T: PixelTrait> {
    pub buffer: std::vec::Vec<RawDataChunk>,
    pub layout: ImageLayout,
    phantom: PhantomData<T>,
    // mut_flat_view: image::flat::FlatSamples<&'a mut [T::Scalar]>,
}

impl<'a, T: PixelTrait + 'a> MutImage<T> {
    pub fn with_size_and_val(size: ImageSize, val: T) -> Self {
        let layout = ImageLayout {
            size,
            stride: size.width,
        };

        let num_chunks = (layout.num_bytes_of_padded_area::<T>() + 64 - 1) / 64;
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
        Self::with_size_and_val(size, T::zero())
    }

    pub fn make_copy_from<V: ImageViewTrait<'a, T>>(view: &'a V) -> Self {
        let mut i = Self::with_size_and_val(view.size(), T::zero());
        i.copy_data_from(view);
        i
    }

    pub fn make_from_transform<T2: PixelTrait + 'a, V: ImageViewTrait<'a, T2>, F: Fn(T2) -> T>(
        view: &'a V,
        op: F,
    ) -> Self {
        let mut i = Self::with_size_and_val(view.size(), T::zero());
        i.transform_from(view, op);
        i
    }

    pub fn from_image(mut img: ArcImage<T>) -> Self {
        let buffer: Vec<RawDataChunk> = Arc::make_mut(&mut img.buffer).clone();
        let layout = img.layout();
        Self {
            buffer,
            layout,
            phantom: PhantomData {},
        }
    }
}

impl<T: PixelTrait> ImageSizeTrait for MutImage<T> {
    fn size(&self) -> ImageSize {
        self.view().size()
    }
}

impl<T: PixelTrait> ImageLayoutTrait for MutImage<T> {
    fn layout(&self) -> ImageLayout {
        self.view().layout()
    }
}

impl<'a, T: PixelTrait + 'a> ImageViewTrait<'a, T> for MutImage<T> {
    fn view(&self) -> ImageView<'_, T> {
        let slice;
        unsafe {
            slice = std::slice::from_raw_parts(
                T::cast(self.buffer.as_ptr()),
                self.layout.padded_area(),
            );
        }
        let scalar_slice;
        unsafe {
            scalar_slice = std::slice::from_raw_parts(
                T::scalar_cast(self.buffer.as_ptr()),
                self.layout.padded_area() * T::NUM_CHANNELS,
            );
        }
        ImageView::<'_, T> {
            layout: self.layout,
            slice,
            scalar_slice,
        }
    }

    fn scalar_slice(&self) -> &[T::Scalar] {
        self.view().scalar_slice
    }
}

impl<'a, T: PixelTrait + 'a> MutImageViewTrait<'a, T> for MutImage<T> {
    fn mut_view(&mut self) -> MutImageView<'_, T> {
        let mut_slice;
        unsafe {
            mut_slice = std::slice::from_raw_parts_mut(
                T::mut_cast(self.buffer.as_mut_ptr()),
                self.layout.padded_area(),
            );
        }
        let mut_scalar_slice;
        unsafe {
            mut_scalar_slice = std::slice::from_raw_parts_mut(
                T::mut_scalar_cast(self.buffer.as_mut_ptr()),
                self.layout.padded_area() * T::NUM_CHANNELS,
            );
        }
        MutImageView::<'_, T> {
            layout: self.layout,
            mut_slice,
            mut_scalar_slice,
        }
    }
}

pub trait MutImageTrait<T: PixelTrait> {
    fn buffer(self) -> std::vec::Vec<RawDataChunk>;
}

impl<T: PixelTrait> MutImageTrait<T> for MutImage<T> {
    fn buffer(self) -> std::vec::Vec<RawDataChunk> {
        self.buffer
    }
}

#[cfg(test)]
mod tests {

    use crate::image::pixel::Pixel;

    use super::*;

    #[test]
    fn empty_image() {
        let img = MutImage::<u8>::default();
        assert!(img.is_empty());

        let size_2_x_3 = ImageSize::from_width_and_height(2, 3);
        let img_f32 = MutImage::<f32>::with_size_and_val(size_2_x_3, 0.0);
        assert!(!img_f32.is_empty());
        assert_eq!(img_f32.size(), size_2_x_3);
    }

    #[test]
    fn create_copy_access() {
        // 1. create new mut image.
        let size_2_x_3 = ImageSize::from_width_and_height(2, 3);

        let mut img_f32 = MutImage::<f32>::with_size_and_val(size_2_x_3, 0.25);

        // create a copy of it.
        let img_f32_copy = MutImage::make_copy_from(&img_f32);

        // test that copy contains the data expected.
        assert_eq!(img_f32.slice(), img_f32_copy.slice());
        img_f32.fill(0.23);
        assert_ne!(img_f32.slice(), img_f32_copy.slice());
    }

    #[test]
    pub fn transform() {
        let size_2_x_3 = ImageSize::from_width_and_height(2, 3);

        let img_f32 = MutImage::<f32>::with_size_and_val(size_2_x_3, 1.0);

        let op = |v: f32| {
            let mut pixel = Pixel::default();
            pixel.0[0] = v;
            pixel.0[1] = 0.2 * v;
            pixel
        };

        let pattern = MutImage::make_from_transform(&img_f32, op);
        assert_eq!(
            pattern.slice(),
            MutImage::<Pixel<3, f32>>::with_size_and_val(size_2_x_3, op(1.0)).slice()
        );
    }
}

#[derive(Debug, Clone)]
pub enum MutIntensityImageEnum {
    PU8(MutImage<u8>),
    PU16(MutImage<u16>),
    PF32(MutImage<f32>),
    P3U8(MutImage<Pixel<3, u8>>),
    P3U16(MutImage<Pixel<3, u16>>),
    P3F32(MutImage<Pixel<3, f32>>),
    P4U8(MutImage<Pixel<4, u8>>),
    P4U16(MutImage<Pixel<4, u16>>),
    P4F32(MutImage<Pixel<4, f32>>),
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
