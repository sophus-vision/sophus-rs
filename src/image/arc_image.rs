use std::{marker::PhantomData, sync::Arc};

use super::{
    dyn_view::{DynImageView, DynImageViewTrait},
    layout::{ImageLayout, ImageLayoutTrait, ImageSize, ImageSizeTrait},
    mut_image::MutImage,
    pixel::{Pixel, PixelFormat, PixelTrait, RawDataChunk},
    view::{ImageView, ImageViewTrait},
};

#[derive(Debug, Clone)]
pub struct ArcImage<T: PixelTrait> {
    pub buffer: Arc<std::vec::Vec<RawDataChunk>>,
    pub layout: ImageLayout,
    phantom: PhantomData<T>,
}

impl<T: PixelTrait> ImageSizeTrait for ArcImage<T> {
    fn size(&self) -> ImageSize {
        self.view().size()
    }
}

impl<T: PixelTrait> ImageLayoutTrait for ArcImage<T> {
    fn layout(&self) -> ImageLayout {
        self.view().layout()
    }
}

pub trait ImageTrait<T: PixelTrait> {
    fn buffer(&self) -> Arc<std::vec::Vec<RawDataChunk>>;
}

impl<T: PixelTrait> ImageTrait<T> for ArcImage<T> {
    fn buffer(&self) -> Arc<std::vec::Vec<RawDataChunk>> {
        self.buffer.clone()
    }
}

impl<T: PixelTrait> ArcImage<T> {
    pub fn from_mut_image(image: MutImage<T>) -> Self {
        let layout = image.layout();

        Self {
            buffer: Arc::new(image.buffer),
            layout,
            phantom: PhantomData::<T> {},
        }
    }

    pub fn with_size_and_val(size: ImageSize, val: T) -> Self {
        Self::from_mut_image(MutImage::with_size_and_val(size, val))
    }
}

impl<'a, T: PixelTrait + 'a> ImageViewTrait<'a, T> for ArcImage<T> {
    fn view(&self) -> ImageView<T> {
        let slice;
        unsafe {
            slice =
                std::slice::from_raw_parts(T::cast(self.buffer.as_ptr()), self.layout.padded_area())
        }
        let scalar_slice;
        unsafe {
            scalar_slice = std::slice::from_raw_parts(
                T::scalar_cast(self.buffer.as_ptr()),
                self.layout.padded_area() * T::NUM_CHANNELS,
            )
        }
        ImageView {
            layout: self.layout,
            slice,
            scalar_slice,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    #[test]
    fn from_mut_image() {
        let size_6_x_4 = ImageSize::from_width_and_height(6, 4);
        let mut_img = MutImage::with_size_and_val(size_6_x_4, 0.5f32);

        let copy = MutImage::make_copy_from(&mut_img);
        assert_eq!(copy.size(), size_6_x_4);

        let img = ArcImage::from_mut_image(copy);
        assert_eq!(img.size(), size_6_x_4);
        assert_eq!(img.slice(), mut_img.slice());
        assert_eq!(Arc::strong_count(&img.buffer), 1);

        let mut_img2 = MutImage::from_image(img);
        assert_eq!(mut_img2.slice(), mut_img.slice());
    }

    #[test]
    fn shared_ownership() {
        let size_6_x_4 = ImageSize::from_width_and_height(6, 4);
        let mut_img = MutImage::with_size_and_val(size_6_x_4, 0.5f32);
        let img = ArcImage::from_mut_image(mut_img);

        let mut img2 = img.clone();
        assert_eq!(Arc::strong_count(&img.buffer), 2);
        assert_eq!(Arc::strong_count(&img2.buffer), 2);

        assert_eq!(img.slice(), img2.slice());

        let mut_img2 = MutImage::make_copy_from(&img2);
        assert_eq!(Arc::strong_count(&img2.buffer), 2);
        assert_ne!(mut_img2.slice().as_ptr(), img2.slice().as_ptr());

        img2 = img.clone();
        assert_eq!(Arc::strong_count(&img.buffer), 2);
        assert_eq!(Arc::strong_count(&img2.buffer), 2);
    }

    #[test]
    fn multi_threading() {
        let size_6_x_4 = ImageSize::from_width_and_height(6, 4);
        let mut_img =
            MutImage::<Pixel<3, u16>>::with_size_and_val(size_6_x_4, Pixel([10, 20, 300]));
        let img = ArcImage::from_mut_image(mut_img);

        thread::scope(|s| {
            s.spawn(|| {
                println!("{:?}", img);
            });
            s.spawn(|| {
                println!("{:?}", img);
            });
        });
    }
}

#[derive(Debug, Clone)]
pub enum IntensityImageEnum {
    PU8(ArcImage<u8>),
    PU16(ArcImage<u16>),
    PF32(ArcImage<f32>),
    P3U8(ArcImage<Pixel<3, u8>>),
    P3U16(ArcImage<Pixel<3, u16>>),
    P3F32(ArcImage<Pixel<3, f32>>),
    P4U8(ArcImage<Pixel<4, u8>>),
    P4U16(ArcImage<Pixel<4, u16>>),
    P4F32(ArcImage<Pixel<4, f32>>),
}

impl IntensityImageEnum {
    fn pixel_format(&self) -> PixelFormat {
        match self {
            IntensityImageEnum::PU8(_) => PixelFormat::new::<u8>(),
            IntensityImageEnum::PU16(_) => PixelFormat::new::<u16>(),
            IntensityImageEnum::PF32(_) => PixelFormat::new::<f32>(),
            IntensityImageEnum::P3U8(_) => PixelFormat::new::<Pixel<3, u8>>(),
            IntensityImageEnum::P3U16(_) => PixelFormat::new::<Pixel<3, u16>>(),
            IntensityImageEnum::P3F32(_) => PixelFormat::new::<Pixel<3, f32>>(),
            IntensityImageEnum::P4U8(_) => PixelFormat::new::<Pixel<3, u8>>(),
            IntensityImageEnum::P4U16(_) => PixelFormat::new::<Pixel<3, u16>>(),
            IntensityImageEnum::P4F32(_) => PixelFormat::new::<Pixel<3, f32>>(),
        }
    }

    fn u8_ptr(&self) -> *const u8 {
        match self {
            IntensityImageEnum::PU8(i) => i.buffer.as_ptr() as *const u8,
            IntensityImageEnum::PU16(i) => i.buffer.as_ptr() as *const u8,
            IntensityImageEnum::PF32(i) => i.buffer.as_ptr() as *const u8,
            IntensityImageEnum::P3U8(i) => i.buffer.as_ptr() as *const u8,
            IntensityImageEnum::P3U16(i) => i.buffer.as_ptr() as *const u8,
            IntensityImageEnum::P3F32(i) => i.buffer.as_ptr() as *const u8,
            IntensityImageEnum::P4U8(i) => i.buffer.as_ptr() as *const u8,
            IntensityImageEnum::P4U16(i) => i.buffer.as_ptr() as *const u8,
            IntensityImageEnum::P4F32(i) => i.buffer.as_ptr() as *const u8,
        }
    }
}

pub trait IntensityImagelTrait<T: PixelTrait> {
    fn to_enum(&self) -> IntensityImageEnum;
}

impl IntensityImagelTrait<u8> for ArcImage<u8> {
    fn to_enum(&self) -> IntensityImageEnum {
        IntensityImageEnum::PU8(self.clone())
    }
}

impl IntensityImagelTrait<u16> for ArcImage<u16> {
    fn to_enum(&self) -> IntensityImageEnum {
        IntensityImageEnum::PU16(self.clone())
    }
}

impl IntensityImagelTrait<f32> for ArcImage<f32> {
    fn to_enum(&self) -> IntensityImageEnum {
        IntensityImageEnum::PF32(self.clone())
    }
}

impl IntensityImagelTrait<Pixel<3, u8>> for ArcImage<Pixel<3, u8>> {
    fn to_enum(&self) -> IntensityImageEnum {
        IntensityImageEnum::P3U8(self.clone())
    }
}

impl IntensityImagelTrait<Pixel<3, u16>> for ArcImage<Pixel<3, u16>> {
    fn to_enum(&self) -> IntensityImageEnum {
        IntensityImageEnum::P3U16(self.clone())
    }
}

impl IntensityImagelTrait<Pixel<3, f32>> for ArcImage<Pixel<3, f32>> {
    fn to_enum(&self) -> IntensityImageEnum {
        IntensityImageEnum::P3F32(self.clone())
    }
}

impl IntensityImagelTrait<Pixel<4, u8>> for ArcImage<Pixel<4, u8>> {
    fn to_enum(&self) -> IntensityImageEnum {
        IntensityImageEnum::P4U8(self.clone())
    }
}

impl IntensityImagelTrait<Pixel<4, u16>> for ArcImage<Pixel<4, u16>> {
    fn to_enum(&self) -> IntensityImageEnum {
        IntensityImageEnum::P4U16(self.clone())
    }
}

impl IntensityImagelTrait<Pixel<4, f32>> for ArcImage<Pixel<4, f32>> {
    fn to_enum(&self) -> IntensityImageEnum {
        IntensityImageEnum::P4F32(self.clone())
    }
}

impl ImageSizeTrait for IntensityImageEnum {
    fn size(&self) -> ImageSize {
        self.layout().size
    }
}

impl ImageLayoutTrait for IntensityImageEnum {
    fn layout(&self) -> ImageLayout {
        match self {
            IntensityImageEnum::PU8(i) => i.layout,
            IntensityImageEnum::PU16(i) => i.layout,
            IntensityImageEnum::PF32(i) => i.layout,
            IntensityImageEnum::P3U8(i) => i.layout,
            IntensityImageEnum::P3U16(i) => i.layout,
            IntensityImageEnum::P3F32(i) => i.layout,
            IntensityImageEnum::P4U8(i) => i.layout,
            IntensityImageEnum::P4U16(i) => i.layout,
            IntensityImageEnum::P4F32(i) => i.layout,
        }
    }
}

impl DynImageViewTrait<'_> for IntensityImageEnum {
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
