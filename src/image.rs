use crate::glue::ffi::create_intensity_image_from_mut;

use super::glue;

pub use glue::ffi::FfiImageShape as ImageLayout;
pub use glue::ffi::FfiImageSize as ImageSize;
pub use glue::ffi::FfiIntensityImage as IntensityImage;
pub use glue::ffi::FfiRuntimePixelType as PixelFormat;

use strum_macros::EnumIter;

#[derive(Debug, EnumIter, Clone)]
pub enum PixelTag {
    PU8,
    P3U8,
    P4U8,
    PU16,
    P3U16,
    P4U16,
    PF32,
    P3F32,
    P4F32,
}

impl PixelFormat {
    pub fn from(p: PixelTag) -> Self {
        match p {
            PixelTag::PU8 => PixelFormat {
                is_floating_point: false,
                num_channels: 1,
                num_bytes_per_pixel_channel: 1,
            },
            PixelTag::P3U8 => PixelFormat {
                is_floating_point: false,
                num_channels: 3,
                num_bytes_per_pixel_channel: 1,
            },
            PixelTag::P4U8 => PixelFormat {
                is_floating_point: false,
                num_channels: 4,
                num_bytes_per_pixel_channel: 1,
            },
            PixelTag::PU16 => PixelFormat {
                is_floating_point: false,
                num_channels: 1,
                num_bytes_per_pixel_channel: 2,
            },
            PixelTag::P3U16 => PixelFormat {
                is_floating_point: false,
                num_channels: 3,
                num_bytes_per_pixel_channel: 2,
            },
            PixelTag::P4U16 => PixelFormat {
                is_floating_point: false,
                num_channels: 4,
                num_bytes_per_pixel_channel: 2,
            },
            PixelTag::PF32 => PixelFormat {
                is_floating_point: true,
                num_channels: 1,
                num_bytes_per_pixel_channel: 4,
            },
            PixelTag::P3F32 => PixelFormat {
                is_floating_point: true,
                num_channels: 3,
                num_bytes_per_pixel_channel: 4,
            },
            PixelTag::P4F32 => PixelFormat {
                is_floating_point: true,
                num_channels: 4,
                num_bytes_per_pixel_channel: 4,
            },
        }
    }

    fn num_bytes_per_pixel(&self) -> usize {
        self.num_bytes_per_pixel_channel * self.num_channels
    }
}

pub trait ImageSizeTrait {
    fn width(&self) -> usize {
        self.size().width
    }

    fn height(&self) -> usize {
        self.size().height
    }

    fn is_empty(&self) -> bool {
        self.width() == 0 || self.height() == 0
    }

    fn size(&self) -> ImageSize;
}

impl ImageSizeTrait for ImageSize {
    fn size(&self) -> ImageSize {
        *self
    }
}

impl ImageSize {
    pub fn from_width_and_height(width: usize, height: usize) -> Self {
        ImageSize { width, height }
    }
}

impl ImageLayout {
    pub fn from_width_and_height<T>(width: usize, height: usize) -> Self {
        ImageLayout {
            size: ImageSize { width, height },
            pitch_in_bytes: width * std::mem::size_of::<T>(),
        }
    }

    pub fn from_size_and_tag(size: ImageSize, tag: PixelTag) -> Self {
        ImageLayout {
            size,
            pitch_in_bytes: size.width * PixelFormat::from(tag).num_bytes_per_pixel(),
        }
    }
}

mod tests_image_size {
    #[test]
    fn image_size() {
        use super::*;

        let null_size = ImageSize::default();
        assert_eq!(null_size.width(), 0);
        assert_eq!(null_size.height(), 0);
        assert!(null_size.is_empty());

        let size_64_48 = ImageSize::from_width_and_height(64, 48);
        assert_eq!(size_64_48.width(), 64);
        assert_eq!(size_64_48.height(), 48);
        assert!(!size_64_48.is_empty());

        assert_eq!(null_size, null_size);
        assert_eq!(size_64_48, size_64_48);
        assert_ne!(null_size, size_64_48);
    }
}

pub trait ImageLayoutTrait: ImageSizeTrait {
    fn pitch_in_bytes(&self) -> usize {
        self.layout().pitch_in_bytes
    }

    fn layout(&self) -> ImageLayout;
}

impl ImageSizeTrait for ImageLayout {
    fn size(&self) -> ImageSize {
        self.size
    }
}

impl ImageLayoutTrait for ImageLayout {
    fn layout(&self) -> ImageLayout {
        *self
    }
}

mod tests_image_layout {

    #[test]
    fn image_layout() {
        use super::*;
        let null_layout = ImageLayout::default();
        assert_eq!(null_layout.width(), 0);
        assert_eq!(null_layout.height(), 0);
        assert_eq!(null_layout.pitch_in_bytes(), 0);
        assert!(null_layout.is_empty());

        let layout_64_48 = ImageLayout::from_width_and_height::<u16>(64, 48);
        assert_eq!(layout_64_48.width(), 64);
        assert_eq!(layout_64_48.height(), 48);
        assert_eq!(layout_64_48.pitch_in_bytes(), layout_64_48.width() * 2);
        assert!(!layout_64_48.is_empty());

        assert_eq!(null_layout, null_layout);
        assert_eq!(layout_64_48, layout_64_48);
        assert_ne!(null_layout, layout_64_48);
    }
}

use proptest::prelude::*;
pub fn pixel_enum_strategy() -> impl Strategy<Value = PixelTag> {
    prop_oneof![
        Just(PixelTag::PU8),
        Just(PixelTag::PU16),
        Just(PixelTag::PF32),
        Just(PixelTag::P3U8),
        Just(PixelTag::P4F32),
    ]
}

proptest! {
    #[test]
    fn image_size(w in 0..1000usize, h in 0..1000usize, p in pixel_enum_strategy()) {
        let layout = ImageLayout::from_size_and_tag(ImageSize::from_width_and_height(w, h), p);
        assert_eq!(layout.width(), w);
        assert_eq!(layout.height(), h);
        assert!(layout.width() <= layout.pitch_in_bytes());

    }
}

pub trait ScalarTrait: std::marker::Copy {
    fn is_floating() -> bool;
}

impl ScalarTrait for u8 {
    fn is_floating() -> bool {
        false
    }
}

impl ScalarTrait for u16 {
    fn is_floating() -> bool {
        false
    }
}

impl ScalarTrait for f32 {
    fn is_floating() -> bool {
        true
    }
}

pub trait PixelTrait: std::marker::Copy {
    type Scalar: ScalarTrait;
    const NUM_CHANNELS: usize;

    fn pixel_tag() -> PixelTag;

    fn pixel_format() -> PixelFormat {
        PixelFormat::from(Self::pixel_tag())
    }
}

impl PixelTrait for u8 {
    type Scalar = u8;
    const NUM_CHANNELS: usize = 1;

    fn pixel_tag() -> PixelTag {
        PixelTag::PU8
    }
}

impl PixelTrait for u16 {
    type Scalar = u16;
    const NUM_CHANNELS: usize = 1;

    fn pixel_tag() -> PixelTag {
        PixelTag::PU16
    }
}

impl PixelTrait for f32 {
    type Scalar = f32;
    const NUM_CHANNELS: usize = 1;

    fn pixel_tag() -> PixelTag {
        PixelTag::PF32
    }
}

impl PixelTrait for [u8; 3] {
    type Scalar = u8;
    const NUM_CHANNELS: usize = 3;

    fn pixel_tag() -> PixelTag {
        PixelTag::P3U8
    }
}

impl PixelTrait for [u16; 3] {
    type Scalar = u16;
    const NUM_CHANNELS: usize = 3;

    fn pixel_tag() -> PixelTag {
        PixelTag::P3U16
    }
}

impl PixelTrait for [f32; 3] {
    type Scalar = f32;
    const NUM_CHANNELS: usize = 3;

    fn pixel_tag() -> PixelTag {
        PixelTag::P3F32
    }
}

impl PixelTrait for [u8; 4] {
    type Scalar = u8;
    const NUM_CHANNELS: usize = 4;

    fn pixel_tag() -> PixelTag {
        PixelTag::P4U8
    }
}

impl PixelTrait for [u16; 4] {
    type Scalar = u16;
    const NUM_CHANNELS: usize = 4;

    fn pixel_tag() -> PixelTag {
        PixelTag::P4U16
    }
}

impl PixelTrait for [f32; 4] {
    type Scalar = f32;
    const NUM_CHANNELS: usize = 4;

    fn pixel_tag() -> PixelTag {
        PixelTag::P4F32
    }
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct ImageView<'a, T: PixelTrait> {
    layout: ImageLayout,
    slice: &'a [T],
}

pub trait ImageViewTrait<T: PixelTrait>: ImageLayoutTrait {
    fn view(&self) -> ImageView<T>;

    fn row_slice(&self, u: usize) -> &[T] {
        self.slice()
            .get(u * self.stride()..u * self.stride() + self.width())
            .unwrap()
    }

    fn slice(&self) -> &[T] {
        self.view().slice
    }

    fn stride(&self) -> usize {
        self.pitch_in_bytes() / (std::mem::size_of::<T>())
    }

    fn pixel(&self, u: usize, v: usize) -> T {
        *self.row_slice(v).get(u).unwrap()
    }
}

impl<'a, T: PixelTrait> ImageSizeTrait for ImageView<'a, T> {
    fn size(&self) -> ImageSize {
        self.layout.size
    }
}

impl<'a, T: PixelTrait> ImageLayoutTrait for ImageView<'a, T> {
    fn layout(&self) -> ImageLayout {
        self.layout
    }
}

impl<'a, T: PixelTrait> ImageViewTrait<T> for ImageView<'a, T> {
    fn view(&self) -> ImageView<T> {
        *self
    }
}

#[derive(Debug, Clone)]
pub struct Image<'a, T: PixelTrait> {
    img: IntensityImage,
    view: ImageView<'a, T>,
    flat_view: image::flat::FlatSamples<&'a [T::Scalar]>,
}

impl<'a, T: PixelTrait> ImageSizeTrait for Image<'a, T> {
    fn size(&self) -> ImageSize {
        self.img.image_size()
    }
}

impl<'a, T: PixelTrait> ImageLayoutTrait for Image<'a, T> {
    fn layout(&self) -> ImageLayout {
        self.img.layout
    }
}

impl<'a, T: PixelTrait> ImageViewTrait<T> for Image<'a, T> {
    fn view(&self) -> ImageView<T> {
        self.view
    }
}

impl<'a, T: PixelTrait> Image<'a, T> {
    pub fn from_mut(mut_img: MutImage<T>) -> Self {
        Self::try_from(IntensityImage::from_mut(mut_img.mut_img)).unwrap()
    }

    pub fn copy_from(view: &ImageView<T>) -> Self {
        Self::from_mut(MutImage::<T>::copy_from(view))
    }

    pub fn try_from(img: IntensityImage) -> Option<Image<'a, T>> {
        if T::Scalar::is_floating() != img.pixel_format.is_floating_point
            || std::mem::size_of::<T::Scalar>() != img.pixel_format.num_bytes_per_pixel_channel
            || T::NUM_CHANNELS != img.pixel_format.num_channels
        {
            return None;
        }
        let samples = img.try_get_flat_slice::<T::Scalar>().unwrap();
        let flat_view = image::flat::FlatSamples {
            samples,
            layout: image::flat::SampleLayout {
                channels: img.pixel_format.num_channels as u8,
                channel_stride: 1,
                width: img.layout.size.width as u32,
                width_stride: img.stride(),
                height: img.layout.size.height as u32,
                height_stride: img.layout.size.height,
            },
            color_hint: None,
        };
        Some(Image {
            img: img.clone(),
            view: ImageView {
                layout: img.layout,
                slice: img.try_get_slice::<T>().unwrap(),
            },
            flat_view,
        })
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct MutImageView<'a, T: PixelTrait> {
    layout: ImageLayout,
    mut_slice: &'a mut [T],
}

trait MutImageViewTrait<T: PixelTrait>: ImageViewTrait<T> {
    fn mut_view(&mut self) -> MutImageView<T>;

    fn mut_row_slice(&mut self, v: usize) -> &mut [T] {
        let stride = self.stride();
        let width = self.width();
        self.mut_slice()
            .get_mut(v * stride..v * stride + width)
            .unwrap()
    }

    fn mut_slice(&mut self) -> &mut [T] {
        self.mut_view().mut_slice
    }

    fn copy_data_from<V: ImageViewTrait<T>>(&mut self, view: &V) {
        let width_bytes = self.width() * std::mem::size_of::<T>();
        let dst_pitch_bytes = self.layout().pitch_in_bytes;
        let src_pitch_bytes = view.layout().pitch_in_bytes;

        if dst_pitch_bytes == width_bytes && src_pitch_bytes == width_bytes {
            self.mut_slice().copy_from_slice(view.slice());
        } else {
            for v in 0..self.height() {
                self.mut_row_slice(v).copy_from_slice(view.row_slice(v));
            }
        }
    }

    fn fill(&mut self, value: T) {
        for v in 0..self.height() {
            self.mut_row_slice(v).fill(value)
        }
    }

    fn mut_pixel(&mut self, u: usize, v: usize) -> &mut T {
        self.mut_row_slice(v).get_mut(u).unwrap()
    }
}

impl<'a, T: PixelTrait> ImageSizeTrait for MutImageView<'a, T> {
    fn size(&self) -> ImageSize {
        self.layout.size
    }
}

impl<'a, T: PixelTrait> ImageLayoutTrait for MutImageView<'a, T> {
    fn layout(&self) -> ImageLayout {
        self.layout
    }
}

impl<'a, T: PixelTrait> ImageViewTrait<T> for MutImageView<'a, T> {
    fn view(&self) -> ImageView<T> {
        ImageView::<T> {
            layout: self.layout(),
            slice: self.mut_slice,
        }
    }
}

impl<'a, T: PixelTrait> MutImageViewTrait<T> for MutImageView<'a, T> {
    fn mut_view(&mut self) -> MutImageView<T> {
        MutImageView {
            layout: self.layout,
            mut_slice: self.mut_slice,
        }
    }
}

impl<'a, T: PixelTrait> ImageSizeTrait for MutImage<'a, T> {
    fn size(&self) -> ImageSize {
        self.mut_img.image_size()
    }
}

impl<'a, T: PixelTrait> ImageLayoutTrait for MutImage<'a, T> {
    fn layout(&self) -> ImageLayout {
        self.mut_img.layout
    }
}

pub struct MutImage<'a, T: PixelTrait> {
    mut_img: MutIntensityImage,
    mut_view: MutImageView<'a, T>,
    mut_flat_view: image::flat::FlatSamples<&'a mut [T::Scalar]>,
}

impl<'a, T: PixelTrait> ImageViewTrait<T> for MutImage<'a, T> {
    fn view(&self) -> ImageView<T> {
        self.mut_view.view()
    }
}

impl<'a, T: PixelTrait> MutImageViewTrait<T> for MutImage<'a, T> {
    fn mut_view(&mut self) -> MutImageView<T> {
        MutImageView {
            layout: self.mut_img.layout,
            mut_slice: self.mut_view.mut_slice,
        }
    }
}

impl<T: PixelTrait> MutImage<'_, T> {
    pub fn from_size_uninit(size: ImageSize) -> Self {
        let dyn_img = MutIntensityImage::from_size_and_pixel_format(size, T::pixel_format());
        MutImage::try_from(dyn_img).unwrap()
    }

    pub fn from_width_height_uninit(width: usize, height: usize) -> Self {
        let dyn_img = MutIntensityImage::from_size_and_pixel_format(
            ImageSize { width, height },
            T::pixel_format(),
        );
        MutImage::try_from(dyn_img).unwrap()
    }

    pub fn copy_from<V: ImageViewTrait<T>>(view: &V) -> Self {
        let mut mut_img = MutImage::from_size_uninit(view.size());
        mut_img.copy_data_from(view);
        mut_img
    }

    pub fn try_from<'a>(mut_img: MutIntensityImage) -> Option<MutImage<'a, T>> {
        if T::Scalar::is_floating() != mut_img.pixel_format.is_floating_point
            || std::mem::size_of::<T::Scalar>() != mut_img.pixel_format.num_bytes_per_pixel_channel
            || T::NUM_CHANNELS != mut_img.pixel_format.num_channels
        {
            return None;
        }
        let samples = mut_img.try_get_flat_mut_slice::<T::Scalar>().unwrap();

        let mut_flat_view = image::flat::FlatSamples {
            samples,
            layout: image::flat::SampleLayout {
                channels: mut_img.pixel_format.num_channels as u8,
                channel_stride: 1,
                width: mut_img.layout.size.width as u32,
                width_stride: mut_img.stride(),
                height: mut_img.layout.size.height as u32,
                height_stride: mut_img.layout.size.height,
            },
            color_hint: None,
        };
        let mut_view = MutImageView {
            layout: mut_img.layout,
            mut_slice: mut_img.try_get_mut_slice().unwrap(),
        };
        Some(MutImage {
            mut_img,
            mut_view,
            mut_flat_view,
        })
    }
}

mod test_image {

    #[test]
    fn image() {
        use super::*;

        let width = 6;
        let height = 4;

        let mut img_u8 = MutImageU8::from_width_height_uninit(width, height);
        img_u8.fill(127);

        for v in 0..img_u8.height() {
            for p in img_u8.mut_row_slice(v).iter() {
                assert_eq!(*p, 127);
            }
        }
        let p = img_u8.mut_pixel(2, 3);
        *p = 7;
        assert_eq!(img_u8.pixel(2, 3), 7);

        let mut img_3f32 = MutImage3F32::from_width_height_uninit(width, height);
        img_3f32.fill([0.1, 0.5, 1.0]);

        for v in 0..img_3f32.height() {
            for p in img_3f32.row_slice(v).iter() {
                assert_eq!(*p, [0.1, 0.5, 1.0]);
            }
        }
        let p = img_3f32.mut_pixel(2, 3);
        *p = [0.5, 0.5, 0.0];
        assert_eq!(img_3f32.pixel(2, 3), [0.5, 0.5, 0.0]);
    }
}

use paste::paste;

macro_rules! MakeMutImages {
    ($type:ty) => {
        paste! {
            type  [<MutImage $type:camel>]<'a>  = MutImage::<'a, $type>;
            impl [<MutImage $type:camel>]<'_> {
                pub fn get_image_crate_view(&self) -> image::flat::View<&[$type], image::Luma<$type>> {
                    self.mut_flat_view.as_view::<image::Luma<$type>>().unwrap()
                }
            }
            impl [<MutImage $type:camel>]<'_> {
                pub fn get_mut_image_view(&mut self) -> image::flat::ViewMut<&mut [$type], image::Luma<$type>> {
                    self.mut_flat_view.as_view_mut ::<image::Luma<$type>>().unwrap()
                }
            }

            type [<MutImage 3  $type:camel>]<'a>  = MutImage::<'a, [$type; 3]>;
            impl [<MutImage 3 $type:camel>]<'_> {
                pub fn get_image_crate_view(&self) -> image::flat::View<&[$type], image::Rgb<$type>> {
                    self.mut_flat_view.as_view::<image::Rgb<$type>>().unwrap()
                }
            }
            impl [<MutImage 3 $type:camel>]<'_> {
                pub fn get_mut_image_view(&mut self) -> image::flat::ViewMut<&mut [$type], image::Luma<$type>> {
                    self.mut_flat_view.as_view_mut ::<image::Luma<$type>>().unwrap()
                }
            }

            type [<MutImage 4 $type:camel>]<'a>  = MutImage::<'a, [$type; 4]>;
            impl [<MutImage 4 $type:camel>]<'_> {
                pub fn get_image_crate_view(&self) -> image::flat::View<&[$type], image::Rgba<$type>> {
                    self.mut_flat_view.as_view::<image::Rgba<$type>>().unwrap()
                }
            }
            impl [<MutImage 4 $type:camel>]<'_> {
                pub fn get_mut_image_view(&mut self) -> image::flat::ViewMut<&mut [$type], image::Luma<$type>> {
                    self.mut_flat_view.as_view_mut ::<image::Luma<$type>>().unwrap()
                }
            }
        }
    };
}

MakeMutImages!(u8);
MakeMutImages!(u16);
MakeMutImages!(f32);

macro_rules! MakeImages {
    ($type:ty) => {
        paste! {

           type  [<Image $type:camel>]<'a>  = Image::<'a, $type>;
           impl [<Image $type:camel>]<'_> {
                pub fn get_image_crate_view(&self) -> image::flat::View<&[$type], image::Luma<$type>> {
                   self.flat_view.as_view::<image::Luma<$type>>().unwrap()
                }
           }

           type [<Image 3  $type:camel>]<'a>  = Image::<'a, [$type;3]>;
           impl [<Image 3 $type:camel>]<'_> {
                pub fn get_image_crate_view(&self) -> image::flat::View<&[$type], image::Rgb<$type>> {
                    self.flat_view.as_view::<image::Rgb<$type>>().unwrap()
                }
           }

           type [<Image 4 $type:camel>]<'a>  = Image::<'a, [$type; 4]>;
           impl [<Image 4 $type:camel>]<'_> {
                pub fn get_image_crate_view(&self) -> image::flat::View<&[$type], image::Rgba<$type>> {
                    self.flat_view.as_view::<image::Rgba<$type>>().unwrap()
                }
           }
        }
    };
}

MakeImages!(u8);
MakeImages!(u16);
MakeImages!(f32);

pub struct MutIntensityImage {
    cpp_impl: cxx::UniquePtr<glue::ffi::FfiMutIntensityImage>,
    layout: ImageLayout,
    pixel_format: PixelFormat,
}

impl MutIntensityImage {
    pub fn from_size_and_pixel_format(size: ImageSize, pixel_format: PixelFormat) -> Self {
        MutIntensityImage {
            cpp_impl: glue::ffi::create_mut_intensity_image_from_size(size, pixel_format),
            layout: ImageLayout {
                size,
                pitch_in_bytes: size.width
                    * pixel_format.num_bytes_per_pixel_channel
                    * pixel_format.num_channels,
            },
            pixel_format,
        }
    }

    pub fn mut_raw_u8_ptr(&self) -> *mut u8 {
        glue::ffi::get_mut_raw_ptr(&self.cpp_impl)
    }

    pub fn mut_raw_u8_slice<'a>(&self) -> &'a mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.mut_raw_u8_ptr(),
                self.layout.size.width * self.layout.pitch_in_bytes,
            )
        }
    }

    pub fn try_get_flat_mut_slice<'a, T: ScalarTrait>(&self) -> Option<&'a mut [T]> {
        if T::is_floating() != self.pixel_format.is_floating_point
            || std::mem::size_of::<T>() != self.pixel_format.num_bytes_per_pixel_channel
        {
            None
        } else {
            unsafe {
                Some(std::slice::from_raw_parts_mut(
                    self.mut_raw_u8_ptr() as *mut T,
                    self.layout.size.width * self.stride(),
                ))
            }
        }
    }

    pub fn try_get_mut_slice<'a, T: PixelTrait>(&self) -> Option<&'a mut [T]> {
        println!(
            "{} = {}, {} = {}, {} = {}",
            T::Scalar::is_floating(),
            self.pixel_format.is_floating_point,
            std::mem::size_of::<T::Scalar>(),
            self.pixel_format.num_bytes_per_pixel_channel,
            T::NUM_CHANNELS,
            self.pixel_format.num_channels
        );
        if T::Scalar::is_floating() != self.pixel_format.is_floating_point
            || std::mem::size_of::<T::Scalar>() != self.pixel_format.num_bytes_per_pixel_channel
            || T::NUM_CHANNELS != self.pixel_format.num_channels
        {
            None
        } else {
            unsafe {
                Some(std::slice::from_raw_parts_mut(
                    self.mut_raw_u8_ptr() as *mut T,
                    self.layout.size.width * self.stride() * self.pixel_format.num_channels,
                ))
            }
        }
    }

    pub fn try_move_out_as<'a, T: PixelTrait, const N: usize>(self) -> Option<MutImage<'a, T>> {
        return MutImage::try_from(self);
    }

    pub fn stride(&self) -> usize {
        self.layout.pitch_in_bytes
            / (self.pixel_format.num_bytes_per_pixel_channel * self.pixel_format.num_channels)
    }

    pub fn layout(&self) -> ImageLayout {
        self.layout
    }

    pub fn image_size(&self) -> ImageSize {
        self.layout.size
    }

    pub fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }
}

impl IntensityImage {
    pub fn from_size_and_pixel_format(size: ImageSize, pixel_format: PixelFormat) -> Self {
        let mut mut_img = glue::ffi::create_mut_intensity_image_from_size(size, pixel_format);
        create_intensity_image_from_mut(&mut mut_img)
    }

    pub fn from_mut(mut mut_image: MutIntensityImage) -> Self {
        create_intensity_image_from_mut(&mut mut_image.cpp_impl)
    }

    pub fn raw_u8_ptr(&self) -> *const u8 {
        glue::ffi::get_raw_ptr(self)
    }

    pub fn raw_u8_slice<'a>(&self) -> &'a [u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.raw_u8_ptr(),
                self.layout.size.width * self.layout.pitch_in_bytes,
            )
        }
    }

    pub fn try_get<'a, T: PixelTrait>(self) -> Option<Image<'a, T>> {
        return Image::try_from(self);
    }

    pub fn try_get_slice<'a, T: PixelTrait>(&self) -> Option<&'a [T]> {
        if !T::Scalar::is_floating() != self.pixel_format.is_floating_point
            || std::mem::size_of::<T>() != self.pixel_format.num_bytes_per_pixel_channel
        {
            None
        } else {
            unsafe {
                Some(std::slice::from_raw_parts(
                    self.raw_u8_ptr() as *const T,
                    self.layout.size.height * self.stride(),
                ))
            }
        }
    }

    pub fn try_get_flat_slice<'a, T: ScalarTrait>(&self) -> Option<&'a [T]> {
        if !T::is_floating() != self.pixel_format.is_floating_point
            || std::mem::size_of::<T>() != self.pixel_format.num_bytes_per_pixel_channel
        {
            None
        } else {
            unsafe {
                Some(std::slice::from_raw_parts(
                    self.raw_u8_ptr() as *const T,
                    self.layout.size.width * self.stride() * self.pixel_format.num_channels,
                ))
            }
        }
    }

    pub fn stride(&self) -> usize {
        self.layout.pitch_in_bytes
            / (self.pixel_format.num_bytes_per_pixel_channel * self.pixel_format.num_channels)
    }

    pub fn layout(&self) -> ImageLayout {
        self.layout
    }

    pub fn image_size(&self) -> ImageSize {
        self.layout.size
    }

    pub fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ImageSize, IntensityImage, MutImage3F32, MutImage3U8, MutIntensityImage, PixelFormat,
        PixelTag,
    };
    use strum::IntoEnumIterator;

    #[test]
    fn create_intensity_image() {
        let width = 640;
        let height = 480;
        for p in PixelTag::iter() {
            let mut_img = MutIntensityImage::from_size_and_pixel_format(
                ImageSize { width, height },
                PixelFormat::from(p),
            );

            assert_eq!(mut_img.image_size().width, width);
            assert_eq!(mut_img.image_size().height, height);

            let img = IntensityImage::from_mut(mut_img);
            assert_eq!(img.image_size().width, width);
            assert_eq!(img.image_size().height, height);
        }
        {
            let mut_img = MutIntensityImage::from_size_and_pixel_format(
                ImageSize { width, height },
                PixelFormat::from(PixelTag::P3F32),
            );
            let _img_3f32 = MutImage3F32::try_from(mut_img).unwrap();
        }
        {
            let mut_img = MutIntensityImage::from_size_and_pixel_format(
                ImageSize { width, height },
                PixelFormat::from(PixelTag::P4F32),
            );
            let maybe_img_3u8 = MutImage3U8::try_from(mut_img);
            assert!(maybe_img_3u8.is_none());
        }
    }
}
