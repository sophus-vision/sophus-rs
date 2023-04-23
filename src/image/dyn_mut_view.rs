use crate::image::layout::ImageLayout;
use crate::image::layout::ImageLayoutTrait;
use crate::image::layout::ImageSize;
use crate::image::layout::ImageSizeTrait;
use crate::image::pixel::PixelFormat;

use super::dyn_view::DynImageView;
use super::dyn_view::DynImageViewTrait;

#[derive(Debug, PartialEq, Eq)]
pub struct DynMutImageView<'a> {
    pub layout: ImageLayout,
    pub mut_byte_slice: &'a mut [u8],
    pub pixel_format: PixelFormat,
}

pub trait DynMutImageViewTrait<'a>: DynImageViewTrait<'a> {
    fn dyn_mut_view(&mut self) -> DynMutImageView<'_>;

    fn mut_row_byte_slice(&mut self, v: usize) -> &mut [u8] {
        let stride = self.stride();
        let width = self.width();
        self.mut_byte_slice()
            .get_mut(v * stride..v * stride + width)
            .unwrap()
    }

    fn mut_byte_slice(&mut self) -> &mut [u8] {
        self.dyn_mut_view().mut_byte_slice
    }
}

impl<'a> ImageSizeTrait for DynMutImageView<'a> {
    fn size(&self) -> ImageSize {
        self.layout.size
    }
}

impl<'a> ImageLayoutTrait for DynMutImageView<'a> {
    fn layout(&self) -> ImageLayout {
        self.layout
    }
}

impl<'a> DynImageViewTrait<'a> for DynMutImageView<'a> {
    fn dyn_view(&self) -> DynImageView<'_> {
        DynImageView {
            layout: self.layout,
            byte_slice: self.mut_byte_slice,
            pixel_format: self.pixel_format,
        }
    }
}

impl<'a> DynMutImageViewTrait<'a> for DynMutImageView<'a> {
    fn dyn_mut_view(&mut self) -> DynMutImageView<'_> {
        DynMutImageView {
            layout: self.layout,
            mut_byte_slice: self.mut_byte_slice,
            pixel_format: self.pixel_format,
        }
    }
}
