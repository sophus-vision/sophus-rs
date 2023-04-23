use crate::image::layout::ImageLayout;
use crate::image::layout::ImageLayoutTrait;
use crate::image::layout::ImageSize;
use crate::image::layout::ImageSizeTrait;
use crate::image::pixel::PixelFormat;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct DynImageView<'a> {
    pub layout: ImageLayout,
    pub byte_slice: &'a [u8],
    pub pixel_format: PixelFormat,
}

pub trait DynImageViewTrait<'a>: ImageLayoutTrait {
    fn dyn_view(&self) -> DynImageView<'_>;

    fn row_byte_slice(&self, u: usize) -> &[u8] {
        self.byte_slice()
            .get(u * self.stride()..u * self.stride() + self.width())
            .unwrap()
    }

    fn byte_slice(&self) -> &[u8] {
        self.dyn_view().byte_slice
    }

    fn byte(&self, u: usize, v: usize) -> u8 {
        *self.row_byte_slice(v).get(u).unwrap()
    }

    fn pixel_format(&self) -> PixelFormat {
        self.dyn_view().pixel_format
    }
}

impl<'a> ImageSizeTrait for DynImageView<'a> {
    fn size(&self) -> ImageSize {
        self.layout.size
    }
}

impl<'a> ImageLayoutTrait for DynImageView<'a> {
    fn layout(&self) -> ImageLayout {
        self.layout
    }
}

impl<'a> DynImageViewTrait<'a> for DynImageView<'a> {
    fn dyn_view(&self) -> DynImageView<'a> {
        *self
    }
}
