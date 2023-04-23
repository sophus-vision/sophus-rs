use crate::image::layout::ImageLayout;
use crate::image::layout::ImageLayoutTrait;
use crate::image::layout::ImageSize;
use crate::image::layout::ImageSizeTrait;
use crate::image::pixel::PixelTrait;

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct ImageView<'a, T: PixelTrait> {
    pub layout: ImageLayout,
    pub slice: &'a [T],
    pub scalar_slice: &'a [T::Scalar],
}

pub trait ImageViewTrait<'a, T: PixelTrait + 'a>: ImageLayoutTrait {
    fn view(&'a self) -> ImageView<'a, T>;

    fn row_slice(&'a self, u: usize) -> &'a [T] {
        self.slice()
            .get(u * self.stride()..u * self.stride() + self.width())
            .unwrap()
    }

    fn slice(&'a self) -> &'a [T] {
        self.view().slice
    }

    fn scalar_slice(&'a self) -> &'a [T::Scalar] {
        self.view().scalar_slice
    }

    fn pixel(&'a self, u: usize, v: usize) -> T {
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

impl<'a, T: PixelTrait> ImageViewTrait<'a, T> for ImageView<'a, T> {
    fn view(&self) -> ImageView<T> {
        *self
    }
}
