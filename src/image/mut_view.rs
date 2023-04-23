use crate::image::layout::ImageLayout;
use crate::image::layout::ImageLayoutTrait;
use crate::image::layout::ImageSize;
use crate::image::layout::ImageSizeTrait;
use crate::image::pixel::PixelTrait;
use crate::image::view::ImageView;
use crate::image::view::ImageViewTrait;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct MutImageView<'a, T: PixelTrait> {
    pub layout: ImageLayout,
    pub mut_slice: &'a mut [T],
    pub mut_scalar_slice: &'a mut [T::Scalar],
}

pub trait MutImageViewTrait<'a, T: PixelTrait + 'a>: ImageViewTrait<'a, T> {
    fn mut_view(&mut self) -> MutImageView<'_, T>;

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

    fn copy_data_from<V: ImageViewTrait<'a, T>>(&mut self, view: &'a V) {
        if self.layout().stride == self.width() && view.stride() == view.width() {
            self.mut_slice().copy_from_slice(view.slice());
        } else {
            for v in 0..self.height() {
                self.mut_row_slice(v).copy_from_slice(view.row_slice(v));
            }
        }
    }

    fn fill(&'a mut self, value: T) {
        for v in 0..self.height() {
            self.mut_row_slice(v).fill(value)
        }
    }

    fn mut_pixel(&'a mut self, u: usize, v: usize) -> &'a mut T {
        self.mut_row_slice(v).get_mut(u).unwrap()
    }

    fn transform_from<'b, T2: PixelTrait + 'b, V: ImageViewTrait<'b, T2>, F: Fn(T2) -> T>(
        &'a mut self,
        view: &'b V,
        op: F,
    ) {
        for (a, b) in self.mut_slice().iter_mut().zip(view.slice().iter()) {
            *a = op(*b);
        }
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

impl<'a, T: PixelTrait + 'a> ImageViewTrait<'a, T> for MutImageView<'a, T> {
    fn view(&'a self) -> ImageView<'a, T> {
        ImageView::<T> {
            layout: self.layout(),
            slice: self.mut_slice,
            scalar_slice: self.mut_scalar_slice,
        }
    }
}

impl<'a, T: PixelTrait> MutImageViewTrait<'a, T> for MutImageView<'a, T> {
    fn mut_view(&mut self) -> MutImageView<'_, T> {
        MutImageView {
            layout: self.layout,
            mut_slice: self.mut_slice,
            mut_scalar_slice: self.mut_scalar_slice,
        }
    }
}
