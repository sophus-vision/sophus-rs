use crate::image::layout::ImageLayout;
use crate::image::layout::ImageLayoutTrait;
use crate::image::layout::ImageSize;
use crate::image::layout::ImageSizeTrait;
use crate::image::pixel::PixelTrait;
use crate::image::view::ImageView;
use crate::image::view::ImageViewTrait;

use super::pixel::ScalarTrait;
use super::pixel::P;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct MutImageView<'a, const NUM: usize, Scalar: ScalarTrait+'static> {
    pub layout: ImageLayout,
    pub mut_slice: &'a mut [P<NUM, Scalar>],
    pub mut_scalar_slice: &'a mut [Scalar],
}

pub trait MutImageViewTrait<'a, const NUM: usize, Scalar: ScalarTrait+'static>:
    ImageViewTrait<'a, NUM, Scalar>
{
    fn mut_view(&mut self) -> MutImageView<'_, NUM, Scalar>;

    fn mut_row_slice(&mut self, v: usize) -> &mut [P<NUM, Scalar>] {
        let stride = self.stride();
        let width = self.width();
        self.mut_slice()
            .get_mut(v * stride..v * stride + width)
            .unwrap()
    }

    fn mut_slice(&mut self) -> &mut [P<NUM, Scalar>] {
        self.mut_view().mut_slice
    }

    fn copy_data_from<V: ImageViewTrait<'a, NUM, Scalar>>(&mut self, view: &'a V) {
        if self.layout().stride == self.width() && view.stride() == view.width() {
            self.mut_slice().copy_from_slice(view.slice());
        } else {
            for v in 0..self.height() {
                self.mut_row_slice(v).copy_from_slice(view.row_slice(v));
            }
        }
    }

    fn fill(&'a mut self, value: P<NUM, Scalar>) {
        for v in 0..self.height() {
            self.mut_row_slice(v).fill(value)
        }
    }

    fn mut_pixel(&'a mut self, u: usize, v: usize) -> &'a mut P<NUM, Scalar> {
        self.mut_row_slice(v).get_mut(u).unwrap()
    }

    fn transform_from<
        'b,
        const NUM2: usize,
        Scalar2: ScalarTrait+'static,
        V: ImageViewTrait<'b, NUM2, Scalar2>,
        F: Fn(P<NUM2, Scalar2>) -> P<NUM, Scalar>,
    >(
        &'a mut self,
        view: &'b V,
        op: F,
    ) {
        for (a, b) in self.mut_slice().iter_mut().zip(view.slice().iter()) {
            *a = op(*b);
        }
    }
}

impl<'a, const NUM: usize, Scalar: ScalarTrait+'static> ImageSizeTrait for MutImageView<'a, NUM, Scalar> {
    fn size(&self) -> ImageSize {
        self.layout.size
    }
}

impl<'a, const NUM: usize, Scalar: ScalarTrait+'static> ImageLayoutTrait for MutImageView<'a, NUM, Scalar> {
    fn layout(&self) -> ImageLayout {
        self.layout
    }
}

impl<'a, const NUM: usize, Scalar: ScalarTrait+'static> ImageViewTrait<'a, NUM, Scalar> for MutImageView<'a, NUM, Scalar> {
    fn view(&'a self) -> ImageView<'a, NUM, Scalar> {
        ImageView::<NUM, Scalar> {
            layout: self.layout(),
            slice: self.mut_slice,
            scalar_slice: self.mut_scalar_slice,
        }
    }
}

impl<'a,  const NUM: usize, Scalar: ScalarTrait+'static> MutImageViewTrait<'a,  NUM, Scalar> for MutImageView<'a,  NUM, Scalar> {
    fn mut_view(&mut self) -> MutImageView<'_,  NUM, Scalar> {
        MutImageView {
            layout: self.layout,
            mut_slice: self.mut_slice,
            mut_scalar_slice: self.mut_scalar_slice,
        }
    }
}
