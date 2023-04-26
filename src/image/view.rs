use crate::image::layout::ImageLayout;
use crate::image::layout::ImageLayoutTrait;
use crate::image::layout::ImageSize;
use crate::image::layout::ImageSizeTrait;

use super::pixel::ScalarTrait;
use super::pixel::P;

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct ImageView<'a, const NUM: usize, Scalar: ScalarTrait + 'static> {
    pub layout: ImageLayout,
    pub slice: &'a [P<NUM, Scalar>],
    //pub scalar_slice: &'a [Scalar],
}

pub trait ImageViewTrait<'a, const NUM: usize, Scalar: ScalarTrait + 'static>:
    ImageLayoutTrait
{
    fn view(&'a self) -> ImageView<'a, NUM, Scalar>;

    fn row_slice(&'a self, u: usize) -> &'a [P<NUM, Scalar>] {
        self.slice()
            .get(u * self.stride()..u * self.stride() + self.width())
            .unwrap()
    }

    fn slice(&'a self) -> &'a [P<NUM, Scalar>] {
        self.view().slice
    }

    // fn scalar_slice(&'a self) -> &'a [Scalar] {
    //     self.view().scalar_slice
    // }

    fn pixel(&'a self, u: usize, v: usize) -> P<NUM, Scalar> {
        *self.row_slice(v).get(u).unwrap()
    }
}

impl<'a, const NUM: usize, Scalar: ScalarTrait + 'static> ImageSizeTrait
    for ImageView<'a, NUM, Scalar>
{
    fn size(&self) -> ImageSize {
        self.layout.size
    }
}

impl<'a, const NUM: usize, Scalar: ScalarTrait + 'static> ImageLayoutTrait
    for ImageView<'a, NUM, Scalar>
{
    fn layout(&self) -> ImageLayout {
        self.layout
    }
}

impl<'a, const NUM: usize, Scalar: ScalarTrait + 'static> ImageViewTrait<'a, NUM, Scalar>
    for ImageView<'a, NUM, Scalar>
{
    fn view(&self) -> ImageView<NUM, Scalar> {
        *self
    }
}
