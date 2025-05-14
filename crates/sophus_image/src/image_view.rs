use ndarray::ShapeBuilder;
use sophus_tensor::TensorView;

use crate::{
    ImageSize,
    prelude::*,
};

/// Image view of static tensors
#[derive(Debug, Clone, PartialEq)]
pub struct GenImageView<
    'a,
    const TOTAL_RANK: usize,
    const SRANK: usize,
    Scalar: IsCoreScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS> + 'static,
    const ROWS: usize,
    const COLS: usize,
> where
    ndarray::Dim<[ndarray::Ix; TOTAL_RANK]>: ndarray::Dimension,
{
    /// underlying tensor view
    pub tensor_view: TensorView<'a, TOTAL_RANK, 2, SRANK, Scalar, STensor, ROWS, COLS>,
}

/// Image view of static tensors
pub trait IsImageView<
    'a,
    const TOTAL_RANK: usize,
    const SRANK: usize,
    Scalar: IsCoreScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS> + 'static,
    const ROWS: usize,
    const COLS: usize,
> where
    ndarray::Dim<[ndarray::Ix; TOTAL_RANK]>: ndarray::Dimension,
{
    /// Get the image view
    fn image_view(&'a self) -> GenImageView<'a, TOTAL_RANK, SRANK, Scalar, STensor, ROWS, COLS>;

    /// Get the row stride of the image
    fn stride(&'a self) -> usize {
        let v = self.image_view();
        v.tensor_view.elem_view.strides()[0] as usize
    }

    /// Get the u,v pixel
    fn pixel(&'a self, u: usize, v: usize) -> STensor;

    /// Get the image size
    fn image_size(&'a self) -> ImageSize;
}

macro_rules! image_view {
    ($scalar_rank:literal, $srank:literal) => {
        impl<
            'a,
            Scalar: IsCoreScalar + 'static,
            STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS> + 'static,
            const ROWS: usize,
            const COLS: usize,
        > GenImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS>
        where
            TensorView<'a, $scalar_rank, 2, $srank, Scalar, STensor, ROWS, COLS>:
                IsTensorView<'a, $scalar_rank, 2, $srank, Scalar, STensor, ROWS, COLS>,
            ndarray::Dim<[ndarray::Ix; $scalar_rank]>: ndarray::Dimension,
        {
            /// Create a new image view from an array view
            pub fn new(
                elem_view: ndarray::ArrayView<'a, STensor, ndarray::Dim<[ndarray::Ix; 2]>>,
            ) -> Self {
                Self {
                    tensor_view: TensorView::<
                        'a,
                        $scalar_rank,
                        2,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS,
                    >::new(elem_view),
                }
            }

            /// Create a new image view from an image size and a slice of data
            pub fn from_size_and_slice(image_size: ImageSize, slice: &'a [STensor]) -> Self {
                Self {
                    tensor_view: TensorView::<
                        'a,
                        $scalar_rank,
                        2,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS,
                    >::from_shape_and_slice(
                        [image_size.height, image_size.width], slice
                    ),
                }
            }

            /// Create a new image view from an image size, stride and a slice of data
            pub fn from_stride_and_slice(
                image_size: ImageSize,
                stride: usize,
                slice: &'a [STensor],
            ) -> Self {
                let elem_view = ndarray::ArrayView::from_shape(
                    (image_size.height, image_size.width).strides((stride, 1)),
                    slice,
                )
                .unwrap();

                Self {
                    tensor_view: TensorView::<
                        'a,
                        $scalar_rank,
                        2,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS,
                    >::new(elem_view),
                }
            }

            /// return the underlying slice of pixel data
            pub fn as_slice(&self) -> &[STensor] {
                self.tensor_view.elem_view.as_slice().unwrap()
            }

            /// return the underlying slice of scalar data
            pub fn as_scalar_slice(&self) -> &[Scalar] {
                self.tensor_view.scalar_view.as_slice().unwrap()
            }

            /// return a sub-view of the image view
            pub fn sub_view(&'a self, start: [usize; 2], size: [usize; 2]) -> Self {
                Self {
                    tensor_view: TensorView::<
                        'a,
                        $scalar_rank,
                        2,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS,
                    >::new(self.tensor_view.elem_view.slice(ndarray::s![
                        start[0]..start[0] + size[0],
                        start[1]..start[1] + size[1]
                    ])),
                }
            }
        }

        impl<
            'a,
            Scalar: IsCoreScalar + 'static,
            STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS> + 'static,
            const ROWS: usize,
            const COLS: usize,
        > IsImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS>
            for GenImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS>
        where
            TensorView<'a, $scalar_rank, 2, $srank, Scalar, STensor, ROWS, COLS>:
                IsTensorView<'a, $scalar_rank, 2, $srank, Scalar, STensor, ROWS, COLS>,
            ndarray::Dim<[ndarray::Ix; $scalar_rank]>: ndarray::Dimension,
        {
            fn pixel(&'a self, u: usize, v: usize) -> STensor {
                // NOTE:
                // We are converting from Image Indexing Convention (d0 = u = col_idx, d1 = v =
                // row_idx) to tensor / matrix convention (d0 = rows, d1 = cols).
                self.tensor_view.get([v, u])
            }

            fn image_view(
                &'a self,
            ) -> GenImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS> {
                Self {
                    tensor_view: self.tensor_view.clone(),
                }
            }

            fn image_size(&'a self) -> ImageSize {
                self.tensor_view.dims().into()
            }
        }
    };
}

image_view!(2, 0);
image_view!(3, 1);
image_view!(4, 2);

#[cfg(test)]
mod tests {

    #[test]
    fn empty_image() {}
}
