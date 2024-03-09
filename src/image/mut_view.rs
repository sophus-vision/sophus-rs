use crate::tensor::element::IsStaticTensor;
use crate::tensor::element::IsTensorScalar;
use crate::tensor::mut_view::IsMutTensorLike;
use crate::tensor::mut_view::MutTensorView;
use crate::tensor::view::IsTensorLike;

use super::view::GenImageView;
use super::view::ImageSize;
use super::view::IsImageView;

#[derive(Debug, PartialEq)]
pub struct GenMutImageView<
    'a,
    const SCALAR_RANK: usize,
    const SRANK: usize,
    Scalar: IsTensorScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS, BATCHES> + 'static,
    const ROWS: usize,
    const COLS: usize,
    const BATCHES: usize,
> where
    ndarray::Dim<[ndarray::Ix; SCALAR_RANK]>: ndarray::Dimension,
{
    pub mut_tensor_view:
        MutTensorView<'a, SCALAR_RANK, 2, SRANK, Scalar, STensor, ROWS, COLS, BATCHES>,
}

macro_rules! mut_image_view {
    ($scalar_rank:literal, $srank:literal) => {
        impl<
                'a,
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS, BATCHES> + 'static,
                const ROWS: usize,
                const COLS: usize,
                const BATCHES: usize,
            > IsImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
            for GenMutImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
        where
            ndarray::Dim<[ndarray::Ix; $scalar_rank]>: ndarray::Dimension,
        {
            fn pixel(&'a self, u: usize, v: usize) -> STensor {
                // NOTE:
                // We are converting from Image Indexing Convention (d0 = u = col_idx, d1 = v = row_idx)
                // to tensor (and matrix) convention (d0 = rows, d1 = cols).
                self.mut_tensor_view.get([v, u])
            }

            fn image_view(
                &'a self,
            ) -> GenImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES> {
                let view = self.mut_tensor_view.view();
                GenImageView { tensor_view: view }
            }

            fn image_size(&'a self) -> ImageSize {
                self.mut_tensor_view.dims().into()
            }
        }

        impl<
                'a,
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS, BATCHES> + 'static,
                const ROWS: usize,
                const COLS: usize,
                const BATCHES: usize,
            > IsMutImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
            for GenMutImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
        where
            MutTensorView<'a, $scalar_rank, 2, $srank, Scalar, STensor, ROWS, COLS, BATCHES>:
                IsMutTensorLike<'a, $scalar_rank, 2, $srank, Scalar, STensor, ROWS, COLS, BATCHES>,
            ndarray::Dim<[ndarray::Ix; $scalar_rank]>: ndarray::Dimension,
        {
            fn mut_image_view<'b: 'a>(
                &'b mut self,
            ) -> GenMutImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
            {
                GenMutImageView {
                    mut_tensor_view: MutTensorView::<
                        'a,
                        $scalar_rank,
                        2,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS,
                        BATCHES,
                    >::new(
                        self.mut_tensor_view.elem_view_mut.view_mut()
                    ),
                }
            }

            fn mut_pixel(&'a mut self, u: usize, v: usize) -> &mut STensor {
                // NOTE:
                // We are converting from Image Indexing Convention (d0 = u = col_idx, d1 = v = row_idx)
                // to tensor (and matrix) convention (d0 = rows, d1 = cols).
                self.mut_tensor_view.get_mut([v, u])
            }
        }
    };
}

mut_image_view!(2, 0);
mut_image_view!(3, 1);
mut_image_view!(4, 2);

pub trait IsMutImageView<
    'a,
    const SCALAR_RANK: usize,
    const SRANK: usize,
    Scalar: IsTensorScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS, BATCHES> + 'static,
    const ROWS: usize,
    const COLS: usize,
    const BATCHES: usize,
> where
    ndarray::Dim<[ndarray::Ix; SCALAR_RANK]>: ndarray::Dimension,
{
    fn mut_image_view<'b: 'a>(
        &'b mut self,
    ) -> GenMutImageView<'a, SCALAR_RANK, SRANK, Scalar, STensor, ROWS, COLS, BATCHES>;

    fn mut_pixel(&'a mut self, u: usize, v: usize) -> &mut STensor;
}
