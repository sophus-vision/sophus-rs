use crate::tensor::element::IsStaticTensor;
use crate::tensor::element::IsTensorScalar;
use crate::tensor::mut_view::IsMutTensorLike;
use crate::tensor::mut_view::MutTensorView;
use crate::tensor::view::IsTensorLike;

use super::view::ImageSize;
use super::view::ImageView;
use super::view::IsImageView;

#[derive(Debug, PartialEq)]
pub struct MutImageView<
    'a,
    const HYPER_RANK: usize,
    const SRANK: usize,
    Scalar: IsTensorScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, BATCHES, ROWS, COLS> + 'static,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
> where
    ndarray::Dim<[ndarray::Ix; HYPER_RANK]>: ndarray::Dimension,
{
    pub mut_tensor_view:
        MutTensorView<'a, HYPER_RANK, SRANK, 2, Scalar, STensor, BATCHES, ROWS, COLS>,
}

macro_rules! mut_image_view {
    ($hyper_rank:literal, $srank:literal) => {
        impl<
                'a,
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, BATCHES, ROWS, COLS> + 'static,
                const BATCHES: usize,
                const ROWS: usize,
                const COLS: usize,
            > IsImageView<'a, $hyper_rank, $srank, Scalar, STensor, BATCHES, ROWS, COLS>
            for MutImageView<'a, $hyper_rank, $srank, Scalar, STensor, BATCHES, ROWS, COLS>
        where
            ndarray::Dim<[ndarray::Ix; $hyper_rank]>: ndarray::Dimension,
        {
            fn pixel(&'a self, u: usize, v: usize) -> STensor {
                // NOTE:
                // We are converting from Image Indexing Convention (d0 = u = col_idx, d1 = v = row_idx)
                // to tensor (and matrix) convention (d0 = rows, d1 = cols).
                self.mut_tensor_view.get([v, u])
            }

            fn image_view(
                &'a self,
            ) -> ImageView<'a, $hyper_rank, $srank, Scalar, STensor, BATCHES, ROWS, COLS> {
                let view = self.mut_tensor_view.view();
                ImageView { tensor_view: view }
            }

            fn image_size(&'a self) -> ImageSize {
                self.mut_tensor_view.dims().into()
            }
        }

        impl<
                'a,
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, BATCHES, ROWS, COLS> + 'static,
                const BATCHES: usize,
                const ROWS: usize,
                const COLS: usize,
            > IsMutImageView<'a, $hyper_rank, $srank, Scalar, STensor, BATCHES, ROWS, COLS>
            for MutImageView<'a, $hyper_rank, $srank, Scalar, STensor, BATCHES, ROWS, COLS>
        where
            MutTensorView<'a, $hyper_rank, $srank, 2, Scalar, STensor, BATCHES, ROWS, COLS>:
                IsMutTensorLike<'a, $hyper_rank, $srank, 2, Scalar, STensor, BATCHES, ROWS, COLS>,
            ndarray::Dim<[ndarray::Ix; $hyper_rank]>: ndarray::Dimension,
        {
            fn mut_image_view<'b: 'a>(
                &'b mut self,
            ) -> MutImageView<'a, $hyper_rank, $srank, Scalar, STensor, BATCHES, ROWS, COLS> {
                MutImageView {
                    mut_tensor_view: MutTensorView::<
                        'a,
                        $hyper_rank,
                        $srank,
                        2,
                        Scalar,
                        STensor,
                        BATCHES,
                        ROWS,
                        COLS,
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
    const HYPER_RANK: usize,
    const SRANK: usize,
    Scalar: IsTensorScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, BATCHES, ROWS, COLS> + 'static,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
> where
    ndarray::Dim<[ndarray::Ix; HYPER_RANK]>: ndarray::Dimension,
{
    fn mut_image_view<'b: 'a>(
        &'b mut self,
    ) -> MutImageView<'a, HYPER_RANK, SRANK, Scalar, STensor, BATCHES, ROWS, COLS>;

    fn mut_pixel(&'a mut self, u: usize, v: usize) -> &mut STensor;
}
