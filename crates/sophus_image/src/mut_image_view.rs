use crate::image_view::GenImageView;
use crate::image_view::ImageSize;
use crate::image_view::IsImageView;

use sophus_tensor::element::IsStaticTensor;
use sophus_tensor::element::IsTensorScalar;
use sophus_tensor::mut_view::IsMutTensorLike;
use sophus_tensor::mut_view::MutTensorView;
use sophus_tensor::view::IsTensorLike;

/// Mutable image view of a static tensors
#[derive(Debug, PartialEq)]
pub struct GenMutImageView<
    'a,
    const TOTAL_RANK: usize,
    const SRANK: usize,
    Scalar: IsTensorScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS, BATCH_SIZE> + 'static,
    const ROWS: usize,
    const COLS: usize,
    const BATCH_SIZE: usize,
> where
    ndarray::Dim<[ndarray::Ix; TOTAL_RANK]>: ndarray::Dimension,
{
    /// underlying mutable tensor view
    pub mut_tensor_view:
        MutTensorView<'a, TOTAL_RANK, 2, SRANK, Scalar, STensor, ROWS, COLS, BATCH_SIZE>,
}

macro_rules! mut_image_view {
    ($scalar_rank:literal, $srank:literal) => {
        impl<
                'a,
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS, BATCH_SIZE> + 'static,
                const ROWS: usize,
                const COLS: usize,
                const BATCH_SIZE: usize,
            > IsImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCH_SIZE>
            for GenMutImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCH_SIZE>
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
            ) -> GenImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCH_SIZE>
            {
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
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS, BATCH_SIZE> + 'static,
                const ROWS: usize,
                const COLS: usize,
                const BATCH_SIZE: usize,
            > IsMutImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCH_SIZE>
            for GenMutImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCH_SIZE>
        where
            MutTensorView<'a, $scalar_rank, 2, $srank, Scalar, STensor, ROWS, COLS, BATCH_SIZE>:
                IsMutTensorLike<
                    'a,
                    $scalar_rank,
                    2,
                    $srank,
                    Scalar,
                    STensor,
                    ROWS,
                    COLS,
                    BATCH_SIZE,
                >,
            ndarray::Dim<[ndarray::Ix; $scalar_rank]>: ndarray::Dimension,
        {
            fn mut_image_view<'b: 'a>(
                &'b mut self,
            ) -> GenMutImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCH_SIZE>
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
                        BATCH_SIZE,
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

/// is a mutable image view
pub trait IsMutImageView<
    'a,
    const TOTAL_RANK: usize,
    const SRANK: usize,
    Scalar: IsTensorScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS, BATCH_SIZE> + 'static,
    const ROWS: usize,
    const COLS: usize,
    const BATCH_SIZE: usize,
> where
    ndarray::Dim<[ndarray::Ix; TOTAL_RANK]>: ndarray::Dimension,
{
    /// returns mutable image view
    fn mut_image_view<'b: 'a>(
        &'b mut self,
    ) -> GenMutImageView<'a, TOTAL_RANK, SRANK, Scalar, STensor, ROWS, COLS, BATCH_SIZE>;

    /// returns mutable u,v pixel
    fn mut_pixel(&'a mut self, u: usize, v: usize) -> &mut STensor;
}
