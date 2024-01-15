use crate::tensor::element::IsStaticTensor;
use crate::tensor::element::IsTensorScalar;
use crate::tensor::view::IsTensorLike;
use crate::tensor::view::IsTensorView;
use crate::tensor::view::TensorView;

#[derive(Debug, Copy, Clone, Default)]
pub struct ImageSize {
    pub width: usize,
    pub height: usize,
}

impl From<[usize; 2]> for ImageSize {
    // We are converting from Tensor (and matrix) convention (d0 = rows, d1 = cols)
    // to Matrix convention (d0 = width = cols, d1 = height = rows)
    fn from(rows_cols: [usize; 2]) -> Self {
        ImageSize {
            width: rows_cols[1],
            height: rows_cols[0],
        }
    }
}

impl From<ImageSize> for [usize; 2] {
    // We are converting from Image Indexing Convention (d0 = width = cols, d1 = height = rows)
    // to tensor (and matrix) convention  (d0 = rows, d1 = cols).
    fn from(image_size: ImageSize) -> Self {
        [image_size.height, image_size.width]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageView<
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
    pub tensor_view: TensorView<'a, SCALAR_RANK, 2, SRANK, Scalar, STensor, ROWS, COLS, BATCHES>,
}

pub trait IsImageView<
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
    fn image_view(
        &'a self,
    ) -> ImageView<'a, SCALAR_RANK, SRANK, Scalar, STensor, ROWS, COLS, BATCHES>;
    fn pixel(&'a self, u: usize, v: usize) -> STensor;
    fn image_size(&'a self) -> ImageSize;
}

macro_rules! image_view {
    ($scalar_rank:literal, $srank:literal) => {
        impl<
                'a,
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS, BATCHES> + 'static,
                const BATCHES: usize,
                const ROWS: usize,
                const COLS: usize,
            > IsImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
            for ImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
        where
            TensorView<'a, $scalar_rank, 2, $srank, Scalar, STensor, ROWS, COLS, BATCHES>:
                IsTensorView<'a, $scalar_rank, 2, $srank, Scalar, STensor, ROWS, COLS, BATCHES>,
            ndarray::Dim<[ndarray::Ix; $scalar_rank]>: ndarray::Dimension,
        {
            fn pixel(&'a self, u: usize, v: usize) -> STensor {
                // NOTE:
                // We are converting from Image Indexing Convention (d0 = u = col_idx, d1 = v = row_idx)
                // to tensor (and matrix) convention (d0 = rows, d1 = cols).
                self.tensor_view.get([v, u])
            }

            fn image_view(
                &'a self,
            ) -> ImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES> {
                Self {
                    tensor_view: self.tensor_view,
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
image_view!(5, 2);

#[cfg(test)]
mod tests {

    #[test]
    fn empty_image() {}
}
