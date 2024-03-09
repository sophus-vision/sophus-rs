use crate::tensor::element::IsStaticTensor;
use crate::tensor::element::IsTensorScalar;
use crate::tensor::element::SVec;
use crate::tensor::view::IsTensorLike;
use crate::tensor::view::IsTensorView;
use crate::tensor::view::TensorView;

#[derive(Debug, Copy, Clone, Default)]
pub struct ImageSize {
    pub width: usize,
    pub height: usize,
}

impl ImageSize {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    pub fn area(&self) -> usize {
        self.width * self.height
    }
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
pub struct GenImageView<
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

// Image view of scalar values
pub type ImageViewS<'a, Scalar> = GenImageView<'a, 2, 0, Scalar, Scalar, 1, 1, 1>;

// Image view of vector values
//
// Here, R indicates the number of rows in the vector
pub type ImageViewV<'a, Scalar, const ROWS: usize> =
    GenImageView<'a, 3, 1, Scalar, SVec<Scalar, ROWS>, ROWS, 1, 1>;

pub type ImageViewU8<'a> = ImageViewS<'a, u8>;
pub type ImageViewU16<'a> = ImageViewS<'a, u16>;
pub type ImageViewF32<'a> = ImageViewS<'a, f32>;
pub type ImageView2U8<'a> = ImageViewV<'a, u8, 2>;
pub type ImageView2U16<'a> = ImageViewV<'a, u16, 2>;
pub type ImageView2F32<'a> = ImageViewV<'a, f32, 2>;
pub type ImageView3U8<'a> = ImageViewV<'a, u8, 3>;
pub type ImageView3U16<'a> = ImageViewV<'a, u16, 3>;
pub type ImageView3F32<'a> = ImageViewV<'a, f32, 3>;
pub type ImageView4U8<'a> = ImageViewV<'a, u8, 4>;
pub type ImageView4U16<'a> = ImageViewV<'a, u16, 4>;
pub type ImageView4F32<'a> = ImageViewV<'a, f32, 4>;

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
    ) -> GenImageView<'a, SCALAR_RANK, SRANK, Scalar, STensor, ROWS, COLS, BATCHES>;
    fn stride(&'a self) -> usize {
        let v = self.image_view();
        v.tensor_view.elem_view.strides()[0] as usize
    }
    fn pixel(&'a self, u: usize, v: usize) -> STensor;
    fn image_size(&'a self) -> ImageSize;
}

macro_rules! image_view {
    ($scalar_rank:literal, $srank:literal) => {
        impl<
                'a,
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS, BATCHES> + 'static,
                const ROWS: usize,
                const COLS: usize,
                const BATCHES: usize,
            > GenImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
        where
            TensorView<'a, $scalar_rank, 2, $srank, Scalar, STensor, ROWS, COLS, BATCHES>:
                IsTensorView<'a, $scalar_rank, 2, $srank, Scalar, STensor, ROWS, COLS, BATCHES>,
            ndarray::Dim<[ndarray::Ix; $scalar_rank]>: ndarray::Dimension,
        {
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
                        BATCHES,
                    >::from_shape_and_slice(
                        [image_size.height, image_size.width], slice
                    ),
                }
            }

            pub fn as_slice(&self) -> &[STensor] {
                self.tensor_view.elem_view.as_slice().unwrap()
            }

            pub fn as_scalar_slice(&self) -> &[Scalar] {
                self.tensor_view.scalar_view.as_slice().unwrap()
            }

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
                        BATCHES,
                    >::new(self.tensor_view.elem_view.slice(ndarray::s![
                        start[0]..start[0] + size[0],
                        start[1]..start[1] + size[1]
                    ])),
                }
            }
        }

        impl<
                'a,
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS, BATCHES> + 'static,
                const ROWS: usize,
                const COLS: usize,
                const BATCHES: usize,
            > IsImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
            for GenImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
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
            ) -> GenImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES> {
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

#[cfg(test)]
mod tests {

    #[test]
    fn empty_image() {}
}
