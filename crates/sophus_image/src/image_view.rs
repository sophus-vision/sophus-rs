use sophus_core::linalg::scalar::IsCoreScalar;
use sophus_core::linalg::SVec;
use sophus_core::tensor::element::IsStaticTensor;
use sophus_core::tensor::tensor_view::IsTensorLike;
use sophus_core::tensor::tensor_view::IsTensorView;
use sophus_core::tensor::tensor_view::TensorView;

/// Image size
#[derive(Debug, Copy, Clone, Default)]
pub struct ImageSize {
    /// Width of the image - number of columns
    pub width: usize,
    /// Height of the image - number of rows
    pub height: usize,
}

impl ImageSize {
    /// Create a new image size from width and height
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Get the area of the image - width * height
    pub fn area(&self) -> usize {
        self.width * self.height
    }
}

impl From<[usize; 2]> for ImageSize {
    /// We are converting from Tensor (and matrix) convention (d0 = rows, d1 = cols)
    /// to Matrix convention (d0 = width = cols, d1 = height = rows)
    fn from(rows_cols: [usize; 2]) -> Self {
        ImageSize {
            width: rows_cols[1],
            height: rows_cols[0],
        }
    }
}

impl From<ImageSize> for [usize; 2] {
    /// We are converting from Image Indexing Convention (d0 = width = cols, d1 = height = rows)
    /// to tensor (and matrix) convention  (d0 = rows, d1 = cols).
    fn from(image_size: ImageSize) -> Self {
        [image_size.height, image_size.width]
    }
}

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

/// Image view of scalar values
pub type ImageView<'a, Scalar> = GenImageView<'a, 2, 0, Scalar, Scalar, 1, 1>;

/// Image view of vector values
///
/// Here, R indicates the number of rows in the vector
pub type ImageViewR<'a, Scalar, const ROWS: usize> =
    GenImageView<'a, 3, 1, Scalar, SVec<Scalar, ROWS>, ROWS, 1>;

/// Image view of u8 values
pub type ImageViewU8<'a> = ImageView<'a, u8>;
/// Image view of u16 values
pub type ImageViewU16<'a> = ImageView<'a, u16>;
/// Image view of f32 values
pub type ImageViewF32<'a> = ImageView<'a, f32>;
/// Image view of u8 2-vectors
pub type ImageView2U8<'a> = ImageViewR<'a, u8, 2>;
/// Image view of u16 2-vectors
pub type ImageView2U16<'a> = ImageViewR<'a, u16, 2>;
/// Image view of f32 2-vectors
pub type ImageView2F32<'a> = ImageViewR<'a, f32, 2>;
/// Image view of u8 3-vectors
pub type ImageView3U8<'a> = ImageViewR<'a, u8, 3>;
/// Image view of u16 3-vectors
pub type ImageView3U16<'a> = ImageViewR<'a, u16, 3>;
/// Image view of f32 3-vectors
pub type ImageView3F32<'a> = ImageViewR<'a, f32, 3>;
/// Image view of u8 4-vectors
pub type ImageView4U8<'a> = ImageViewR<'a, u8, 4>;
/// Image view of u16 4-vectors
pub type ImageView4U16<'a> = ImageViewR<'a, u16, 4>;
/// Image view of f32 4-vectors
pub type ImageView4F32<'a> = ImageViewR<'a, f32, 4>;

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
                // We are converting from Image Indexing Convention (d0 = u = col_idx, d1 = v = row_idx)
                // to tensor (and matrix) convention (d0 = rows, d1 = cols).
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
