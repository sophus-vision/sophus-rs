use crate::tensor::element::IsStaticTensor;
use crate::tensor::element::IsTensorScalar;
use crate::tensor::element::SVec;
use crate::tensor::mut_tensor::MutTensor;
use crate::tensor::mut_view::IsMutTensorLike;

use super::mut_view::IsMutImageView;
use super::view::ImageSize;
use super::view::ImageView;
use super::view::IsImageView;

//pub type MutImage<Element, const ROWS: usize> = MutTensor<2, Element, ROWS, 1>;

#[derive(Debug, Clone, Default)]
pub struct MutImage<
    const SCALAR_RANK: usize,
    const SRANK: usize,
    Scalar: IsTensorScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS, BATCHES> + 'static,
    const ROWS: usize,
    const COLS: usize,
    const BATCHES: usize,
> {
    pub mut_tensor: MutTensor<SCALAR_RANK, 2, SRANK, Scalar, STensor, ROWS, COLS, BATCHES>,
}

pub type MutImageT<Scalar, const ROWS: usize> =
    MutImage<3, 1, Scalar, SVec<Scalar, ROWS>, ROWS, 1, 1>;

pub type MutImageU8 = MutImageT<u8, 1>;
pub type MutImageU16 = MutImageT<u16, 1>;
pub type MutImageF32 = MutImageT<f32, 1>;
pub type MutImage2U8 = MutImageT<u8, 2>;
pub type MutImage2U16 = MutImageT<u16, 2>;
pub type MutImage2F32 = MutImageT<f32, 2>;
pub type MutImage3U8 = MutImageT<u8, 3>;
pub type MutImage3U16 = MutImageT<u16, 3>;
pub type MutImage3F32 = MutImageT<f32, 3>;
pub type MutImage4U8 = MutImageT<u8, 4>;
pub type MutImage4U16 = MutImageT<u16, 4>;
pub type MutImage4F32 = MutImageT<f32, 4>;

pub trait IsMutImage<
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
    fn from_image_size(size: ImageSize) -> Self;
    fn from_image_size_and_val(size: ImageSize, val: STensor) -> Self;
}

macro_rules! mut_image {
    ($scalar_rank:literal, $srank:literal) => {
        impl<
            'a,
            Scalar: IsTensorScalar + 'static,
            STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS, BATCHES> + 'static,
            const BATCHES: usize,
            const ROWS: usize,
            const COLS: usize,
        > IsImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
        for MutImage<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
        where
            ndarray::Dim<[ndarray::Ix; $scalar_rank]>: ndarray::Dimension,
        {
            fn image_view(
                &'a self,
            ) -> super::view::ImageView<
                'a,
                $scalar_rank,
                $srank,
                Scalar,
                STensor,
                ROWS,
                COLS,
                BATCHES,
            > {
                let v = self.mut_tensor.view();
                ImageView { tensor_view: v }
            }

            fn pixel(&'a self, u: usize, v: usize) -> STensor {
                self.mut_tensor.mut_array[[v, u]]
            }

            fn image_size(&self) -> super::view::ImageSize {
                self.image_view().image_size()
            }
        }

        impl<
            'a,
            Scalar: IsTensorScalar + 'static,
            STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS, BATCHES> + 'static,
            const BATCHES: usize,
            const ROWS: usize,
            const COLS: usize,
        > MutImage<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
        {
            pub fn from_image_size(size: super::view::ImageSize) -> Self {
                Self {
                    mut_tensor: MutTensor::<
                        $scalar_rank,
                        2,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS,
                        BATCHES,
                    >::from_shape(size.into()),
                }
            }

            pub fn from_image_size_and_val(size: super::view::ImageSize, val: STensor) -> Self {
                Self {
                    mut_tensor: MutTensor::<
                        $scalar_rank,
                        2,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS,
                        BATCHES,
                    >::from_shape_and_val(size.into(), val),
                }
            }
        }

        impl<
            'a,
            Scalar: IsTensorScalar + 'static,
            STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS, BATCHES> + 'static,
            const BATCHES: usize,
            const ROWS: usize,
            const COLS: usize,
        > IsMutImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
        for MutImage<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
        where
            ndarray::Dim<[ndarray::Ix; $scalar_rank]>: ndarray::Dimension,
        {
            fn mut_image_view<'b: 'a>(
                &'b mut self,
            ) -> super::mut_view::MutImageView<
                'a,
                $scalar_rank,
                $srank,
                Scalar,
                STensor,
                ROWS,
                COLS,
                BATCHES,
            > {
                super::mut_view::MutImageView {
                    mut_tensor_view: self.mut_tensor.mut_view(),
                }
            }

            fn mut_pixel(&'a mut self, u: usize, v: usize) -> &mut STensor {
                self.mut_tensor.get_mut([v, u])
            }
        }
    };
}

mut_image!(2, 0);
mut_image!(3, 1);
mut_image!(4, 2);
