use crate::image::view::ImageSize;
use crate::image::view::ImageView;
use crate::image::view::IsImageView;
use crate::image::mut_image::MutImage;
use crate::tensor::arc_tensor::ArcTensor;
use crate::tensor::element::IsStaticTensor;
use crate::tensor::element::IsTensorScalar;
use crate::tensor::element::SVec;
use crate::tensor::view::IsTensorLike;

#[derive(Debug, Clone)]
pub struct ArcImage<
    const HYPER_RANK: usize,
    const SRANK: usize,
    Scalar: IsTensorScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, BATCHES, ROWS, COLS> + 'static,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
> {
    pub tensor: ArcTensor<HYPER_RANK, SRANK, 2, Scalar, STensor, BATCHES, ROWS, COLS>,
}

pub type ArcImageT<Scalar, const ROWS: usize> =
    ArcImage<3, 1, Scalar, SVec<Scalar, ROWS>, 1, ROWS, 1>;

pub type ArcImageU8 = ArcImageT<u8, 1>;
pub type ArcImageU16 = ArcImageT<u16, 1>;
pub type ArcImageF32 = ArcImageT<f32, 1>;
pub type ArcImage2U8 = ArcImageT<u8, 2>;
pub type ArcImage2U16 = ArcImageT<u16, 2>;
pub type ArcImage2F32 = ArcImageT<f32, 2>;
pub type ArcImage3U8 = ArcImageT<u8, 3>;
pub type ArcImage3U16 = ArcImageT<u16, 3>;
pub type ArcImage3F32 = ArcImageT<f32, 3>;
pub type ArcImage4U8 = ArcImageT<u8, 4>;
pub type ArcImage4U16 = ArcImageT<u16, 4>;
pub type ArcImage4F32 = ArcImageT<f32, 4>;

macro_rules! arc_image {
    ($hyper_rank:literal, $srank:literal) => {

        impl<
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, BATCHES, ROWS, COLS> + 'static,
                const BATCHES: usize,
                const ROWS: usize,
                const COLS: usize,
            > ArcImage<$hyper_rank, $srank, Scalar, STensor, BATCHES, ROWS, COLS>
        {
            pub fn from_mut_image(
                image: MutImage<$hyper_rank, $srank, Scalar, STensor, BATCHES, ROWS, COLS>,
            ) -> Self {
                Self {
                    tensor: ArcTensor::<$hyper_rank, $srank, 2,Scalar, STensor, BATCHES, ROWS, COLS>
                        ::from_mut_tensor(image.mut_tensor),
                }
            }

            pub fn from_image_size_and_val(size: ImageSize, val: STensor) -> Self {
                Self {
                    tensor: ArcTensor::<$hyper_rank, $srank, 2,Scalar, STensor, BATCHES, ROWS, COLS>::from_shape_and_val(size.into(), val),
                }
            }
        }


        impl<
                'b,
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, BATCHES, ROWS, COLS> + 'static,
                const BATCHES: usize,
                const ROWS: usize,
                const COLS: usize,
            > IsImageView<'b, $hyper_rank, $srank, Scalar, STensor, BATCHES, ROWS, COLS>
            for ArcImage<$hyper_rank, $srank, Scalar, STensor, BATCHES, ROWS, COLS> where
            ndarray::Dim<[ndarray::Ix; $hyper_rank]>: ndarray::Dimension,
        {
            fn image_view(
                &'b self,
            ) -> super::view::ImageView<
                '_,
                $hyper_rank,
                $srank,
                Scalar,
                STensor,
                BATCHES,
                ROWS,
                COLS,
            > {
                ImageView {
                    tensor_view: self.tensor.view(),
                }
            }

            fn pixel(&'b self, u: usize, v: usize) -> STensor {
                self.tensor.get([v, u])
            }

            fn image_size(&self) -> super::view::ImageSize {
                self.image_view().image_size()
            }
        }
    };
}

arc_image!(2, 0);
arc_image!(3, 1);
arc_image!(4, 2);
