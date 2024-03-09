use crate::tensor::element::IsStaticTensor;
use crate::tensor::element::IsTensorScalar;
use crate::tensor::element::SVec;
use crate::tensor::mut_tensor::MutTensor;
use crate::tensor::mut_view::IsMutTensorLike;

use super::mut_view::IsMutImageView;
use super::view::GenImageView;
use super::view::ImageSize;
use super::view::IsImageView;
use crate::image::arc_image::GenArcImage;
use crate::tensor::view::IsTensorView;
use crate::tensor::view::TensorView;

#[derive(Debug, Clone, Default)]
pub struct GenMutImage<
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

// Mutable image of scalar values
pub type MutImage<Scalar> = GenMutImage<2, 0, Scalar, Scalar, 1, 1, 1>;

// Mutable image of vector values
//
// Here, R indicates the number of rows in the vector
pub type MutImageR<Scalar, const ROWS: usize> =
    GenMutImage<3, 1, Scalar, SVec<Scalar, ROWS>, ROWS, 1, 1>;

pub type MutImageU8 = MutImage<u8>;
pub type MutImageU16 = MutImage<u16>;
pub type MutImageF32 = MutImage<f32>;
pub type MutImage2U8 = MutImageR<u8, 2>;
pub type MutImage2U16 = MutImageR<u16, 2>;
pub type MutImage2F32 = MutImageR<f32, 2>;
pub type MutImage3U8 = MutImageR<u8, 3>;
pub type MutImage3U16 = MutImageR<u16, 3>;
pub type MutImage3F32 = MutImageR<f32, 3>;
pub type MutImage4U8 = MutImageR<u8, 4>;
pub type MutImage4U16 = MutImageR<u16, 4>;
pub type MutImage4F32 = MutImageR<f32, 4>;

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
                const ROWS: usize,
                const COLS: usize,
                const BATCHES: usize,
            > IsImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
            for GenMutImage<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
        where
            ndarray::Dim<[ndarray::Ix; $scalar_rank]>: ndarray::Dimension,
        {
            fn image_view(
                &'a self,
            ) -> super::view::GenImageView<
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
                GenImageView { tensor_view: v }
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
                const ROWS: usize,
                const COLS: usize,
                const BATCHES: usize,
            > GenMutImage<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
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

            pub fn make_copy_from(
                v: &GenImageView<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>,
            ) -> Self {
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
                    >::make_copy_from(&v.tensor_view),
                }
            }

            pub fn make_copy_from_size_and_slice(image_size: ImageSize, slice: &'a [STensor]) -> Self {
                Self::make_copy_from(&GenImageView::<
                        $scalar_rank,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS,
                        BATCHES,
                    >::from_size_and_slice(image_size, slice))

            }


            pub fn try_make_copy_from_make_from_size_and_bytes(
                image_size: ImageSize,
                bytes: &'a [u8],
            ) -> Result<Self,String> {
                let num_scalars_in_pixel = STensor::num_scalars();
                let num_scalars_in_image =  num_scalars_in_pixel * image_size.width * image_size.height;
                let size_in_bytes = num_scalars_in_image * std::mem::size_of::<Scalar>();
                if(bytes.len() != size_in_bytes){
                    return Err(format!("bytes.len() = {} != size_in_bytes = {}",
                                       bytes.len(), size_in_bytes));
                }
                let stensor_slice = unsafe {
                    std::slice::from_raw_parts(
                        bytes.as_ptr() as *const Scalar,
                        num_scalars_in_image)
                };

                let mut img = Self::from_image_size(image_size);

                for v in 0..image_size.height {
                    for u in 0..image_size.width {
                        let idx = (v * image_size.width + u) * STensor::num_scalars() ;
                        let pixel = &stensor_slice[idx..idx + STensor::num_scalars()];
                        *img.mut_pixel(u, v) = STensor::from_slice(pixel);
                    }
                }
                Ok(img)
            }

            pub fn make_copy_from_make_from_size_and_bytes(
                image_size: ImageSize,
                bytes: &'a [u8],
            ) -> Self {
               Self::try_make_copy_from_make_from_size_and_bytes(image_size, bytes).unwrap()
            }


            pub fn from_map<
            'b,
            const OTHER_HRANK: usize,
            const OTHER_SRANK: usize,
            OtherScalar: IsTensorScalar + 'static,
            OtherSTensor: IsStaticTensor<
                OtherScalar,
                OTHER_SRANK,
                OTHER_ROWS,
                OTHER_COLS,
                OTHER_BATCHES,
            > + 'static,
            const OTHER_ROWS: usize,
            const OTHER_COLS: usize,
            const OTHER_BATCHES: usize,
            F: FnMut(&OtherSTensor)-> STensor
            >(
                v: &'b  GenImageView::<
                'b,
                OTHER_HRANK,
                OTHER_SRANK,
                OtherScalar,
                OtherSTensor,
                OTHER_ROWS,
                OTHER_COLS,
                OTHER_BATCHES,
            >,
                op: F,

            ) -> Self
              where ndarray::Dim<[ndarray::Ix; OTHER_HRANK]>: ndarray::Dimension,
                TensorView<'b, OTHER_HRANK, 2, OTHER_SRANK, OtherScalar, OtherSTensor,
                           OTHER_ROWS, OTHER_COLS, OTHER_BATCHES>:
                IsTensorView<'b, OTHER_HRANK, 2, OTHER_SRANK, OtherScalar, OtherSTensor,
                             OTHER_ROWS, OTHER_COLS, OTHER_BATCHES>,

            {
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
                    >::from_map(&v.tensor_view, op),
                }
            }


            pub fn to_shared(
                self,
            ) -> GenArcImage<
                $scalar_rank,
                $srank,
                Scalar,
                STensor,
                ROWS,
                COLS,
                BATCHES,
            > {
                GenArcImage {
                    tensor: self.mut_tensor.to_shared(),
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
            > IsMutImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
            for GenMutImage<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
        where
            ndarray::Dim<[ndarray::Ix; $scalar_rank]>: ndarray::Dimension,
        {
            fn mut_image_view<'b: 'a>(
                &'b mut self,
            ) -> super::mut_view::GenMutImageView<
                'a,
                $scalar_rank,
                $srank,
                Scalar,
                STensor,
                ROWS,
                COLS,
                BATCHES,
            > {
                super::mut_view::GenMutImageView {
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
