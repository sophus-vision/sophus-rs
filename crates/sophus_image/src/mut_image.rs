use sophus_tensor::{
    MutTensor,
    TensorView,
};

use crate::{
    ImageSize,
    arc_image::GenArcImage,
    image_view::GenImageView,
    prelude::*,
};

extern crate alloc;

/// Mutable image of static tensors
#[derive(Debug, Clone, Default)]
pub struct GenMutImage<
    const TOTAL_RANK: usize,
    const SRANK: usize,
    Scalar: IsCoreScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS> + 'static,
    const ROWS: usize,
    const COLS: usize,
> {
    /// underlying mutable tensor
    pub mut_tensor: MutTensor<TOTAL_RANK, 2, SRANK, Scalar, STensor, ROWS, COLS>,
}

/// is a mutable image
pub trait IsMutImage<
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
    /// creates a mutable image view from image size
    fn from_image_size(size: ImageSize) -> Self;
    /// creates a mutable image view from image size and value
    fn from_image_size_and_val(size: ImageSize, val: STensor) -> Self;
}

macro_rules! mut_image {
    ($scalar_rank:literal, $srank:literal) => {
        impl<
                'a,
                Scalar: IsCoreScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS> + 'static,
                const ROWS: usize,
                const COLS: usize,

            > IsImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS>
            for GenMutImage<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS>
        where
            ndarray::Dim<[ndarray::Ix; $scalar_rank]>: ndarray::Dimension,
        {
            fn image_view(
                &'a self,
            ) -> crate::image_view::GenImageView<
                'a,
                $scalar_rank,
                $srank,
                Scalar,
                STensor,
                ROWS,
                COLS
            > {
                let v = self.mut_tensor.view();
                GenImageView { tensor_view: v }
            }

            fn pixel(&'a self, u: usize, v: usize) -> STensor {
                self.mut_tensor.mut_array[[v, u]].clone()
            }

            fn image_size(&self) -> ImageSize {
                self.image_view().image_size()
            }
        }

        impl<
                'a,
                Scalar: IsCoreScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS> + 'static,
                const ROWS: usize,
                const COLS: usize,

            > GenMutImage<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS>
        {
            /// creates a mutable image view from image size
            pub fn from_image_size(size: ImageSize) -> Self {
                Self {
                    mut_tensor: MutTensor::<
                        $scalar_rank,
                        2,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS
                    >::from_shape(size.into()),
                }

            }

            /// creates a mutable image from image size and value
            pub fn from_image_size_and_val(size: ImageSize, val: STensor) -> Self {
                Self {
                    mut_tensor: MutTensor::<
                        $scalar_rank,
                        2,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS
                    >::from_shape_and_val(size.into(), val),
                }
            }

            /// creates a mutable image from image view
            pub fn make_copy_from(
                v: &GenImageView<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS>,
            ) -> Self {
                Self {
                    mut_tensor: MutTensor::<
                        $scalar_rank,
                        2,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS
                    >::make_copy_from(&v.tensor_view),
                }
            }

            /// creates a mutable image from image size and slice
            pub fn make_copy_from_size_and_slice(image_size: ImageSize, slice: &'a [STensor]) -> Self {
                Self::make_copy_from(&GenImageView::<
                        $scalar_rank,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS,
                    >::from_size_and_slice(image_size, slice))

            }

            /// creates a mutable image from image size and byte slice
            pub fn try_make_copy_from_size_and_bytes(
                image_size: ImageSize,
                bytes: &'a [u8],
            ) -> Result<Self, alloc::string::String> {
                let num_scalars_in_pixel = STensor::num_scalars();
                let num_scalars_in_image =  num_scalars_in_pixel * image_size.width * image_size.height;
                let size_in_bytes = num_scalars_in_image * core::mem::size_of::<Scalar>();
                if(bytes.len() != size_in_bytes){
                    return Err(alloc::format!("bytes.len() = {} != size_in_bytes = {}",
                                       bytes.len(), size_in_bytes));
                }
                let stensor_slice = unsafe {
                    core::slice::from_raw_parts(
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

            /// creates a mutable image from image size and byte slice
            pub fn make_copy_from_make_from_size_and_bytes(
                image_size: ImageSize,
                bytes: &'a [u8],
            ) -> Self {
               Self::try_make_copy_from_size_and_bytes(image_size, bytes).unwrap()
            }

            /// creates a mutable image from unary operator applied to image view
            pub fn from_map<
            'b,
            const OTHER_HRANK: usize,
            const OTHER_SRANK: usize,
            OtherScalar: IsCoreScalar + 'static,
            OtherSTensor: IsStaticTensor<
                OtherScalar,
                OTHER_SRANK,
                OTHER_ROWS,
                OTHER_COLS,
            > + 'static,
            const OTHER_ROWS: usize,
            const OTHER_COLS: usize,
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
            >,
                op: F,

            ) -> Self
              where ndarray::Dim<[ndarray::Ix; OTHER_HRANK]>: ndarray::Dimension,
                TensorView<'b, OTHER_HRANK, 2, OTHER_SRANK, OtherScalar, OtherSTensor,
                           OTHER_ROWS, OTHER_COLS>:
                IsTensorView<'b, OTHER_HRANK, 2, OTHER_SRANK, OtherScalar, OtherSTensor,
                             OTHER_ROWS, OTHER_COLS>,

            {
                Self {
                    mut_tensor: MutTensor::<
                        $scalar_rank,
                        2,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS
                    >::from_map(&v.tensor_view, op),
                }
            }

            /// creates shared image from mutable image
            pub fn to_shared(
                self,
            ) -> GenArcImage<
                $scalar_rank,
                $srank,
                Scalar,
                STensor,
                ROWS,
                COLS
            > {
                GenArcImage {
                    tensor: self.mut_tensor.to_shared(),
                }
            }
        }

        impl<
                'a,
                Scalar: IsCoreScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS> + 'static,
                const ROWS: usize,
                const COLS: usize,

            > IsMutImageView<'a, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS>
            for GenMutImage<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS>
        where
            ndarray::Dim<[ndarray::Ix; $scalar_rank]>: ndarray::Dimension,
        {
            fn mut_image_view<'b: 'a>(
                &'b mut self,
            ) -> crate::mut_image_view::GenMutImageView<
                'a,
                $scalar_rank,
                $srank,
                Scalar,
                STensor,
                ROWS,
                COLS
            > {
                crate::mut_image_view::GenMutImageView {
                    mut_tensor_view: self.mut_tensor.mut_view(),
                }
            }

            fn mut_pixel(&'a mut self, u: usize, v: usize) -> &'a mut STensor {
                self.mut_tensor.get_mut([v, u])
            }
        }
    };
}

mut_image!(2, 0);
mut_image!(3, 1);
mut_image!(4, 2);
