use sophus_tensor::{
    ArcTensor,
    TensorView,
};

use crate::{
    ImageSize,
    image_view::GenImageView,
    mut_image::GenMutImage,
    prelude::*,
};

/// Image of static tensors with shared ownership
#[derive(Debug, Clone)]
pub struct GenArcImage<
    const TOTAL_RANK: usize,
    const SRANK: usize,
    Scalar: IsCoreScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS> + 'static,
    const ROWS: usize,
    const COLS: usize,
> {
    /// underlying tensor
    pub tensor: ArcTensor<TOTAL_RANK, 2, SRANK, Scalar, STensor, ROWS, COLS>,
}

macro_rules! arc_image {
    ($scalar_rank:literal, $srank:literal) => {
        /// Convert a GenMutImage to an GenArcImage
        ///
        /// If an image is created inside a function and needs to be returned, it is best practice to
        /// return a GenMutImage. The GenMutImage can be converted to an GenArcImage cheaply and efficiently
        /// using ``.into()`` generated by this trait.
        ///
        impl<
                'a,
                Scalar: IsCoreScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS> + 'static,
                const ROWS: usize,
                const COLS: usize,
            > From<GenMutImage<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS>>
            for GenArcImage<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS>
            where
            ndarray::Dim<[ndarray::Ix; $scalar_rank]>: ndarray::Dimension,
        {
            fn from(value: GenMutImage<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS>) -> Self {
                Self::from_mut_image(value)
            }
        }



        impl<
        'a,
                Scalar: IsCoreScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS> + 'static,
                const ROWS: usize,
                const COLS: usize,
            > GenArcImage<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS>
        {
            /// Convert an GenArcImage to a GenMutImage
            ///
            /// It is best practice to not call this function directly. Instead, use the ``.into()``
            /// method generated by the ``From`` trait.
            pub fn from_mut_image(
                image: GenMutImage<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS>,
            ) -> Self {
                Self {
                    tensor: ArcTensor::<
                        $scalar_rank,
                        2,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS,
                    >::from_mut_tensor(image.mut_tensor),
                }
            }

            /// create a new image from image size and a value
            pub fn from_image_size_and_val(size: ImageSize, val: STensor) -> Self {
                Self {
                    tensor: ArcTensor::<
                        $scalar_rank,
                        2,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS,
                    >::from_shape_and_val(size.into(), val),
                }
            }

            /// create a new image from an image view
            pub fn make_copy_from(
                v: &GenImageView<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS>,
            ) -> Self {
                Self {
                    tensor: ArcTensor::<
                        $scalar_rank,
                        2,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS,
                    >::make_copy_from(&v.tensor_view),
                }
            }

            /// create a new image from image size and a slice
            pub fn make_copy_from_size_and_slice(image_size: ImageSize, slice: &'a [STensor]) -> Self {
                GenMutImage::<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS>
                    ::make_copy_from_size_and_slice(image_size, slice).into()
            }

            /// create a new image from a uniform operator applied to an image view
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
            where    ndarray::Dim<[ndarray::Ix; OTHER_HRANK]>: ndarray::Dimension,
            TensorView<'b, OTHER_HRANK, 2, OTHER_SRANK, OtherScalar, OtherSTensor, OTHER_ROWS, OTHER_COLS>:
              IsTensorView<'b, OTHER_HRANK, 2, OTHER_SRANK, OtherScalar, OtherSTensor, OTHER_ROWS, OTHER_COLS>,

            {
                Self {
                    tensor: ArcTensor::<
                        $scalar_rank,
                        2,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS,
                    >::from_map(&v.tensor_view, op),
                }
            }

            /// create a new image from fn
            pub fn from_fn<F: FnMut([usize; 2]) -> STensor>(
                image_size: ImageSize,
                op: F) -> Self {
                Self {
                    tensor: ArcTensor::<
                        $scalar_rank,
                        2,
                        $srank,
                        Scalar,
                        STensor,
                        ROWS,
                        COLS,
                    >::from_fn(image_size.into(), op),
                }
            }
        }

        /// creates an image from a binary operator applied to two image views
        impl<
                'b,
                Scalar: IsCoreScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS> + 'static,
                const ROWS: usize,
                const COLS: usize,

            > Default for GenArcImage<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS>
        where
            ndarray::Dim<[ndarray::Ix; $scalar_rank]>: ndarray::Dimension,
        {
            fn default() -> Self {
                Self::from_image_size_and_val(ImageSize::default(), num_traits::Zero::zero())
            }
        }

        impl<
                'b,
                Scalar: IsCoreScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS> + 'static,
                const ROWS: usize,
                const COLS: usize,

            > IsImageView<'b, $scalar_rank, $srank, Scalar, STensor, ROWS, COLS>
            for GenArcImage<$scalar_rank, $srank, Scalar, STensor, ROWS, COLS>
        where
            ndarray::Dim<[ndarray::Ix; $scalar_rank]>: ndarray::Dimension,
        {
            fn image_view(
                &'b self,
            ) -> super::image_view::GenImageView<
                'b,
                $scalar_rank,
                $srank,
                Scalar,
                STensor,
                ROWS,
                COLS,
            > {
                GenImageView {
                    tensor_view: self.tensor.view(),
                }
            }



            fn pixel(&'b self, u: usize, v: usize) -> STensor {
                self.tensor.get([v, u])
            }

            fn image_size(&self) -> ImageSize {
                self.image_view().image_size()
            }
        }

    };
}

arc_image!(2, 0);
arc_image!(3, 1);
arc_image!(4, 2);
