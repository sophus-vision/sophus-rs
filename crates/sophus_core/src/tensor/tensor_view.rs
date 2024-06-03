use crate::linalg::SMat;
use crate::linalg::SVec;
use crate::prelude::*;
use crate::tensor::MutTensor;
use concat_arrays::concat_arrays;
use std::marker::PhantomData;

/// Tensor view
///
/// There are two ways of describing the tensor (TensorView as well as its siblings ArcTensor,
/// MutTensor and MutTensorView):
///
///  1. A dynamic tensor of static tensors:
///    * The dynamic tensor is of rank DRANK
///      - ``self.dims()`` is used to access its dynamic dimensions of type
///        ``[usize: DRANK]``.
///      - an individual element (= static tensor) can be accessed with
///        ``self.get(idx)``, where idx is f type ``[usize: DRANK]``.
///      - Each element is of type ``STensor``.
///    * Each static tensor is of SRANK. In particular we have.
///      - rank 0: scalars of type ``Scalar`` (such as ``u8`` or BatchF64<8>).
///      - rank 1:
///         * A column vector ``SVec<Scalar, ROWS>`` aka ``nalgebra::SVector<Scalar, ROWS>`` with
///           number of ROWS.
///      - rank 2:
///         * A matrix ``SMat<Scalar, ROWS, COLS>`` aka ``nalgebra::SMatrix<Scalar, ROWS, COLS>``
///           with static shape (ROWS x COLS).
///  2. A scalar tensor of TOTAL_RANK = DRANK + SRANK.
///    *  ``self.scalar_dims()`` is used to access its dimensions of type
///        ``[usize: TOTAL_RANK]`` at runtime.
///    *  an individual element (= static tensor) can be accessed with
///       ``self.scalar_get(idx)``, where idx is of type ``[usize: DRANK]``.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TensorView<
    'a,
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar: IsCoreScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS> + 'static,
    const ROWS: usize,
    const COLS: usize,
> where
    ndarray::Dim<[ndarray::Ix; DRANK]>: ndarray::Dimension,
    ndarray::Dim<[ndarray::Ix; TOTAL_RANK]>: ndarray::Dimension,
{
    /// Element view - an ndarray of static tensors with shape [D0, D1, ...]
    pub elem_view: ndarray::ArrayView<'a, STensor, ndarray::Dim<[ndarray::Ix; DRANK]>>,
    /// Scalar view - an ndarray of scalars with shape [D0, D1, ..., S0, S1, ...]
    pub scalar_view: ndarray::ArrayView<'a, Scalar, ndarray::Dim<[ndarray::Ix; TOTAL_RANK]>>,
}

/// Tensor view of scalars
pub type TensorViewX<'a, const DRANK: usize, Scalar> =
    TensorView<'a, DRANK, DRANK, 0, Scalar, Scalar, 1, 1>;

/// Tensor view of vectors with shape R
pub type TensorViewXR<
    'a,
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
> = TensorView<'a, TOTAL_RANK, DRANK, SRANK, Scalar, SVec<Scalar, R>, R, 1>;

/// Tensor view of matrices with shape [R x C]
pub type TensorViewXRC<
    'a,
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
    const C: usize,
> = TensorView<'a, TOTAL_RANK, DRANK, SRANK, Scalar, SMat<Scalar, R, C>, R, C>;

/// rank-1 tensor view of scalars with shape D0
pub type TensorViewD<'a, Scalar> = TensorViewX<'a, 1, Scalar>;

/// rank-2 tensor view of scalars with shape [D0 x D1]
pub type TensorViewDD<'a, Scalar> = TensorViewX<'a, 2, Scalar>;

/// rank-2 tensor view of vectors with shape [D0 x R]
pub type TensorViewDR<'a, Scalar, const R: usize> = TensorViewXR<'a, 2, 1, 1, Scalar, R>;

/// rank-3 tensor view of scalars with shape [D0 x R x B]
pub type TensorViewDDD<'a, Scalar> = TensorViewX<'a, 3, Scalar>;

/// rank-3 tensor view of vectors with shape [D0 x D1 x R]
pub type TensorViewDDR<'a, Scalar, const R: usize> = TensorViewXR<'a, 3, 2, 1, Scalar, R>;

/// rank-3 tensor view of matrices with shape [D0 x R x C]
pub type TensorViewDRC<'a, Scalar, const R: usize, const C: usize> =
    TensorViewXRC<'a, 3, 1, 2, Scalar, R, C>;

/// rank-4 tensor view of scalars with shape [D0 x D1 x D2 x D3]
pub type TensorViewDDDD<'a, Scalar> = TensorViewX<'a, 4, Scalar>;

/// rank-4 tensor view of vectors with shape [D0 x D1 x D2 x R]
pub type TensorViewDDDR<'a, Scalar, const R: usize> = TensorViewXR<'a, 4, 3, 1, Scalar, R>;

/// rank-4 tensor view of matrices with shape [D0 x R x C x B]
pub type TensorViewDDRC<'a, Scalar, const R: usize, const C: usize> =
    TensorViewXRC<'a, 4, 2, 2, Scalar, R, C>;

/// Is a tensor-like object
pub trait IsTensorLike<
    'a,
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar: IsCoreScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS> + 'static,
    const ROWS: usize,
    const COLS: usize,
> where
    ndarray::Dim<[ndarray::Ix; DRANK]>: ndarray::Dimension,
    ndarray::Dim<[ndarray::Ix; TOTAL_RANK]>: ndarray::Dimension,
{
    /// Element view - that is a tensor view of static tensors
    fn elem_view<'b: 'a>(
        &'b self,
    ) -> ndarray::ArrayView<'a, STensor, ndarray::Dim<[ndarray::Ix; DRANK]>>;

    /// Get the element at index idx
    fn get(&self, idx: [usize; DRANK]) -> STensor;

    /// Get the dimensions of the tensor [D0, D1, ...]
    fn dims(&self) -> [usize; DRANK];

    /// Scalar view - that is a tensor view of scalars
    fn scalar_view<'b: 'a>(
        &'b self,
    ) -> ndarray::ArrayView<'a, Scalar, ndarray::Dim<[ndarray::Ix; TOTAL_RANK]>>;

    /// Get the scalar at index idx
    fn scalar_get(&'a self, idx: [usize; TOTAL_RANK]) -> Scalar;

    /// Get the dimensions of the scalar view [D0, D1, ..., S0, S1, ...]
    fn scalar_dims(&self) -> [usize; TOTAL_RANK];

    /// Convert to a mutable tensor - this will copy the tensor
    fn to_mut_tensor(&self) -> MutTensor<TOTAL_RANK, DRANK, SRANK, Scalar, STensor, ROWS, COLS>;
}

/// Is a tensor view like object
pub trait IsTensorView<
    'a,
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar: IsCoreScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS> + 'static,
    const ROWS: usize,
    const COLS: usize,
>: IsTensorLike<'a, TOTAL_RANK, DRANK, SRANK, Scalar, STensor, ROWS, COLS> where
    ndarray::Dim<[ndarray::Ix; DRANK]>: ndarray::Dimension,
    ndarray::Dim<[ndarray::Ix; TOTAL_RANK]>: ndarray::Dimension,
{
    /// return tensor view
    fn view<'b: 'a>(&'b self) -> Self;
}

macro_rules! tensor_view_is_view {
    ($scalar_rank:literal, $srank:literal, $drank:literal) => {
        impl<
                'a,
                Scalar: IsCoreScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS>,
                const ROWS: usize,
                const COLS: usize,
            > TensorView<'a, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS>
        {
            /// Create a new tensor view from an ndarray of static tensors
            pub fn new(
                elem_view: ndarray::ArrayView<'a, STensor, ndarray::Dim<[ndarray::Ix; $drank]>>,
            ) -> Self {
                let dims: [usize; $drank] = elem_view.shape().try_into().unwrap();
                #[allow(clippy::drop_non_drop)]
                let shape: [usize; $scalar_rank] = concat_arrays!(dims, STensor::sdims());

                let dstrides: [isize; $drank] = elem_view.strides().try_into().unwrap();
                let mut dstrides: [usize; $drank] = dstrides.map(|x| x as usize);
                let num_scalars = STensor::num_scalars();
                for d in dstrides.iter_mut() {
                    *d *= num_scalars;
                }
                #[allow(clippy::drop_non_drop)]
                let strides = concat_arrays!(dstrides, STensor::strides());

                let ptr = elem_view.as_ptr() as *const Scalar;
                use ndarray::ShapeBuilder;

                assert_eq!(
                    std::mem::size_of::<STensor>(),
                    std::mem::size_of::<Scalar>() * ROWS * COLS
                );
                let scalar_view =
                    unsafe { ndarray::ArrayView::from_shape_ptr(shape.strides(strides), ptr) };

                Self {
                    elem_view,
                    scalar_view,
                }
            }

            /// Create a new tensor view from a slice of static tensors
            pub fn from_shape_and_slice(shape: [usize; $drank], slice: &'a [STensor]) -> Self {
                let elem_view = ndarray::ArrayView::from_shape(shape, slice).unwrap();
                Self::new(elem_view)
            }
        }

        impl<
                'a,
                Scalar: IsCoreScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS> + 'static,
                const ROWS: usize,
                const COLS: usize,
            > IsTensorLike<'a, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS>
            for TensorView<'a, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS>
        {
            fn elem_view<'b: 'a>(
                &'b self,
            ) -> ndarray::ArrayView<'a, STensor, ndarray::Dim<[ndarray::Ix; $drank]>> {
                self.elem_view
            }

            fn get(&self, idx: [usize; $drank]) -> STensor {
                self.elem_view[idx].clone()
            }

            fn dims(&self) -> [usize; $drank] {
                self.elem_view.shape().try_into().unwrap()
            }

            fn scalar_view<'b: 'a>(
                &'b self,
            ) -> ndarray::ArrayView<'a, Scalar, ndarray::Dim<[ndarray::Ix; $scalar_rank]>> {
                self.scalar_view
            }

            fn scalar_get(&'a self, idx: [usize; $scalar_rank]) -> Scalar {
                self.scalar_view[idx].clone()
            }

            fn scalar_dims(&self) -> [usize; $scalar_rank] {
                self.scalar_view.shape().try_into().unwrap()
            }

            fn to_mut_tensor(
                &self,
            ) -> MutTensor<$scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS> {
                MutTensor {
                    mut_array: self.elem_view.to_owned(),
                    phantom: PhantomData::default(),
                }
            }
        }

        impl<
                'a,
                Scalar: IsCoreScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS> + 'static,
                const ROWS: usize,
                const COLS: usize,
            > IsTensorView<'a, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS>
            for TensorView<'a, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS>
        {
            fn view<'b: 'a>(
                &'b self,
            ) -> TensorView<'a, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS> {
                self.clone()
            }
        }

        impl<
                'a,
                Scalar: IsCoreScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS> + 'static,
                const ROWS: usize,
                const COLS: usize,
            > TensorView<'a, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS>
        {
        }
    };
}

tensor_view_is_view!(1, 0, 1);
tensor_view_is_view!(2, 0, 2);
tensor_view_is_view!(2, 1, 1);
tensor_view_is_view!(3, 0, 3);
tensor_view_is_view!(3, 1, 2);
tensor_view_is_view!(3, 2, 1);
tensor_view_is_view!(4, 0, 4);
tensor_view_is_view!(4, 1, 3);
tensor_view_is_view!(4, 2, 2);
tensor_view_is_view!(5, 0, 5);
tensor_view_is_view!(5, 1, 4);
tensor_view_is_view!(5, 2, 3);

#[test]
fn tensor_view_tests() {
    use ndarray::ShapeBuilder;
    {
        let rank1_shape = [3];
        let arr: [u8; 3] = [5, 6, 7];

        let ndview = ndarray::ArrayView::from_shape(rank1_shape.strides([1]), &arr[..]).unwrap();
        assert!(ndview.is_standard_layout());
        let view = TensorViewD::new(ndview);

        for i in 0..view.dims()[0] {
            assert_eq!(arr[i], view.get([i]));
        }
    }
    {
        const ROWS: usize = 2;
        const COLS: usize = 3;

        type Mat2x3 = SMat<f32, 2, 3>;

        let a = Mat2x3::new(0.1, 0.56, 0.77, 2.0, 5.1, 7.0);
        let b = Mat2x3::new(0.6, 0.5, 0.78, 2.0, 5.2, 7.1);
        let c = Mat2x3::new(0.9, 0.58, 0.7, 2.0, 5.3, 7.2);
        let d = Mat2x3::new(0.9, 0.50, 0.9, 2.0, 5.0, 7.3);

        let rank2_shape = [4, 2];
        let arr = [a, a, b, c, d, c, b, b];

        let strides = [2, 1];
        let ndview = ndarray::ArrayView::from_shape(rank2_shape.strides([2, 1]), &arr[..]).unwrap();
        assert!(ndview.is_standard_layout());
        let view = TensorViewDDRC::new(ndview);

        println!("{}", view.elem_view);
        for d0 in 0..view.dims()[0] {
            for d1 in 0..view.dims()[1] {
                assert_eq!(view.get([d0, d1]), arr[strides[0] * d0 + strides[1] * d1]);
            }
        }

        println!("{:?}", view.scalar_view);
        assert!(!view.scalar_view.is_standard_layout());
        for d0 in 0..view.scalar_dims()[0] {
            for d1 in 0..view.scalar_dims()[1] {
                for c in 0..COLS {
                    for r in 0..ROWS {
                        assert_eq!(
                            view.scalar_get([d0, d1, r, c]),
                            arr[strides[0] * d0 + strides[1] * d1][c * ROWS + r]
                        );
                    }
                }
            }
        }
    }

    {
        let rank3_shape = [4, 2, 3];
        let raw_arr = [
            4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28,
        ];

        let arr = raw_arr.map(SVec::<u8, 1>::new);

        let strides = [6, 3, 1];
        let ndview =
            ndarray::ArrayView::from_shape(rank3_shape.strides(strides), &arr[..]).unwrap();
        assert!(ndview.is_standard_layout());
        let view = TensorViewDDDR::new(ndview);

        println!("{}", view.elem_view);
        for d0 in 0..view.dims()[0] {
            for d1 in 0..view.dims()[1] {
                for d2 in 0..view.dims()[2] {
                    assert_eq!(
                        view.get([d0, d1, d2]),
                        arr[strides[0] * d0 + strides[1] * d1 + strides[2] * d2]
                    );
                }
            }
        }

        println!("{:?}", view.scalar_view);
        for d0 in 0..view.scalar_dims()[0] {
            for d1 in 0..view.scalar_dims()[1] {
                for d2 in 0..view.scalar_dims()[2] {
                    for r in 0..1 {
                        assert_eq!(
                            view.scalar_get([d0, d1, d2, r]),
                            arr[strides[0] * d0 + strides[1] * d1 + strides[2] * d2][r]
                        );
                    }
                }
            }
        }
    }
}
