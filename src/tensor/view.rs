use super::element::BatchMat;
use super::element::BatchScalar;
use super::element::BatchVec;
use super::element::IsStaticTensor;
use super::element::IsTensorScalar;
use super::element::SMat;
use super::element::SVec;
use super::mut_tensor::MutTensor;
use concat_arrays::concat_arrays;
use std::marker::PhantomData;

/// See MutTensor
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TensorView<
    'a,
    const HYPER_SHAPE: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar: IsTensorScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS, BATCHES> + 'static,
    const ROWS: usize,
    const COLS: usize,
    const BATCHES: usize,
> where
    ndarray::Dim<[ndarray::Ix; DRANK]>: ndarray::Dimension,
    ndarray::Dim<[ndarray::Ix; HYPER_SHAPE]>: ndarray::Dimension,
{
    pub elem_view: ndarray::ArrayView<'a, STensor, ndarray::Dim<[ndarray::Ix; DRANK]>>,
    pub scalar_view: ndarray::ArrayView<'a, Scalar, ndarray::Dim<[ndarray::Ix; HYPER_SHAPE]>>,
}

pub type TensorViewX<'a, const DRANK: usize, Scalar> =
    TensorView<'a, DRANK, DRANK, 0, Scalar, Scalar, 1, 1, 1>;

pub type TensorViewXB<
    'a,
    const HYPER_SHAPE: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const B: usize,
> = TensorView<'a, HYPER_SHAPE, DRANK, SRANK, Scalar, BatchScalar<Scalar, B>, 1, 1, B>;

pub type TensorViewXR<
    'a,
    const HYPER_SHAPE: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
> = TensorView<'a, HYPER_SHAPE, DRANK, SRANK, Scalar, SVec<Scalar, R>, R, 1, 1>;

pub type TensorViewXRB<
    'a,
    const HYPER_SHAPE: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
    const B: usize,
> = TensorView<'a, HYPER_SHAPE, DRANK, SRANK, Scalar, BatchVec<Scalar, R, B>, R, 1, B>;

pub type TensorViewXRC<
    'a,
    const HYPER_SHAPE: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
    const C: usize,
> = TensorView<'a, HYPER_SHAPE, DRANK, SRANK, Scalar, SMat<Scalar, R, C>, R, C, 1>;

pub type TensorViewXRCB<
    'a,
    const HYPER_SHAPE: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
    const C: usize,
    const B: usize,
> = TensorView<'a, HYPER_SHAPE, DRANK, SRANK, Scalar, BatchMat<Scalar, R, C, B>, R, C, B>;

// rank 1 - D0
pub type TensorViewD<'a, Scalar> = TensorViewX<'a, 1, Scalar>;

// rank 2
// D0 x D1
pub type TensorViewDD<'a, Scalar> = TensorViewX<'a, 2, Scalar>;

// rank 2
// D0 x [B]
pub type TensorViewDB<'a, Scalar, const B: usize> = TensorViewXB<'a, 2, 1, 1, Scalar, B>;

// rank 2
// D0 x [R]
pub type TensorViewDR<'a, Scalar, const R: usize> = TensorViewXR<'a, 2, 1, 1, Scalar, R>;

// rank 3
// D0 x D1 x D2
pub type TensorViewRRR<'a, Scalar> = TensorViewX<'a, 3, Scalar>;

// rank 3
// D0 x D1 x [B]
pub type TensorViewDDB<'a, Scalar, const B: usize> = TensorViewXB<'a, 3, 2, 1, Scalar, B>;

// rank 3
// D0 x D1 x [R]
pub type TensorViewDDR<'a, Scalar, const R: usize> = TensorViewXR<'a, 3, 2, 1, Scalar, R>;

// rank 3
// D0 x [R x B]
pub type TensorViewDRB<'a, Scalar, const R: usize, const B: usize> =
    TensorViewXRB<'a, 3, 1, 2, Scalar, R, B>;

// rank 3
// D0 x [R x C]
pub type TensorViewDRC<'a, Scalar, const R: usize, const C: usize> =
    TensorViewXRC<'a, 3, 1, 2, Scalar, R, C>;

// rank 4
// srank 0, drank 4
pub type TensorViewDDDD<'a, Scalar> = TensorViewX<'a, 4, Scalar>;

// rank 4
// D0 x D1 x D2 x [B]
pub type TensorViewDDDB<'a, Scalar, const B: usize> = TensorViewXB<'a, 4, 3, 1, Scalar, B>;

// rank 4
// D0 x D1 x D2 x [R]
pub type TensorViewDDDR<'a, Scalar, const R: usize> = TensorViewXR<'a, 4, 3, 1, Scalar, R>;

// rank 4
// D0 x D1 x [R x B]
pub type TensorViewDDRB<'a, Scalar, const R: usize, const B: usize> =
    TensorViewXRB<'a, 4, 2, 2, Scalar, R, B>;

// rank 4
// D0 x D1 x [R x C]
pub type TensorViewDDRC<'a, Scalar, const R: usize, const C: usize> =
    TensorViewXRC<'a, 4, 2, 2, Scalar, R, C>;

// rank 4
// D0 x [R x C x B]
pub type TensorViewDRCB<'a, Scalar, const R: usize, const C: usize, const B: usize> =
    TensorViewXRCB<'a, 4, 1, 3, Scalar, R, C, B>;

pub trait IsTensorLike<
    'a,
    const SCALAR_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar: IsTensorScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS, BATCHES> + 'static,
    const ROWS: usize,
    const COLS: usize,
    const BATCHES: usize,
> where
    ndarray::Dim<[ndarray::Ix; DRANK]>: ndarray::Dimension,
    ndarray::Dim<[ndarray::Ix; SCALAR_RANK]>: ndarray::Dimension,
{
    fn elem_view<'b: 'a>(
        &'b self,
    ) -> ndarray::ArrayView<'a, STensor, ndarray::Dim<[ndarray::Ix; DRANK]>>;

    fn get(&self, idx: [usize; DRANK]) -> STensor;

    fn dims(&self) -> [usize; DRANK];

    fn scalar_view<'b: 'a>(
        &'b self,
    ) -> ndarray::ArrayView<'a, Scalar, ndarray::Dim<[ndarray::Ix; SCALAR_RANK]>>;

    fn scalar_get(&'a self, idx: [usize; SCALAR_RANK]) -> Scalar;

    fn scalar_dims(&self) -> [usize; SCALAR_RANK];

    fn to_mut_image(
        &self,
    ) -> MutTensor<SCALAR_RANK, DRANK, SRANK, Scalar, STensor, ROWS, COLS, BATCHES>;
}

pub trait IsTensorView<
    'a,
    const SCALAR_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar: IsTensorScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS, BATCHES> + 'static,
    const ROWS: usize,
    const COLS: usize,
    const BATCHES: usize,
>: IsTensorLike<'a, SCALAR_RANK, DRANK, SRANK, Scalar, STensor, ROWS, COLS, BATCHES> where
    ndarray::Dim<[ndarray::Ix; DRANK]>: ndarray::Dimension,
    ndarray::Dim<[ndarray::Ix; SCALAR_RANK]>: ndarray::Dimension,
{
    fn view<'b: 'a>(&'b self) -> Self;
}

macro_rules! tensor_view_is_view {
    ($scalar_rank:literal, $srank:literal, $drank:literal) => {
        impl<
                'a,
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS, BATCHES>,
                const ROWS: usize,
                const COLS: usize,
                const BATCHES: usize,
            > TensorView<'a, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
        {
            pub fn new(
                elem_view: ndarray::ArrayView<'a, STensor, ndarray::Dim<[ndarray::Ix; $drank]>>,
            ) -> Self {
                let dims: [usize; $drank] = elem_view.shape().try_into().unwrap();
                let shape: [usize; $scalar_rank] = concat_arrays!(dims, STensor::sdims());

                let dstrides: [isize; $drank] = elem_view.strides().try_into().unwrap();
                let mut dstrides: [usize; $drank] = dstrides.map(|x| x as usize);
                let num_scalars = STensor::num_scalars();
                for d in dstrides.iter_mut() {
                    *d *= num_scalars;
                }
                let strides = concat_arrays!(dstrides, STensor::strides());

                let ptr = elem_view.as_ptr() as *const Scalar;
                use ndarray::ShapeBuilder;

                assert_eq!(
                    std::mem::size_of::<STensor>(),
                    std::mem::size_of::<Scalar>() * ROWS * COLS * BATCHES
                );
                let scalar_view =
                    unsafe { ndarray::ArrayView::from_shape_ptr(shape.strides(strides), ptr) };

                Self {
                    elem_view,
                    scalar_view,
                }
            }

            pub fn from_shape_and_slice(shape: [usize; $drank], slice: &'a [STensor]) -> Self {
                let elem_view = ndarray::ArrayView::from_shape(shape, slice).unwrap();
                Self::new(elem_view)
            }
        }

        impl<
                'a,
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS, BATCHES> + 'static,
                const ROWS: usize,
                const COLS: usize,
                const BATCHES: usize,
            > IsTensorLike<'a, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
            for TensorView<'a, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
        {
            fn elem_view<'b: 'a>(
                &'b self,
            ) -> ndarray::ArrayView<'a, STensor, ndarray::Dim<[ndarray::Ix; $drank]>> {
                self.elem_view
            }

            fn get(&self, idx: [usize; $drank]) -> STensor {
                self.elem_view[idx]
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
                self.scalar_view[idx]
            }

            fn scalar_dims(&self) -> [usize; $scalar_rank] {
                self.scalar_view.shape().try_into().unwrap()
            }

            fn to_mut_image(
                &self,
            ) -> MutTensor<$scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS, BATCHES> {
                MutTensor {
                    mut_array: self.elem_view.to_owned(),
                    phantom: PhantomData::default(),
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
            > IsTensorView<'a, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
            for TensorView<'a, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
        {
            fn view<'b: 'a>(
                &'b self,
            ) -> TensorView<'a, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
            {
                *self
            }
        }

        impl<
                'a,
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS, BATCHES> + 'static,
                const BATCHES: usize,
                const ROWS: usize,
                const COLS: usize,
            > TensorView<'a, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
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
tensor_view_is_view!(4, 3, 1);

#[cfg(test)]
mod tests {

    #[test]
    fn view() {
        use super::*;
        use crate::tensor::element::P1U8;
        use ndarray::ShapeBuilder;
        {
            let rank1_shape = [3];
            let arr: [u8; 3] = [5, 6, 7];

            let ndview =
                ndarray::ArrayView::from_shape(rank1_shape.strides([1]), &arr[..]).unwrap();
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
            let ndview =
                ndarray::ArrayView::from_shape(rank2_shape.strides([2, 1]), &arr[..]).unwrap();
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
                4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                27, 28,
            ];

            let arr = raw_arr.map(P1U8::new);

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
}
