use ndarray::Dimension;

use crate::element::BatchMat;
use crate::element::BatchScalar;
use crate::element::BatchVec;
use crate::element::IsStaticTensor;
use crate::element::IsTensorScalar;
use crate::element::SMat;
use crate::element::SVec;
use crate::mut_tensor::InnerScalarToVec;
use crate::mut_tensor::InnerVecToMat;
use crate::mut_tensor::MutTensor;
use crate::view::IsTensorLike;
use crate::view::IsTensorView;
use crate::view::TensorView;

use std::marker::PhantomData;

/// Arc tensor - a tensor with shared ownership
///
/// See TensorView for more details of the tensor structure
#[derive(Debug, Clone)]
pub struct ArcTensor<
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar: IsTensorScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS, BATCH_SIZE> + 'static,
    const ROWS: usize,
    const COLS: usize,
    const BATCH_SIZE: usize,
> where
    ndarray::Dim<[ndarray::Ix; DRANK]>: Dimension,
{
    /// ndarray of tensors with shape [D1, D2, ...]
    pub array: ndarray::ArcArray<STensor, ndarray::Dim<[ndarray::Ix; DRANK]>>,
    phantom: PhantomData<(Scalar, STensor)>,
}

// /// Tensor view of scalars
// pub type TensorViewX<'a, const DRANK: usize, Scalar> =
//     TensorView<'a, DRANK, DRANK, 0, Scalar, Scalar, 1, 1, 1>;

// /// Tensor view of batched scalars
// pub type TensorViewXB<
//     'a,
//     const TOTAL_RANK: usize,
//     const DRANK: usize,
//     const SRANK: usize,
//     Scalar,
//     const B: usize,
// > = TensorView<'a, TOTAL_RANK, DRANK, SRANK, Scalar, BatchScalar<Scalar, B>, 1, 1, B>;

// /// Tensor view of vectors with shape [R]
// pub type TensorViewXR<
//     'a,
//     const TOTAL_RANK: usize,
//     const DRANK: usize,
//     const SRANK: usize,
//     Scalar,
//     const R: usize,
// > = TensorView<'a, TOTAL_RANK, DRANK, SRANK, Scalar, SVec<Scalar, R>, R, 1, 1>;

// /// Tensor view of batched vectors with shape [R x B]
// pub type TensorViewXRB<
//     'a,
//     const TOTAL_RANK: usize,
//     const DRANK: usize,
//     const SRANK: usize,
//     Scalar,
//     const R: usize,
//     const B: usize,
// > = TensorView<'a, TOTAL_RANK, DRANK, SRANK, Scalar, BatchVec<Scalar, R, B>, R, 1, B>;

// /// Tensor view of matrices with shape [R x C]
// pub type TensorViewXRC<
//     'a,
//     const TOTAL_RANK: usize,
//     const DRANK: usize,
//     const SRANK: usize,
//     Scalar,
//     const R: usize,
//     const C: usize,
// > = TensorView<'a, TOTAL_RANK, DRANK, SRANK, Scalar, SMat<Scalar, R, C>, R, C, 1>;

// /// Tensor view of batched matrices with shape [R x C x B]
// pub type TensorViewXRCB<
//     'a,
//     const TOTAL_RANK: usize,
//     const DRANK: usize,
//     const SRANK: usize,
//     Scalar,
//     const R: usize,
//     const C: usize,
//     const B: usize,
// > = TensorView<'a, TOTAL_RANK, DRANK, SRANK, Scalar, BatchMat<Scalar, R, C, B>, R, C, B>;

// /// rank-1 tensor view of scalars with shape [D0]
// pub type TensorViewD<'a, Scalar> = TensorViewX<'a, 1, Scalar>;

// /// rank-2 tensor view of scalars with shape [D0 x D1]
// pub type TensorViewDD<'a, Scalar> = TensorViewX<'a, 2, Scalar>;

// /// rank-2 tensor view of batched scalars with shape [D0 x B]
// pub type TensorViewDB<'a, Scalar, const B: usize> = TensorViewXB<'a, 2, 1, 1, Scalar, B>;

// /// rank-2 tensor view of vectors with shape [D0 x R]
// pub type TensorViewDR<'a, Scalar, const R: usize> = TensorViewXR<'a, 2, 1, 1, Scalar, R>;

// /// rank-3 tensor view of scalars with shape [D0 x R x B]
// pub type TensorViewDDD<'a, Scalar> = TensorViewX<'a, 3, Scalar>;

// /// rank-3 tensor view of batched scalars with shape [D0 x D1 x B]
// pub type TensorViewDDB<'a, Scalar, const B: usize> = TensorViewXB<'a, 3, 2, 1, Scalar, B>;

// /// rank-3 tensor view of vectors with shape [D0 x D1 x R]
// pub type TensorViewDDR<'a, Scalar, const R: usize> = TensorViewXR<'a, 3, 2, 1, Scalar, R>;

// /// rank-3 tensor view of batched vectors with shape [D0 x R x B]
// pub type TensorViewDRB<'a, Scalar, const R: usize, const B: usize> =
//     TensorViewXRB<'a, 3, 1, 2, Scalar, R, B>;

// /// rank-3 tensor view of matrices with shape [D0 x R x C]
// pub type TensorViewDRC<'a, Scalar, const R: usize, const C: usize> =
//     TensorViewXRC<'a, 3, 1, 2, Scalar, R, C>;

// /// rank-4 tensor view of scalars with shape [D0 x D1 x D2 x D3]
// pub type TensorViewDDDD<'a, Scalar> = TensorViewX<'a, 4, Scalar>;

// /// rank-4 tensor view of batched scalars with shape [D0 x D1 x D2 x B]
// pub type TensorViewDDDB<'a, Scalar, const B: usize> = TensorViewXB<'a, 4, 3, 1, Scalar, B>;

// /// rank-4 tensor view of vectors with shape [D0 x D1 x D2 x R]
// pub type TensorViewDDDR<'a, Scalar, const R: usize> = TensorViewXR<'a, 4, 3, 1, Scalar, R>;

// /// rank-4 tensor view of batched vectors with shape [D0 x D1 x R x B]
// pub type TensorViewDDRB<'a, Scalar, const R: usize, const B: usize> =
//     TensorViewXRB<'a, 4, 2, 2, Scalar, R, B>;

// /// rank-4 tensor view of matrices with shape [D0 x R x C x B]
// pub type TensorViewDDRC<'a, Scalar, const R: usize, const C: usize> =
//     TensorViewXRC<'a, 4, 2, 2, Scalar, R, C>;

// /// rank-4 tensor view of batched matrices with shape [D0 x R x C x B]
// pub type TensorViewDRCB<'a, Scalar, const R: usize, const C: usize, const B: usize> =
//     TensorViewXRCB<'a, 4, 1, 3, Scalar, R, C, B>;

/// rank-1 tensor of scalars
pub type ArcTensorX<const DRANK: usize, Scalar> =
    ArcTensor<DRANK, DRANK, 0, Scalar, Scalar, 1, 1, 1>;

/// rank-2 tensor of batched scalars
pub type ArcTensorXB<
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const B: usize,
> = ArcTensor<TOTAL_RANK, DRANK, SRANK, Scalar, BatchScalar<Scalar, B>, 1, 1, B>;

/// rank-2 tensor of vectors with shape R
pub type ArcTensorXR<
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
> = ArcTensor<TOTAL_RANK, DRANK, SRANK, Scalar, SVec<Scalar, R>, R, 1, 1>;

/// rank-2 tensor of batched vectors with shape [R x B]
pub type ArcTensorXRB<
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
    const B: usize,
> = ArcTensor<TOTAL_RANK, DRANK, SRANK, Scalar, BatchVec<Scalar, R, B>, R, 1, B>;

/// rank-2 tensor of matrices with shape [R x C]
pub type ArcTensorXRC<
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
    const C: usize,
> = ArcTensor<TOTAL_RANK, DRANK, SRANK, Scalar, SMat<Scalar, R, C>, R, C, 1>;

/// rank-2 tensor of batched matrices with shape [R x C x B]
pub type ArcTensorXRCB<
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
    const C: usize,
    const B: usize,
> = ArcTensor<TOTAL_RANK, DRANK, SRANK, Scalar, BatchMat<Scalar, R, C, B>, R, C, B>;

/// rank-1 tensor of scalars with shape D0
pub type ArcTensorD<const DRANK: usize, Scalar> = ArcTensorX<DRANK, Scalar>;

/// rank-2 tensor of scalars with shape [D0 x D1]
pub type ArcTensorDD<Scalar> = ArcTensorX<2, Scalar>;

/// rank-2 tensor of batched scalars with shape [D0 x B]
pub type ArcTensorDB<Scalar, const B: usize> = ArcTensorXB<2, 1, 1, Scalar, B>;

/// rank-2 tensor of vectors with shape [D0 x R]
pub type ArcTensorDR<Scalar, const R: usize> = ArcTensorXR<2, 1, 1, Scalar, R>;

/// rank-3 tensor of scalars with shape [D0 x D1 x D2]
pub type ArcTensorRRR<Scalar> = ArcTensorX<3, Scalar>;

/// rank-3 tensor of batched scalars with shape [D0 x D1 x B]
pub type ArcTensorDDB<Scalar, const B: usize> = ArcTensorXB<3, 2, 1, Scalar, B>;

/// rank-3 tensor of vectors with shape [D0 x D1 x R]
pub type ArcTensorDDR<Scalar, const R: usize> = ArcTensorXR<3, 2, 1, Scalar, R>;

/// rank-3 tensor of batched vectors with shape [D0 x R x B]
pub type ArcTensorRBD<Scalar, const R: usize, const B: usize> = ArcTensorXRB<3, 1, 2, Scalar, R, B>;

/// rank-3 tensor of matrices with shape [D0 x R x C]
pub type ArcTensorDRC<Scalar, const R: usize, const C: usize> = ArcTensorXRC<3, 1, 2, Scalar, R, C>;

/// rank-4 tensor of scalars with shape [D0 x D1 x D2 x D3]
pub type ArcTensorDDDD<Scalar> = ArcTensorX<4, Scalar>;

/// rank-4 tensor of batched scalars with shape [D0 x D1 x D2 x B]
pub type ArcTensorDDDB<Scalar, const B: usize> = ArcTensorXB<4, 3, 1, Scalar, B>;

/// rank-4 tensor of vectors with shape [D0 x D1 x D2 x R]
pub type ArcTensorDDDR<Scalar, const R: usize> = ArcTensorXR<4, 3, 1, Scalar, R>;

/// rank-4 tensor of batched vectors with shape [D0 x D1 x R x B]
pub type ArcTensorDDRB<Scalar, const R: usize, const B: usize> =
    ArcTensorXRB<4, 2, 2, Scalar, R, B>;

/// rank-4 tensor of matrices with shape [D0 x R x C x B]
pub type ArcTensorDDRC<Scalar, const R: usize, const C: usize> =
    ArcTensorXRC<4, 2, 2, Scalar, R, C>;

/// rank-4 tensor of batched matrices with shape [D0 x R x C x B]
pub type ArcTensorDRCB<Scalar, const R: usize, const C: usize, const B: usize> =
    ArcTensorXRCB<4, 1, 3, Scalar, R, C, B>;

macro_rules! arc_tensor_is_tensor_view {
    ($scalar_rank:literal, $srank:literal,$drank:literal) => {


        impl<
            'a,
            Scalar: IsTensorScalar + 'static,
            STensor: IsStaticTensor<Scalar, $srank,  ROWS, COLS, BATCH_SIZE> + 'static,
            const ROWS: usize,
            const COLS: usize,
            const BATCH_SIZE: usize,
        > IsTensorLike<
            'a, $scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS, BATCH_SIZE
        > for ArcTensor<$scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS, BATCH_SIZE>
        {
            fn elem_view<'b:'a>(
                &'b self,
            ) -> ndarray::ArrayView<'a, STensor, ndarray::Dim<[ndarray::Ix; $drank]>> {
                self.view().elem_view
            }

            fn get(& self, idx: [usize; $drank]) -> STensor {
                self.view().get(idx)
            }

            fn dims(&self) -> [usize; $drank] {
                self.view().dims()
            }

            fn scalar_view<'b:'a>(
                &'b self,
            ) -> ndarray::ArrayView<'a, Scalar, ndarray::Dim<[ndarray::Ix; $scalar_rank]>> {
                self.view().scalar_view
            }

            fn scalar_get(&'a self, idx: [usize; $scalar_rank]) -> Scalar {
                self.view().scalar_get(idx)
            }

            fn scalar_dims(&self) -> [usize; $scalar_rank] {
                self.view().scalar_dims()
            }

            fn to_mut_tensor(
                &self,
            ) -> MutTensor<$scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS, BATCH_SIZE> {
                MutTensor {
                    mut_array: self.elem_view().to_owned(),
                    phantom: PhantomData::default(),
                }
            }
        }

        impl<
            'a,
            Scalar: IsTensorScalar+ 'static,
            STensor: IsStaticTensor<Scalar, $srank,  ROWS, COLS, BATCH_SIZE> + 'static,
            const ROWS: usize,
            const COLS: usize,
            const BATCH_SIZE: usize,
        >
            ArcTensor<$scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS, BATCH_SIZE>
        {

            /// create a new tensor from a tensor view
            pub fn make_copy_from(
                v: &TensorView<$scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS, BATCH_SIZE>
            ) -> Self
            {
                Self::from_mut_tensor(IsTensorLike::to_mut_tensor(v))
            }

            /// create a new tensor from a mutable tensor
           pub fn from_mut_tensor(
                tensor:
                MutTensor<$scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS, BATCH_SIZE>,
            ) -> Self {
                Self {
                    array: tensor.mut_array.into(),
                    phantom: PhantomData {},
                }
            }

            /// create a new tensor from a shape - all elements are zero
            pub fn from_shape(size: [usize; $drank]) -> Self {
                Self::from_mut_tensor(
                    MutTensor::<
                        $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS, BATCH_SIZE
                    >::from_shape(size))
            }

            /// create a new tensor from a shape and a value
            pub fn from_shape_and_val(
                shape: [usize; $drank],
                val:STensor,
            ) -> Self {
                Self::from_mut_tensor(
                    MutTensor::<
                        $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS, BATCH_SIZE
                    >::from_shape_and_val(shape, val),
                )
            }

            /// return a tensor view
            pub fn view<'b: 'a>(&'b self)
                -> TensorView<
                    'a, $scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS, BATCH_SIZE>
            {
                TensorView::<
                    'a, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS, BATCH_SIZE
                >::new(
                    self.array.view()
                )
            }

            /// create a new tensor from a unary operation applied to a tensor view
            pub fn from_map<
                'b,
                const OTHER_HRANK: usize, const OTHER_SRANK: usize,
                OtherScalar: IsTensorScalar+ 'static,
                OtherSTensor: IsStaticTensor<
                    OtherScalar, OTHER_SRANK, OTHER_ROWS, OTHER_COLS, OTHER_BATCHES
                > + 'static,
                const OTHER_ROWS: usize, const OTHER_COLS: usize,const OTHER_BATCHES: usize,
                V : IsTensorView::<
                    'b,
                    OTHER_HRANK, $drank, OTHER_SRANK,
                    OtherScalar, OtherSTensor,
                     OTHER_ROWS, OTHER_COLS,OTHER_BATCHES
                >,
                F: FnMut(&OtherSTensor)-> STensor
            >(
                view: &'b V,
                op: F,
            )
            ->  Self
            where
                ndarray::Dim<[ndarray::Ix; OTHER_HRANK]>: ndarray::Dimension,
                ndarray::Dim<[ndarray::Ix; $drank]>: ndarray::Dimension,
            {
                Self::from_mut_tensor(
                    MutTensor::<
                        $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS, BATCH_SIZE
                    >::from_map(view, op),
                )
            }

            /// create a new tensor from a binary operation applied to two tensor views
            pub fn from_map2<
                'b,
                const OTHER_HRANK: usize, const OTHER_SRANK: usize,
                OtherScalar: IsTensorScalar + 'static,
                OtherSTensor: IsStaticTensor<
                    OtherScalar, OTHER_SRANK, OTHER_ROWS, OTHER_COLS, OTHER_BATCHES
                > + 'static,
               const OTHER_ROWS: usize, const OTHER_COLS: usize, const OTHER_BATCHES: usize,
                V : IsTensorView::<
                        'b,
                        OTHER_HRANK, $drank, OTHER_SRANK,
                        OtherScalar, OtherSTensor,
                        OTHER_ROWS, OTHER_COLS, OTHER_BATCHES
                >,
                const OTHER_HRANK2: usize, const OTHER_SRANK2: usize,
                OtherScalar2: IsTensorScalar + 'static,
                OtherSTensor2: IsStaticTensor<
                    OtherScalar2, OTHER_SRANK2, OTHER_ROWS2, OTHER_COLS2, OTHER_BATCHES2
                > + 'static,
                const OTHER_ROWS2: usize, const OTHER_COLS2: usize, const OTHER_BATCHES2: usize,
                V2 : IsTensorView::<'b,
                    OTHER_HRANK2, $drank, OTHER_SRANK2,
                    OtherScalar2, OtherSTensor2,
                    OTHER_ROWS2, OTHER_COLS2,  OTHER_BATCHES2
                >,
                F: FnMut(&OtherSTensor, &OtherSTensor2) -> STensor
            > (
                view: &'b V,
                view2: &'b V2,
                op: F,
            )
            -> Self where
                ndarray::Dim<[ndarray::Ix; OTHER_HRANK]>: ndarray::Dimension,
                ndarray::Dim<[ndarray::Ix; OTHER_HRANK2]>: ndarray::Dimension
            {
                Self::from_mut_tensor(
                    MutTensor::<
                        $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS, BATCH_SIZE
                    >::from_map2(view,view2, op),
                )
            }
        }
    };
}

impl<Scalar: IsTensorScalar + 'static, const ROWS: usize> InnerVecToMat<3, 1, 2, 4, 2, Scalar, ROWS>
    for ArcTensorXR<3, 2, 1, Scalar, ROWS>
{
    fn inner_vec_to_mat(self) -> ArcTensorXRC<4, 2, 2, Scalar, ROWS, 1> {
        ArcTensorXRC::<4, 2, 2, Scalar, ROWS, 1> {
            array: self.array,
            phantom: PhantomData,
        }
    }

    type Output = ArcTensorXRC<4, 2, 2, Scalar, ROWS, 1>;
}

impl<Scalar: IsTensorScalar + 'static> InnerScalarToVec<2, 0, 2, 3, 1, Scalar>
    for ArcTensorX<2, Scalar>
{
    fn inner_scalar_to_vec(self) -> ArcTensorXR<3, 2, 1, Scalar, 1> {
        ArcTensorXR::<3, 2, 1, Scalar, 1> {
            array: self.array.map(|x| SVec::<Scalar, 1>::new(*x)).to_shared(),
            phantom: PhantomData,
        }
    }

    type Output = ArcTensorXR<3, 2, 1, Scalar, 1>;
}

arc_tensor_is_tensor_view!(1, 0, 1);
arc_tensor_is_tensor_view!(2, 0, 2);
arc_tensor_is_tensor_view!(2, 1, 1);
arc_tensor_is_tensor_view!(3, 0, 3);
arc_tensor_is_tensor_view!(3, 1, 2);
arc_tensor_is_tensor_view!(3, 2, 1);
arc_tensor_is_tensor_view!(4, 0, 4);
arc_tensor_is_tensor_view!(4, 1, 3);
arc_tensor_is_tensor_view!(4, 2, 2);
arc_tensor_is_tensor_view!(4, 3, 1);

#[cfg(test)]
mod tests {
    use crate::element::SVec;

    #[test]
    fn from_mut_tensor() {
        use super::*;

        use crate::mut_tensor::MutTensorDDDR;
        use crate::mut_tensor::MutTensorDDR;
        use crate::mut_tensor::MutTensorDR;

        {
            let shape = [4];
            let mut_img = MutTensorDR::from_shape_and_val(shape, SVec::<f32, 1>::new(0.5f32));
            let copy = MutTensorDR::make_copy_from(&mut_img.view());
            assert_eq!(copy.view().dims(), shape);
            let img = ArcTensorDR::from_mut_tensor(copy);
            assert_eq!(img.view().dims(), shape);
            let mut_img2 = ArcTensorDR::from_mut_tensor(mut_img.clone());
            assert_eq!(
                mut_img2.view().elem_view().as_slice().unwrap(),
                mut_img.view().elem_view().as_slice().unwrap()
            );
        }
        {
            let shape = [4, 2];
            let mut_img = MutTensorDDR::from_shape_and_val(shape, SVec::<f32, 1>::new(0.5f32));
            let copy = MutTensorDDR::make_copy_from(&mut_img.view());
            assert_eq!(copy.dims(), shape);
            let img = ArcTensorDDR::from_mut_tensor(copy);
            assert_eq!(img.dims(), shape);
            assert_eq!(
                img.view().elem_view().as_slice().unwrap(),
                mut_img.view().elem_view().as_slice().unwrap()
            );
        }
        {
            let shape = [3, 2, 7];
            let mut_img = MutTensorDDDR::from_shape_and_val(shape, SVec::<f32, 1>::new(0.5f32));
            let copy = MutTensorDDDR::make_copy_from(&mut_img.view());
            assert_eq!(copy.dims(), shape);
            let img = ArcTensorDDDR::from_mut_tensor(copy);
            assert_eq!(img.dims(), shape);
            assert_eq!(
                img.view().elem_view().as_slice().unwrap(),
                mut_img.view().elem_view().as_slice().unwrap()
            );
        }
    }

    #[test]
    fn shared_ownership() {
        use super::*;

        use crate::mut_tensor::MutTensorDDDR;
        use crate::mut_tensor::MutTensorDDR;
        use crate::mut_tensor::MutTensorDR;
        {
            let shape = [4];
            let mut_img = MutTensorDR::from_shape_and_val(shape, SVec::<f32, 1>::new(0.5f32));
            let img = ArcTensorDR::from_mut_tensor(mut_img);

            let img2 = img.clone();
            assert_eq!(
                img.view().elem_view().as_slice().unwrap(),
                img2.view().elem_view().as_slice().unwrap()
            );

            let mut_img2 = img2.to_mut_tensor();
            assert_ne!(
                mut_img2.view().elem_view().as_slice().unwrap().as_ptr(),
                img2.view().elem_view().as_slice().unwrap().as_ptr()
            );
        }
        {
            let shape = [4, 6];
            let mut_img = MutTensorDDR::from_shape_and_val(shape, SVec::<f32, 1>::new(0.5f32));
            let img = ArcTensorDDR::from_mut_tensor(mut_img);

            let img2 = img.clone();
            let mut_img2 = img2.to_mut_tensor();
            assert_ne!(
                mut_img2.view().elem_view().as_slice().unwrap().as_ptr(),
                img2.view().elem_view().as_slice().unwrap().as_ptr()
            );
        }
        {
            let shape = [4, 6, 7];
            let mut_img = MutTensorDDDR::from_shape_and_val(shape, SVec::<f32, 1>::new(0.5f32));
            let img = ArcTensorDDDR::from_mut_tensor(mut_img);

            let img2 = img.clone();
            let mut_img2 = img2.to_mut_tensor();
            assert_ne!(
                mut_img2.view().elem_view().as_slice().unwrap().as_ptr(),
                img2.view().elem_view().as_slice().unwrap().as_ptr()
            );
        }
    }

    #[test]
    fn multi_threading() {
        use crate::arc_tensor::ArcTensorDDRC;
        use crate::mut_tensor::MutTensorDDRC;
        use std::thread;

        let shape = [4, 6];
        let mut_img = MutTensorDDRC::from_shape_and_val(shape, SVec::<u16, 3>::new(10, 20, 300));
        let img = ArcTensorDDRC::from_mut_tensor(mut_img);

        thread::scope(|s| {
            s.spawn(|| {
                println!("{:?}", img);
            });
            s.spawn(|| {
                println!("{:?}", img);
            });
        });
    }
}
