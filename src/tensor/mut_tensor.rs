use super::element::BatchMat;
use super::element::BatchScalar;
use super::element::BatchVec;
use super::element::IsStaticTensor;
use super::element::IsTensorScalar;
use super::element::SMat;
use super::element::SVec;
use ndarray::Dim;
use ndarray::Ix;
use std::fmt::Debug;
use std::marker::PhantomData;

use super::arc_tensor::ArcTensor;
use super::mut_view::IsMutTensorLike;
use super::mut_view::MutTensorView;
use super::view::IsTensorLike;
use super::view::IsTensorView;
use super::view::TensorView;

/// A mutable tensor
///
/// There are two ways of describing the Tensor (MutTensor as well as its siblings ArcTensor,
/// TensorView and MutTensorView):
///
///  1. A dynamic tensor of static tensors:
///    * The dynamic tensor is of rank DRANK
///      - ``self.dims()`` is used to access its dynamic dimensions of type
///        ``[usize: DRANK]``.
///      - an individual element (= static tensor) can be accessed with
///        ``self.get(idx)``, where idx is f type ``[usize: DRANK]``.
///      - Each element is of type ``STensor``.
///    * Each static tensor is of SRANK. In particular we have.
///      - rank 0: scalars of type ``Scalar`` (such as ``f64`` or ``u8``).
///      - rank 1:
///         * A batch scalar of type ``BatchScalar<Scalar, BATCH>`` with static
///           batch size of BATCHES.
///         * A column vector ``SVec<Scalar, ROWS>`` aka ``nalgebra::SVector<Scalar, ROWS>`` with
///           number of ROWS.
///      - rank 2:
///         * A batch vector of type ``BatchVector<Scalar, BATCHES>`` with static
///           shape (ROWS x BATCHES).
///         * A matrix ``SMat<Scalar, ROWS, COLS>`` aka ``nalgebra::SMatrix<Scalar, ROWS, COLS>``
///           with static shape (ROWS x COLS).
///       - rank 3:
///         * A batch matrix of type ``BatchMatrix<Scalar, ROWS, COLS, BATCH>`` with static
///           shape (BATCHES x ROWS .x COLS).
///  2. A scalar tensor of SCALAR_RANK = DRANK + SRANK.
///    *  ``self.scalar_dims()`` is used to access its dimensions of type
///        ``[usize: SCALAR_RANK]`` at runtime.
///    *  - an individual element (= static tensor) can be accessed with
///        ``self.scalar_get(idx)``, where idx is of type ``[usize: DRANK]``.
#[derive(Default, Debug, Clone)]
pub struct MutTensor<
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
{
    pub mut_array: ndarray::Array<STensor, Dim<[Ix; DRANK]>>,
    pub phantom: PhantomData<(Scalar, STensor)>,
}

pub trait InnerVecToMat<
    const SCALAR_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    const HYBER_RANK_PLUS1: usize,
    const SRANK_PLUS1: usize,
    Scalar: IsTensorScalar + 'static,
    const ROWS: usize,
> where
    SVec<Scalar, ROWS>: IsStaticTensor<Scalar, SRANK_PLUS1, ROWS, 1, 1>,
    ndarray::Dim<[ndarray::Ix; DRANK]>: ndarray::Dimension,
{
    type Output;

    fn inner_vec_to_mat(self) -> Self::Output;
}

pub trait InnerScalarToVec<
    const SCALAR_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    const HYBER_RANK_PLUS1: usize,
    const SRANK_PLUS1: usize,
    Scalar: IsTensorScalar + 'static,
> where
    SVec<Scalar, 1>: IsStaticTensor<Scalar, SRANK_PLUS1, 1, 1, 1>,
    ndarray::Dim<[ndarray::Ix; DRANK]>: ndarray::Dimension,
{
    type Output;
    fn inner_scalar_to_vec(self) -> Self::Output;
}

impl<Scalar: IsTensorScalar + 'static, const ROWS: usize> InnerVecToMat<3, 1, 2, 4, 2, Scalar, ROWS>
    for MutTensorXR<3, 2, 1, Scalar, ROWS>
{
    type Output = MutTensorXRC<4, 2, 2, Scalar, ROWS, 1>;

    fn inner_vec_to_mat(self) -> MutTensorXRC<4, 2, 2, Scalar, ROWS, 1> {
        MutTensorXRC::<4, 2, 2, Scalar, ROWS, 1> {
            mut_array: self.mut_array,
            phantom: PhantomData,
        }
    }
}

impl<Scalar: IsTensorScalar + 'static> InnerScalarToVec<2, 0, 2, 3, 1, Scalar>
    for MutTensorX<2, Scalar>
{
    type Output = MutTensorXR<3, 2, 1, Scalar, 1>;

    fn inner_scalar_to_vec(self) -> MutTensorXR<3, 2, 1, Scalar, 1> {
        MutTensorXR::<3, 2, 1, Scalar, 1> {
            mut_array: self.mut_array.map(|x| SVec::<Scalar, 1>::new(*x)),
            phantom: PhantomData,
        }
    }
}

pub type MutTensorX<const DRANK: usize, Scalar> =
    MutTensor<DRANK, DRANK, 0, Scalar, Scalar, 1, 1, 1>;

pub type MutTensorXB<
    const SCALAR_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const B: usize,
> = MutTensor<SCALAR_RANK, DRANK, SRANK, Scalar, BatchScalar<Scalar, B>, 1, 1, B>;

pub type MutTensorXR<
    const SCALAR_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
> = MutTensor<SCALAR_RANK, DRANK, SRANK, Scalar, SVec<Scalar, R>, R, 1, 1>;

pub type MutTensorXRB<
    const SCALAR_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
    const B: usize,
> = MutTensor<SCALAR_RANK, DRANK, SRANK, Scalar, BatchVec<Scalar, R, B>, R, 1, B>;

pub type MutTensorXRC<
    const SCALAR_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
    const C: usize,
> = MutTensor<SCALAR_RANK, DRANK, SRANK, Scalar, SMat<Scalar, R, C>, R, C, 1>;

pub type MutTensorXRCB<
    const SCALAR_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
    const C: usize,
    const B: usize,
> = MutTensor<SCALAR_RANK, DRANK, SRANK, Scalar, BatchMat<Scalar, R, C, B>, R, C, B>;

// rank 1 - D0
pub type MutTensorD<Scalar> = MutTensorX<1, Scalar>;

// rank 2
// D0 x D1
pub type MutTensorDD<Scalar> = MutTensorX<2, Scalar>;

// rank 2
// D0 x [B]
pub type MutTensorDB<Scalar, const B: usize> = MutTensorXB<2, 1, 1, Scalar, B>;

// rank 2
// D0 x [R]
pub type MutTensorDR<Scalar, const R: usize> = MutTensorXR<2, 1, 1, Scalar, R>;

// rank 3
// D0 x D1 x D2
pub type MutTensorDDD<Scalar> = MutTensorX<3, Scalar>;

// rank 3
//  D0 x D1 x [B]
pub type MutTensorDDB<Scalar, const B: usize> = MutTensorXB<3, 2, 1, Scalar, B>;

// rank 3
// D0 x D1 x [R]
pub type MutTensorDDR<Scalar, const R: usize> = MutTensorXR<3, 2, 1, Scalar, R>;

// rank 3
// D0 x [R, B]
pub type MutTensorDRB<Scalar, const R: usize, const B: usize> = MutTensorXRB<3, 1, 2, Scalar, R, B>;

// rank 3
// D0 x [R x C]
pub type MutTensorDRC<Scalar, const R: usize, const C: usize> = MutTensorXRC<3, 1, 2, Scalar, R, C>;

// rank 4
// srank 0, drank 4
pub type MutTensorDDDD<Scalar> = MutTensorX<4, Scalar>;

// rank 4
// D0 x D1 x D2 x [B]
pub type MutTensorDDDB<Scalar, const B: usize> = MutTensorXB<4, 3, 1, Scalar, B>;

// rank 4
//  D0 x D1 x D2 x [R]
pub type MutTensorDDDR<Scalar, const R: usize> = MutTensorXR<4, 3, 1, Scalar, R>;

// rank 4
// D0 x D1 x [R x B]
pub type MutTensorDDRB<Scalar, const R: usize, const B: usize> =
    MutTensorXRB<4, 2, 2, Scalar, R, B>;

// rank 4
//  D0 x D1 x [R x C]
pub type MutTensorDDRC<Scalar, const R: usize, const C: usize> =
    MutTensorXRC<4, 2, 2, Scalar, R, C>;

// rank 4
// D0 x [R x C x B]
pub type MutTensorDRCB<Scalar, const R: usize, const C: usize, const B: usize> =
    MutTensorXRCB<4, 1, 3, Scalar, R, C, B>;

macro_rules! mut_tensor_is_view {
    ($scalar_rank:literal, $srank:literal, $drank:literal) => {


        impl<
        'a,
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank,  ROWS, COLS, BATCHES> + 'static,
                const ROWS: usize,
                const COLS: usize,
                const BATCHES: usize,
            > IsTensorLike<'a, $scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS, BATCHES>
            for MutTensor<$scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS, BATCHES>
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

            fn to_mut_image(
                &self,
            ) -> MutTensor<$scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS, BATCHES> {
                MutTensor {
                    mut_array: self.elem_view().to_owned(),
                    phantom: PhantomData::default(),
                }
            }
        }

        impl<
        'a,
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank,  ROWS, COLS, BATCHES> + 'static,
                const ROWS: usize,
                const COLS: usize,
                const BATCHES: usize,
            >
            IsMutTensorLike<'a,
                $scalar_rank, $drank, $srank,
                Scalar, STensor,
                ROWS, COLS, BATCHES
            >
            for MutTensor<$scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
        {
            fn elem_view_mut<'b:'a>(
                &'b mut self,
            ) -> ndarray::ArrayViewMut<'a, STensor, ndarray::Dim<[ndarray::Ix; $drank]>>{
                self.mut_view().elem_view_mut
            }
            fn get_mut(& mut self, idx: [usize; $drank]) -> &mut STensor{
                &mut self.mut_array[idx]
            }

            fn scalar_view_mut<'b:'a>(
                &'b mut self,
            ) -> ndarray::ArrayViewMut<'a, Scalar, ndarray::Dim<[ndarray::Ix; $scalar_rank]>>{
                self.mut_view().scalar_view_mut
            }
        }

        impl<'a,  Scalar: IsTensorScalar+ 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS, BATCHES> + 'static,
                const ROWS: usize,
                const COLS: usize,
                const BATCHES: usize,
        >
            MutTensor<$scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
        {

            pub fn mut_view<'b: 'a>(
                &'b mut self,
            ) -> MutTensorView<'a,
                               $scalar_rank, $drank, $srank,
                               Scalar, STensor,
                               ROWS, COLS, BATCHES>
            {
                MutTensorView::<
                    'a,
                    $scalar_rank, $drank, $srank,
                    Scalar, STensor, ROWS, COLS, BATCHES>::new
                (
                    self.mut_array.view_mut()
                )
            }

            pub fn view<'b: 'a>(&'b self
            ) -> TensorView<'a, $scalar_rank, $drank, $srank, Scalar, STensor,
                            ROWS, COLS, BATCHES> {
                TensorView::<'a, $scalar_rank, $drank, $srank, Scalar, STensor,
                             ROWS, COLS, BATCHES>::new(
                    self.mut_array.view())
            }

            pub fn from_shape(size: [usize; $drank]) -> Self {
                MutTensor::<$scalar_rank, $drank, $srank, Scalar, STensor,
                            ROWS, COLS, BATCHES>::from_shape_and_val(
                    size, STensor::zero()
                )
            }

            pub fn make_copy_from(
                v: &TensorView<$scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS, BATCHES>
            ) -> Self
            {
                IsTensorLike::to_mut_image(v)
            }

            pub fn to_shared(self)
                -> ArcTensor::<$scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS, BATCHES>
            {
                ArcTensor::<
                    $scalar_rank,
                    $drank, $srank,
                    Scalar, STensor,
                    ROWS, COLS, BATCHES>::from_mut_tensor(self)
            }



            pub fn from_shape_and_val
            (
                shape: [usize; $drank],
                val: STensor,
            ) -> Self
            {
                Self{
                    mut_array: ndarray::Array::<STensor, Dim<[Ix; $drank]>>::from_elem(shape, val),
                    phantom: PhantomData::default()
                }
            }


            pub fn from_map<
                'b,
                const OTHER_HRANK: usize, const OTHER_SRANK: usize,
                OtherScalar: IsTensorScalar+ 'static,
                OtherSTensor: IsStaticTensor<
                    OtherScalar, OTHER_SRANK,
                    OTHER_ROWS, OTHER_COLS, OTHER_BATCHES
                > + 'static,
                const OTHER_ROWS: usize, const OTHER_COLS: usize, const OTHER_BATCHES: usize,
                V : IsTensorView::<
                    'b,
                    OTHER_HRANK, $drank, OTHER_SRANK,
                    OtherScalar, OtherSTensor,
                    OTHER_ROWS, OTHER_COLS, OTHER_BATCHES
                >,
                F: FnMut(&OtherSTensor)-> STensor
            > (
                view:  &'b V,
                op: F,
            )
            -> Self where
                ndarray::Dim<[ndarray::Ix; OTHER_HRANK]>: ndarray::Dimension,
                ndarray::Dim<[ndarray::Ix; $drank]>: ndarray::Dimension,
            {
                Self {
                    mut_array: view.elem_view().map(op),
                    phantom: PhantomData::default()
                }
            }

            pub fn from_map2<
                'b,
                const OTHER_HRANK: usize, const OTHER_SRANK: usize,
                OtherScalar: IsTensorScalar + 'static,
                OtherSTensor: IsStaticTensor<
                    OtherScalar, OTHER_SRANK, OTHER_ROWS, OTHER_COLS, OTHER_BATCHES
                > + 'static,
                const OTHER_ROWS: usize, const OTHER_COLS: usize, const OTHER_BATCHES: usize,
            V : IsTensorView::<'b,
                OTHER_HRANK, $drank, OTHER_SRANK,
                OtherScalar, OtherSTensor,
                OTHER_ROWS, OTHER_COLS, OTHER_BATCHES
            >,
            const OTHER_HRANK2: usize, const OTHER_SRANK2: usize,
            OtherScalar2: IsTensorScalar + 'static,
            OtherSTensor2: IsStaticTensor<
                OtherScalar2, OTHER_SRANK2, OTHER_ROWS2, OTHER_COLS2, OTHER_BATCHES2,
            > + 'static,
            const OTHER_ROWS2: usize, const OTHER_COLS2: usize, const OTHER_BATCHES2: usize,
            V2 : IsTensorView::<'b,
                OTHER_HRANK2, $drank, OTHER_SRANK2,
                OtherScalar2, OtherSTensor2,
                OTHER_ROWS2, OTHER_COLS2, OTHER_BATCHES2
            >,
            F: FnMut(&OtherSTensor, &OtherSTensor2)->STensor
            >(
                view: &'b V,
                view2: &'b V2,
                mut op: F,
            )
            -> Self
            where
                ndarray::Dim<[ndarray::Ix; OTHER_HRANK]>: ndarray::Dimension,
                ndarray::Dim<[ndarray::Ix; OTHER_HRANK2]>: ndarray::Dimension

            {
                let mut out  = Self::from_shape(view.dims());
                ndarray::Zip::from(&mut out.elem_view_mut())
                .and(&view.elem_view())
                .and(&view2.elem_view())
                .for_each(
                    |out, v, v2|{
                      *out = op(v,v2);
                    });
                out
            }
        }
    };
}

mut_tensor_is_view!(1, 0, 1);
mut_tensor_is_view!(2, 0, 2);
mut_tensor_is_view!(2, 1, 1);
mut_tensor_is_view!(3, 0, 3);
mut_tensor_is_view!(3, 1, 2);
mut_tensor_is_view!(3, 2, 1);
mut_tensor_is_view!(4, 0, 4);
mut_tensor_is_view!(4, 1, 3);
mut_tensor_is_view!(4, 2, 2);
mut_tensor_is_view!(4, 3, 1);

#[cfg(test)]
mod tests {
    use simba::simd::AutoSimd;

    use super::*;

    use crate::tensor::element::P3F32;

    #[test]
    fn empty_image() {
        {
            let _rank1_tensor = MutTensorD::<u8>::default();
            //assert!(rank1_tensor.is_empty());
            let shape = [2];
            let tensor_f32 = MutTensorD::from_shape_and_val(shape, 0.0);
            //assert!(!tensor_f32.is_empty());
            assert_eq!(tensor_f32.view().dims(), shape);
        }
        {
            let _rank2_tensor = MutTensorDD::<u8>::default();
            //assert!(rank2_tensor.is_empty());
            let shape = [3, 2];
            let tensor_f32 = MutTensorDD::<f32>::from_shape(shape);
            // assert!(!tensor_f32.is_empty());
            assert_eq!(tensor_f32.view().dims(), shape);
        }
        {
            let _rank3_tensor = MutTensorDDD::<u8>::default();
            // assert!(rank3_tensor.is_empty());
            let shape = [3, 2, 4];
            let tensor_f32 = MutTensorDDD::<f32>::from_shape(shape);
            //  assert!(!tensor_f32.is_empty());
            assert_eq!(tensor_f32.view().dims(), shape);
        }
    }

    #[test]
    pub fn transform() {
        let shape = [3];
        {
            let tensor_f32 = MutTensorD::from_shape_and_val(shape, 1.0);
            let op = |v: &f32| {
                let mut value = P3F32::default();
                value[0] = *v;
                value[1] = 0.2 * *v;
                value[2] = 0.3 * *v;
                value
            };
            let pattern = MutTensorDR::<f32, 3>::from_map(&tensor_f32.view(), op);

            println!("p :{}", pattern.mut_array);
            // assert_eq!(
            //     pattern.slice(),
            //     MutTensorDR::from_shape_and_val(shape, op(1.0)).slice()
            // );
        }
        let shape = [3, 2];
        {
            let tensor_f32 = MutTensorDD::from_shape_and_val(shape, 1.0);
            let op = |v: &f32| {
                let mut value = P3F32::default();
                value[0] = *v;
                value[1] = 0.2 * *v;
                value[2] = 0.3 * *v;
                value
            };
            let pattern = MutTensorDDR::from_map(&tensor_f32.view(), op);
            println!("p :{}", pattern.mut_array);
            println!("p :{}", pattern.view().scalar_view());
        }
        let shape = [3, 2, 4];
        {
            let tensor_f32 = MutTensorDDD::from_shape_and_val(shape, 1.0);
            let op = |v: &f32| {
                let mut value = P3F32::default();
                value[0] = *v;
                value[1] = 0.2 * *v;
                value[2] = 0.3 * *v;
                value
            };
            let pattern = MutTensorDDDR::from_map(&tensor_f32.view(), op);
            println!("p :{}", pattern.mut_array);
            println!("p :{}", pattern.view().scalar_view());
        }
    }

    #[test]
    pub fn types() {
        let shape = [3];

        let _tensor_u8 = MutTensorD::from_shape_and_val(shape, 0);
        let _tensor_f32 = MutTensorDRC::from_shape_and_val(shape, SMat::<f32, 4, 4>::zeros());
        let _tensor_batched_f32 =
            MutTensorDRCB::from_shape_and_val(shape, SMat::<AutoSimd<[f32; 8]>, 4, 4>::zeros());
    }
}
