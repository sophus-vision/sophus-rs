use crate::linalg::SMat;
use crate::linalg::SVec;
use crate::prelude::*;
use crate::tensor::ArcTensor;
use crate::tensor::MutTensorView;
use crate::tensor::TensorView;
use ndarray::Dim;
use ndarray::Ix;
use std::fmt::Debug;
use std::marker::PhantomData;

/// mutable tensor
///
/// See TensorView for more details of the tensor structure
#[derive(Default, Debug, Clone)]
pub struct MutTensor<
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar: IsCoreScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS> + 'static,
    const ROWS: usize,
    const COLS: usize,
> where
    ndarray::Dim<[ndarray::Ix; DRANK]>: ndarray::Dimension,
{
    /// ndarray of the static tensors with shape [D1, D2, ...]
    pub mut_array: ndarray::Array<STensor, Dim<[Ix; DRANK]>>,
    /// phantom data
    pub phantom: PhantomData<(Scalar, STensor)>,
}

/// Converting a tensor of vectors to a tensor of Rx1 matrices
pub trait InnerVecToMat<
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    const HYBER_RANK_PLUS1: usize,
    const SRANK_PLUS1: usize,
    Scalar: IsCoreScalar + 'static,
    const ROWS: usize,
> where
    SVec<Scalar, ROWS>: IsStaticTensor<Scalar, SRANK_PLUS1, ROWS, 1>,
    ndarray::Dim<[ndarray::Ix; DRANK]>: ndarray::Dimension,
{
    /// The output tensor
    type Output;

    /// Convert to a tensor of Rx1 matrices
    fn inner_vec_to_mat(self) -> Self::Output;
}

/// Converting a tensor of scalars to a tensor of 1-vectors
pub trait InnerScalarToVec<
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    const HYBER_RANK_PLUS1: usize,
    const SRANK_PLUS1: usize,
    Scalar: IsCoreScalar + 'static,
> where
    SVec<Scalar, 1>: IsStaticTensor<Scalar, SRANK_PLUS1, 1, 1>,
    ndarray::Dim<[ndarray::Ix; DRANK]>: ndarray::Dimension,
{
    /// The output tensor
    type Output;

    /// Convert to a tensor of 1-vectors
    fn inner_scalar_to_vec(self) -> Self::Output;
}

impl<Scalar: IsCoreScalar + 'static, const ROWS: usize> InnerVecToMat<3, 1, 2, 4, 2, Scalar, ROWS>
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

impl<Scalar: IsCoreScalar + 'static> InnerScalarToVec<2, 0, 2, 3, 1, Scalar>
    for MutTensorX<2, Scalar>
{
    type Output = MutTensorXR<3, 2, 1, Scalar, 1>;

    fn inner_scalar_to_vec(self) -> MutTensorXR<3, 2, 1, Scalar, 1> {
        MutTensorXR::<3, 2, 1, Scalar, 1> {
            mut_array: self.mut_array.map(|x| SVec::<Scalar, 1>::new(x.clone())),
            phantom: PhantomData,
        }
    }
}

/// Mutable tensor of scalars
pub type MutTensorX<const DRANK: usize, Scalar> = MutTensor<DRANK, DRANK, 0, Scalar, Scalar, 1, 1>;

/// Mutable tensor of vectors with shape R
pub type MutTensorXR<
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
> = MutTensor<TOTAL_RANK, DRANK, SRANK, Scalar, SVec<Scalar, R>, R, 1>;

/// Mutable tensor of matrices with shape [R x C]
pub type MutTensorXRC<
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
    const C: usize,
> = MutTensor<TOTAL_RANK, DRANK, SRANK, Scalar, SMat<Scalar, R, C>, R, C>;

/// rank-1 mutable tensor of scalars with shape D0
pub type MutTensorD<Scalar> = MutTensorX<1, Scalar>;

/// rank-2 mutable tensor of scalars with shape [D0 x D1]
pub type MutTensorDD<Scalar> = MutTensorX<2, Scalar>;

/// rank-2 mutable tensor of vectors with shape [D0 x R]
pub type MutTensorDR<Scalar, const R: usize> = MutTensorXR<2, 1, 1, Scalar, R>;

/// rank-3 mutable tensor of scalars with shape [D0 x D1 x D2]
pub type MutTensorDDD<Scalar> = MutTensorX<3, Scalar>;

/// rank-3 mutable tensor of vectors with shape [D0 x D1 x R]
pub type MutTensorDDR<Scalar, const R: usize> = MutTensorXR<3, 2, 1, Scalar, R>;

/// rank-3 mutable tensor of matrices with shape [D0 x R x C]
pub type MutTensorDRC<Scalar, const R: usize, const C: usize> = MutTensorXRC<3, 1, 2, Scalar, R, C>;

/// rank-4 mutable tensor of scalars with shape [D0 x D1 x D2 x D3]
pub type MutTensorDDDD<Scalar> = MutTensorX<4, Scalar>;

/// rank-4 mutable tensor of vectors with shape [D0 x D1 x D2 x R]
pub type MutTensorDDDR<Scalar, const R: usize> = MutTensorXR<4, 3, 1, Scalar, R>;

/// rank-4 mutable tensor of matrices with shape [D0 x D1 x R x C]
pub type MutTensorDDRC<Scalar, const R: usize, const C: usize> =
    MutTensorXRC<4, 2, 2, Scalar, R, C>;

/// rank-5 mutable tensor of scalars with shape [D0 x D1 x D2 x D3 x D4]
pub type MutTensorDDDDD<Scalar> = MutTensorX<5, Scalar>;

/// rank-5 mutable tensor of vectors with shape [D0 x D1 x D2 x D3 x R]
pub type MutTensorDDDDR<Scalar, const R: usize> = MutTensorXR<5, 4, 1, Scalar, R>;

/// rank-5 mutable tensor of matrices with shape [D0 x D1 x D2 x R x C]
pub type MutTensorDDDRC<Scalar, const R: usize, const C: usize> =
    MutTensorXRC<5, 3, 2, Scalar, R, C>;

macro_rules! mut_tensor_is_view {
    ($scalar_rank:literal, $srank:literal, $drank:literal) => {


        impl<
        'a,
                Scalar: IsCoreScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank,  ROWS, COLS> + 'static,
                const ROWS: usize,
                const COLS: usize,
            > IsTensorLike<'a, $scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS>
            for MutTensor<$scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS>
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
            ) -> MutTensor<$scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS> {
                MutTensor {
                    mut_array: self.elem_view().to_owned(),
                    phantom: PhantomData::default(),
                }
            }
        }

        impl<
        'a,
                Scalar: IsCoreScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank,  ROWS, COLS> + 'static,
                const ROWS: usize,
                const COLS: usize,

            >
            IsMutTensorLike<'a,
                $scalar_rank, $drank, $srank,
                Scalar, STensor,
                ROWS, COLS
            >
            for MutTensor<$scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS>
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

        impl<'a,  Scalar: IsCoreScalar+ 'static,
        STensor: IsStaticTensor<Scalar, $srank,  ROWS, COLS> + 'static,
        const ROWS: usize,
        const COLS: usize,

        > PartialEq for
            MutTensor<$scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS>
        {
            fn eq(&self, other: &Self) -> bool {
                self.view().scalar_view == other.view().scalar_view
            }
        }

        impl<'a,  Scalar: IsCoreScalar+ 'static,
                STensor: IsStaticTensor<Scalar, $srank,  ROWS, COLS> + 'static,
                const ROWS: usize,
                const COLS: usize,

        >
            MutTensor<$scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS>
        {

            /// create a new tensor from a shape - filled with zeros
            pub fn from_shape(size: [usize; $drank]) -> Self {
                MutTensor::<$scalar_rank, $drank, $srank, Scalar, STensor,
                            ROWS, COLS>::from_shape_and_val(
                    size, num_traits::Zero::zero()
                )
            }

             /// create a new mutable tensor by applying a binary operator to each element of two
            /// other tensors
            pub fn from_map2<
                'b,
                const OTHER_HRANK: usize, const OTHER_SRANK: usize,
                OtherScalar: IsCoreScalar + 'static,
                OtherSTensor: IsStaticTensor<
                    OtherScalar, OTHER_SRANK, OTHER_ROWS, OTHER_COLS
                > + 'static,
                const OTHER_ROWS: usize, const OTHER_COLS: usize,
            V : IsTensorView::<'b,
                OTHER_HRANK, $drank, OTHER_SRANK,
                OtherScalar, OtherSTensor,
                OTHER_ROWS, OTHER_COLS
            >,
            const OTHER_HRANK2: usize, const OTHER_SRANK2: usize,
            OtherScalar2: IsCoreScalar + 'static,
            OtherSTensor2: IsStaticTensor<
                OtherScalar2, OTHER_SRANK2, OTHER_ROWS2, OTHER_COLS2,
            > + 'static,
            const OTHER_ROWS2: usize, const OTHER_COLS2: usize,
            V2 : IsTensorView::<'b,
                OTHER_HRANK2, $drank, OTHER_SRANK2,
                OtherScalar2, OtherSTensor2,
                OTHER_ROWS2, OTHER_COLS2
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

        impl<'a,  Scalar: IsCoreScalar+ 'static,
                STensor: IsStaticTensor<Scalar, $srank, ROWS, COLS> + 'static,
                const ROWS: usize,
                const COLS: usize,

        >
            MutTensor<$scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS>
        {


            /// returns a mutable view of the tensor
            pub fn mut_view<'b: 'a>(
                &'b mut self,
            ) -> MutTensorView<'a,
                               $scalar_rank, $drank, $srank,
                               Scalar, STensor,
                               ROWS, COLS>
            {
                MutTensorView::<
                    'a,
                    $scalar_rank, $drank, $srank,
                    Scalar, STensor, ROWS, COLS>::new
                (
                    self.mut_array.view_mut()
                )
            }

            /// returns a view of the tensor
            pub fn view<'b: 'a>(&'b self
            ) -> TensorView<'a, $scalar_rank, $drank, $srank, Scalar, STensor,
                            ROWS, COLS> {
                TensorView::<'a, $scalar_rank, $drank, $srank, Scalar, STensor,
                             ROWS, COLS>::new(
                    self.mut_array.view())
            }


            /// create a new tensor from a shape and a value
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

            /// create a new mutable tensor by copying from another tensor
            pub fn make_copy_from(
                v: &TensorView<$scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS>
            ) -> Self
            {
                IsTensorLike::to_mut_tensor(v)
            }

            /// return ArcTensor copy of the mutable tensor
            pub fn to_shared(self)
                -> ArcTensor::<$scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS>
            {
                ArcTensor::<
                    $scalar_rank,
                    $drank, $srank,
                    Scalar, STensor,
                    ROWS, COLS>::from_mut_tensor(self)
            }

            /// create a new mutable tensor by applying a unary operator to each element of another
            /// tensor
            pub fn from_map<
                'b,
                const OTHER_HRANK: usize, const OTHER_SRANK: usize,
                OtherScalar: IsCoreScalar+ 'static,
                OtherSTensor: IsStaticTensor<
                    OtherScalar, OTHER_SRANK,
                    OTHER_ROWS, OTHER_COLS
                > + 'static,
                const OTHER_ROWS: usize, const OTHER_COLS: usize,
                V : IsTensorView::<
                    'b,
                    OTHER_HRANK, $drank, OTHER_SRANK,
                    OtherScalar, OtherSTensor,
                    OTHER_ROWS, OTHER_COLS
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
mut_tensor_is_view!(5, 0, 5);
mut_tensor_is_view!(5, 1, 4);
mut_tensor_is_view!(5, 2, 3);

#[test]
fn mut_tensor_tests() {
    use crate::linalg::BatchMatF64;
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
    //transform
    {
        let shape = [3];
        {
            let tensor_f32 = MutTensorD::from_shape_and_val(shape, 1.0);
            let op = |v: &f32| {
                let mut value = SVec::<f32, 3>::default();
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
                let mut value = SVec::<f32, 3>::default();
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
                let mut value = SVec::<f32, 3>::default();
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

    //linalg
    {
        let shape = [3];

        let _tensor_u8 = MutTensorD::from_shape_and_val(shape, 0);
        let _tensor_f64 = MutTensorDRC::from_shape_and_val(shape, SMat::<f64, 4, 4>::zeros());
        let _tensor_batched_f32 =
            MutTensorDRC::from_shape_and_val(shape, BatchMatF64::<2, 3, 4>::zeros());
    }

    //from_raw_data
    {
        let shape = [1];
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let data_mat = SMat::<f32, 3, 2>::from_vec(data.to_vec());
        let tensor_f32 = MutTensorDRC::from_shape_and_val(shape, data_mat);
        assert_eq!(tensor_f32.dims(), shape);
        assert_eq!(tensor_f32.view().scalar_get([0, 0, 0]), data[0]);
        assert_eq!(tensor_f32.view().scalar_get([0, 1, 0]), data[1]);
        assert_eq!(tensor_f32.view().scalar_get([0, 2, 0]), data[2]);
        assert_eq!(tensor_f32.view().scalar_get([0, 0, 1]), data[3]);
        assert_eq!(tensor_f32.view().scalar_get([0, 1, 1]), data[4]);
        assert_eq!(tensor_f32.view().scalar_get([0, 2, 1]), data[5]);
    }
}
