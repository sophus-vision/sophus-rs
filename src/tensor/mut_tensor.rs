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

#[derive(Default, Debug, Clone)]
pub struct MutTensor<
    const HYBER_RANK: usize,
    const SRANK: usize,
    const DRANK: usize,
    Scalar: IsTensorScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, BATCHES, ROWS, COLS> + 'static,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
> where
    ndarray::Dim<[ndarray::Ix; DRANK]>: ndarray::Dimension,
{
    pub mut_array: ndarray::Array<STensor, Dim<[Ix; DRANK]>>,
    pub phantom: PhantomData<(Scalar, STensor)>,
}

pub trait InnerVecToMat<
    const HYBER_RANK: usize,
    const SRANK: usize,
    const DRANK: usize,
    const HYBER_RANK_PLUS1: usize,
    const SRANK_PLUS1: usize,
    Scalar: IsTensorScalar + 'static,
    const ROWS: usize,
> where
    SVec<Scalar, ROWS>: IsStaticTensor<Scalar, SRANK_PLUS1, 1, ROWS, 1>,
    ndarray::Dim<[ndarray::Ix; DRANK]>: ndarray::Dimension,
{
    type Output;

    fn inner_vec_to_mat(self) -> Self::Output;
}

pub trait InnerScalarToVec<
    const HYBER_RANK: usize,
    const SRANK: usize,
    const DRANK: usize,
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
    for MutTensorXR<3, 1, 2, Scalar, ROWS>
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
    type Output = MutTensorXR<3, 1, 2, Scalar, 1>;

    fn inner_scalar_to_vec(self) -> MutTensorXR<3, 1, 2, Scalar, 1> {
        MutTensorXR::<3, 1, 2, Scalar, 1> {
            mut_array: self.mut_array.map(|x| SVec::<Scalar, 1>::new(*x)),
            phantom: PhantomData,
        }
    }
}

pub type MutTensorX<const DRANK: usize, Scalar> =
    MutTensor<DRANK, 0, DRANK, Scalar, Scalar, 1, 1, 1>;

pub type MutTensorXB<
    const HYBER_RANK: usize,
    const SRANK: usize,
    const DRANK: usize,
    Scalar,
    const B: usize,
> = MutTensor<HYBER_RANK, SRANK, DRANK, Scalar, BatchScalar<Scalar, B>, B, 1, 1>;

pub type MutTensorXR<
    const HYBER_RANK: usize,
    const SRANK: usize,
    const DRANK: usize,
    Scalar,
    const R: usize,
> = MutTensor<HYBER_RANK, SRANK, DRANK, Scalar, SVec<Scalar, R>, 1, R, 1>;

pub type MutTensorXBR<
    const HYBER_RANK: usize,
    const SRANK: usize,
    const DRANK: usize,
    Scalar,
    const B: usize,
    const R: usize,
> = MutTensor<HYBER_RANK, SRANK, DRANK, Scalar, BatchVec<Scalar, B, R>, B, R, 1>;

pub type MutTensorXRC<
    const HYBER_RANK: usize,
    const SRANK: usize,
    const DRANK: usize,
    Scalar,
    const R: usize,
    const C: usize,
> = MutTensor<HYBER_RANK, SRANK, DRANK, Scalar, SMat<Scalar, R, C>, 1, R, C>;

pub type MutTensorXBRC<
    const HYBER_RANK: usize,
    const SRANK: usize,
    const DRANK: usize,
    Scalar,
    const B: usize,
    const R: usize,
    const C: usize,
> = MutTensor<HYBER_RANK, SRANK, DRANK, Scalar, BatchMat<Scalar, B, R, C>, B, R, C>;

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
pub type MutTensorDDB<Scalar, const B: usize> = MutTensorXB<3, 1, 2, Scalar, B>;

// rank 3
// D0 x D1 x [R]
pub type MutTensorDDR<Scalar, const R: usize> = MutTensorXR<3, 1, 2, Scalar, R>;

// rank 3
// D0 x [B x R]
pub type MutTensorBRD<Scalar, const B: usize, const R: usize> = MutTensorXBR<3, 2, 1, Scalar, B, R>;

// rank 3
// D0 x [R x C]
pub type MutTensorDRC<Scalar, const R: usize, const C: usize> = MutTensorXRC<3, 2, 1, Scalar, R, C>;

// rank 4
// srank 0, drank 4
pub type MutTensorDDDD<Scalar> = MutTensorX<4, Scalar>;

// rank 4
// D0 x D1 x D2 x [B]
pub type MutTensorDDDB<Scalar, const B: usize> = MutTensorXB<4, 1, 3, Scalar, B>;

// rank 4
//  D0 x D1 x D2 x [R]
pub type MutTensorDDDR<Scalar, const R: usize> = MutTensorXR<4, 1, 3, Scalar, R>;

// rank 4
// D0 x D1 x [B x R]
pub type MutTensorDDRB<Scalar, const B: usize, const R: usize> =
    MutTensorXBR<4, 2, 2, Scalar, B, R>;

// rank 4
//  D0 x D1 x [R x C]
pub type MutTensorDDRC<Scalar, const R: usize, const C: usize> =
    MutTensorXRC<4, 2, 2, Scalar, R, C>;

// rank 4
// D0 x [B x R x C]
pub type MutTensorDRCB<Scalar, const B: usize, const R: usize, const C: usize> =
    MutTensorXBRC<4, 3, 1, Scalar, B, R, C>;

macro_rules! mut_tensor_is_view {
    ($hyper_rank:literal, $srank:literal, $drank:literal) => {


        impl<
        'a,
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, BATCHES, ROWS, COLS> + 'static,
                const BATCHES: usize,
                const ROWS: usize,
                const COLS: usize,
            > IsTensorLike<'a, $hyper_rank, $srank, $drank, Scalar, STensor, BATCHES, ROWS, COLS>
            for MutTensor<$hyper_rank, $srank, $drank, Scalar, STensor, BATCHES, ROWS, COLS>
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
            ) -> ndarray::ArrayView<'a, Scalar, ndarray::Dim<[ndarray::Ix; $hyper_rank]>> {
                self.view().scalar_view
            }

            fn scalar_get(&'a self, idx: [usize; $hyper_rank]) -> Scalar {
                self.view().scalar_get(idx)
            }

            fn scalar_dims(&self) -> [usize; $hyper_rank] {
                self.view().scalar_dims()
            }

            fn to_mut_image(
                &self,
            ) -> MutTensor<$hyper_rank, $srank, $drank, Scalar, STensor, BATCHES, ROWS, COLS> {
                MutTensor {
                    mut_array: self.elem_view().to_owned(),
                    phantom: PhantomData::default(),
                }
            }
        }

        impl<
        'a,
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, BATCHES, ROWS, COLS> + 'static,
                const BATCHES: usize,
                const ROWS: usize,
                const COLS: usize,
            > IsMutTensorLike<'a, $hyper_rank, $srank, $drank, Scalar, STensor, BATCHES, ROWS, COLS>
            for MutTensor<$hyper_rank, $srank, $drank, Scalar, STensor, BATCHES, ROWS, COLS>
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
            ) -> ndarray::ArrayViewMut<'a, Scalar, ndarray::Dim<[ndarray::Ix; $hyper_rank]>>{
                self.mut_view().scalar_view_mut
            }
        }

        impl<'a,  Scalar: IsTensorScalar+ 'static,
                STensor: IsStaticTensor<Scalar, $srank, BATCHES, ROWS, COLS> + 'static,
                const BATCHES: usize,
                const ROWS: usize,
                const COLS: usize>
               MutTensor<$hyper_rank, $srank,$drank, Scalar, STensor, BATCHES, ROWS, COLS>
        {

         pub fn mut_view<'b: 'a>(
                &'b mut self,
            ) -> MutTensorView<'a,  $hyper_rank, $srank,$drank, Scalar, STensor, BATCHES, ROWS, COLS>{
                MutTensorView::<'a,  $hyper_rank, $srank,$drank, Scalar, STensor, BATCHES, ROWS, COLS>::new(
                    self.mut_array.view_mut())
            }

         pub    fn view<'b: 'a>(&'b self) -> TensorView<'a, $hyper_rank, $srank,$drank, Scalar, STensor, BATCHES, ROWS, COLS> {
                TensorView::<'a,  $hyper_rank, $srank,$drank, Scalar, STensor, BATCHES, ROWS, COLS>::new(
                    self.mut_array.view())
            }

          pub   fn from_shape(size: [usize; $drank]) -> Self {
                MutTensor::<$hyper_rank, $srank,$drank, Scalar, STensor, BATCHES, ROWS, COLS>::from_shape_and_val(
                    size, STensor::zero()
                )
            }

            pub fn make_copy_from(v: &TensorView<$hyper_rank, $srank,$drank, Scalar, STensor, BATCHES, ROWS, COLS>)->Self {
                IsTensorLike::to_mut_image(v)
            }

            pub fn to_shared(self)  ->  ArcTensor::<$hyper_rank, $srank,$drank, Scalar, STensor, BATCHES, ROWS, COLS> {
                ArcTensor::<$hyper_rank, $srank,$drank, Scalar, STensor, BATCHES, ROWS, COLS>::from_mut_tensor(self)
            }



          pub   fn from_shape_and_val(
                shape: [usize; $drank],
                val: STensor,
            ) -> Self {


                Self{
                    mut_array: ndarray::Array::<STensor, Dim<[Ix; $drank]>>::from_elem(shape, val),
                    phantom: PhantomData::default()
                }
            }


            pub fn from_map<
                'b,
                const OTHER_HRANK:usize,
                const OTHER_SRANK:usize,
                OtherScalar: IsTensorScalar+ 'static,
                OtherSTensor: IsStaticTensor<OtherScalar, OTHER_SRANK, OTHER_BATCHES, OTHER_ROWS, OTHER_COLS> + 'static,
                const OTHER_BATCHES: usize,
                const OTHER_ROWS: usize,
                const OTHER_COLS: usize,
                V : IsTensorView::<'b,
                    OTHER_HRANK,
                    OTHER_SRANK,
                    $drank,
                    OtherScalar,
                    OtherSTensor,
                    OTHER_BATCHES,
                    OTHER_ROWS,
                    OTHER_COLS>,
                F: FnMut(&OtherSTensor)-> STensor
            >(
                view:  &'b V,
                op: F,
            )
            ->  Self
            where
            ndarray::Dim<[ndarray::Ix; OTHER_HRANK]>: ndarray::Dimension,
            ndarray::Dim<[ndarray::Ix; $drank]>: ndarray::Dimension,
            {
                Self{
                    mut_array: view.elem_view().map(op),
                    phantom: PhantomData::default()
                }
            }

            pub fn from_map2<
            'b,
            const OTHER_HRANK: usize,
            const OTHER_SRANK: usize,
            OtherScalar: IsTensorScalar + 'static,
            OtherSTensor: IsStaticTensor<OtherScalar, OTHER_SRANK, OTHER_BATCHES, OTHER_ROWS, OTHER_COLS> + 'static,
            const OTHER_BATCHES: usize,
            const OTHER_ROWS: usize,
            const OTHER_COLS: usize,
            V : IsTensorView::<'b,
                    OTHER_HRANK,
                    OTHER_SRANK,
                    $drank,
                    OtherScalar,
                    OtherSTensor,
                    OTHER_BATCHES,
                    OTHER_ROWS,
                    OTHER_COLS>,
            const OTHER_HRANK2: usize,
            const OTHER_SRANK2: usize,
            OtherScalar2: IsTensorScalar + 'static,
            OtherSTensor2: IsStaticTensor<OtherScalar2, OTHER_SRANK2, OTHER_BATCHES2, OTHER_ROWS2, OTHER_COLS2> + 'static,
            const OTHER_BATCHES2: usize,
            const OTHER_ROWS2: usize,
            const OTHER_COLS2: usize,
            V2 : IsTensorView::<'b,
                    OTHER_HRANK2,
                    OTHER_SRANK2,
                    $drank,
                    OtherScalar2,
                    OtherSTensor2,
                    OTHER_BATCHES2,
                    OTHER_ROWS2,
                    OTHER_COLS2>,
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
