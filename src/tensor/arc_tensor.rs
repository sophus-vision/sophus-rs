use ndarray::Dimension;

use super::element::BatchMat;
use super::element::BatchScalar;
use super::element::BatchVec;
use super::element::IsStaticTensor;
use super::element::IsTensorScalar;
use super::element::SMat;
use super::element::SVec;
use std::marker::PhantomData;

use super::mut_tensor::MutTensor;
use super::view::IsTensorLike;
use super::view::IsTensorView;
use super::view::TensorView;

#[derive(Debug, Clone)]
pub struct ArcTensor<
    const HYBER_RANK: usize,
    const SRANK: usize,
    const DRANK: usize,
    Scalar: IsTensorScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, BATCHES, ROWS, COLS> + 'static,
    const BATCHES: usize,
    const ROWS: usize,
    const COLS: usize,
> where
    ndarray::Dim<[ndarray::Ix; DRANK]>: Dimension,
{
    pub array: ndarray::ArcArray<STensor, ndarray::Dim<[ndarray::Ix; DRANK]>>,
    phantom: PhantomData<(Scalar, STensor)>,
}

pub type ArcTensorX<const DRANK: usize, Scalar> =
    ArcTensor<DRANK, 0, DRANK, Scalar, Scalar, 1, 1, 1>;

pub type ArcTensorXB<
    const HYBER_RANK: usize,
    const SRANK: usize,
    const DRANK: usize,
    Scalar,
    const B: usize,
> = ArcTensor<HYBER_RANK, SRANK, DRANK, Scalar, BatchScalar<Scalar, B>, B, 1, 1>;

pub type ArcTensorXR<
    const HYBER_RANK: usize,
    const SRANK: usize,
    const DRANK: usize,
    Scalar,
    const R: usize,
> = ArcTensor<HYBER_RANK, SRANK, DRANK, Scalar, SVec<Scalar, R>, 1, R, 1>;

pub type ArcTensorXBR<
    const HYBER_RANK: usize,
    const SRANK: usize,
    const DRANK: usize,
    Scalar,
    const B: usize,
    const R: usize,
> = ArcTensor<HYBER_RANK, SRANK, DRANK, Scalar, BatchVec<Scalar, B, R>, B, R, 1>;

pub type ArcTensorXRC<
    const HYBER_RANK: usize,
    const SRANK: usize,
    const DRANK: usize,
    Scalar,
    const R: usize,
    const C: usize,
> = ArcTensor<HYBER_RANK, SRANK, DRANK, Scalar, SMat<Scalar, R, C>, 1, R, C>;

pub type ArcTensorXBRC<
    const HYBER_RANK: usize,
    const SRANK: usize,
    const DRANK: usize,
    Scalar,
    const B: usize,
    const R: usize,
    const C: usize,
> = ArcTensor<HYBER_RANK, SRANK, DRANK, Scalar, BatchMat<Scalar, B, R, C>, B, R, C>;

// rank 1 - D0
pub type ArcTensorD<const DRANK: usize, Scalar> = ArcTensorX<DRANK, Scalar>;

// rank 2
// D0 x D1
pub type ArcTensorDD<Scalar> = ArcTensorX<2, Scalar>;

// rank 2
// D0 x [B]
pub type ArcTensorDB<Scalar, const B: usize> = ArcTensorXB<2, 1, 1, Scalar, B>;

// rank 2
// D0 x [R]
pub type ArcTensorDR<Scalar, const R: usize> = ArcTensorXR<2, 1, 1, Scalar, R>;

// rank 3
// D0 x D1 x D2
pub type ArcTensorRRR<Scalar> = ArcTensorX<3, Scalar>;

// rank 3
//  D0 x D1 x [B]
pub type ArcTensorDDB<Scalar, const B: usize> = ArcTensorXB<3, 1, 2, Scalar, B>;

// rank 3
// D0 x D1 x [R]
pub type ArcTensorDDR<Scalar, const R: usize> = ArcTensorXR<3, 1, 2, Scalar, R>;

// rank 3
// D0 x [B x R]
pub type ArcTensorBRD<Scalar, const B: usize, const R: usize> = ArcTensorXBR<3, 2, 1, Scalar, B, R>;

// rank 3
// D0 x [R x C]
pub type ArcTensorDRC<Scalar, const R: usize, const C: usize> = ArcTensorXRC<3, 2, 1, Scalar, R, C>;

// rank 4
// srank 0, drank 4
pub type ArcTensorDDDD<Scalar> = ArcTensorX<4, Scalar>;

// rank 4
// D0 x D1 x D2 x [B]
pub type ArcTensorDDDB<Scalar, const B: usize> = ArcTensorXB<4, 1, 3, Scalar, B>;

// rank 4
// D0 x D1 x D2 x [R]
pub type ArcTensorDDDR<Scalar, const R: usize> = ArcTensorXR<4, 1, 3, Scalar, R>;

// rank 4
// D0 x D1 x [B x R]
pub type ArcTensorDDRB<Scalar, const B: usize, const R: usize> =
    ArcTensorXBR<4, 2, 2, Scalar, B, R>;

// rank 4
// D0 x D1 x [R x C]
pub type ArcTensorDDRC<Scalar, const R: usize, const C: usize> =
    ArcTensorXRC<4, 2, 2, Scalar, R, C>;

// rank 4
// D0 x [B x R x C]
pub type ArcTensorDRCB<Scalar, const B: usize, const R: usize, const C: usize> =
    ArcTensorXBRC<4, 3, 1, Scalar, B, R, C>;

macro_rules! arc_tensor_is_tensor_view {
    ($hyper_rank:literal, $srank:literal,$drank:literal) => {


        impl<
        'a,
                Scalar: IsTensorScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank, BATCHES, ROWS, COLS> + 'static,
                const BATCHES: usize,
                const ROWS: usize,
                const COLS: usize,
            > IsTensorLike<'a, $hyper_rank, $srank, $drank, Scalar, STensor, BATCHES, ROWS, COLS>
            for ArcTensor<$hyper_rank, $srank, $drank, Scalar, STensor, BATCHES, ROWS, COLS>
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

        impl<'a, Scalar: IsTensorScalar+ 'static,
                STensor: IsStaticTensor<Scalar, $srank, BATCHES, ROWS, COLS> + 'static,
                const BATCHES: usize,
                const ROWS: usize,
                const COLS: usize,
            >
            ArcTensor<$hyper_rank, $srank, $drank, Scalar, STensor, BATCHES, ROWS, COLS>
        {
           pub  fn from_mut_tensor(
                tensor: MutTensor<$hyper_rank, $srank, $drank, Scalar, STensor, BATCHES, ROWS, COLS>,
            ) -> Self {

                Self {
                    array: tensor.mut_array.into(),
                    phantom: PhantomData {},
                }
            }

            pub   fn from_shape(size: [usize; $drank]) -> Self {
                Self::from_mut_tensor(MutTensor::<$hyper_rank, $srank, $drank, Scalar, STensor, BATCHES, ROWS, COLS>::from_shape(size))
            }

            pub fn from_shape_and_val(
                shape: [usize; $drank],
                val:STensor,
            ) -> Self {
                Self::from_mut_tensor(
                    MutTensor::<$hyper_rank, $srank,$drank, Scalar, STensor, BATCHES,ROWS, COLS>::from_shape_and_val(shape, val),
                )
            }

            pub    fn view<'b: 'a>(&'b self) -> TensorView<'a, $hyper_rank, $srank,$drank, Scalar, STensor, BATCHES, ROWS, COLS> {
                TensorView::<'a,  $hyper_rank, $srank,$drank, Scalar, STensor, BATCHES, ROWS, COLS>::new(
                    self.array.view())
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
                Self::from_mut_tensor(
                    MutTensor::<$hyper_rank, $srank,$drank, Scalar, STensor, BATCHES,ROWS, COLS>::from_map(view, op),
                )
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
            F: FnMut(&OtherSTensor, &OtherSTensor2) -> STensor
        >(
            view: &'b V,
            view2: &'b V2,
            op: F,
        )
        -> Self
        where
            ndarray::Dim<[ndarray::Ix; OTHER_HRANK]>: ndarray::Dimension,
            ndarray::Dim<[ndarray::Ix; OTHER_HRANK2]>: ndarray::Dimension{
                Self::from_mut_tensor(
                    MutTensor::<$hyper_rank, $srank,$drank, Scalar, STensor, BATCHES,ROWS, COLS>::from_map2(view,view2, op),
                )
            }

        }
    };
}

pub trait ArcInnerToMat<
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
    fn inner_vec_to_mat(
        self,
    ) -> ArcTensorXRC<HYBER_RANK_PLUS1, SRANK_PLUS1, DRANK, Scalar, ROWS, 1>;
}

pub trait ArcInnerToVec<
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
    fn inner_scalar_to_vec(self) -> ArcTensorXR<HYBER_RANK_PLUS1, SRANK_PLUS1, DRANK, Scalar, 1>;
}

impl<Scalar: IsTensorScalar + 'static, const ROWS: usize> ArcInnerToMat<3, 1, 2, 4, 2, Scalar, ROWS>
    for ArcTensorXR<3, 1, 2, Scalar, ROWS>
{
    fn inner_vec_to_mat(self) -> ArcTensorXRC<4, 2, 2, Scalar, ROWS, 1> {
        ArcTensorXRC::<4, 2, 2, Scalar, ROWS, 1> {
            array: self.array,
            phantom: PhantomData,
        }
    }
}

impl<Scalar: IsTensorScalar + 'static> ArcInnerToVec<2, 0, 2, 3, 1, Scalar>
    for ArcTensorX<2, Scalar>
{
    fn inner_scalar_to_vec(self) -> ArcTensorXR<3, 1, 2, Scalar, 1> {
        ArcTensorXR::<3, 1, 2, Scalar, 1> {
            array: self.array.map(|x| SVec::<Scalar, 1>::new(*x)).to_shared(),
            phantom: PhantomData,
        }
    }
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

    #[test]
    fn from_mut_tensor() {
        use super::*;

        use crate::tensor::element::P1F32;
        use crate::tensor::mut_tensor::MutTensorDDDR;
        use crate::tensor::mut_tensor::MutTensorDDR;
        use crate::tensor::mut_tensor::MutTensorDR;

        {
            let shape = [4];
            let mut_img = MutTensorDR::from_shape_and_val(shape, P1F32::new(0.5f32));
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
            let mut_img = MutTensorDDR::from_shape_and_val(shape, P1F32::new(0.5f32));
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
            let mut_img = MutTensorDDDR::from_shape_and_val(shape, P1F32::new(0.5f32));
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

        use crate::tensor::element::P1F32;
        use crate::tensor::mut_tensor::MutTensorDDDR;
        use crate::tensor::mut_tensor::MutTensorDDR;
        use crate::tensor::mut_tensor::MutTensorDR;
        {
            let shape = [4];
            let mut_img = MutTensorDR::from_shape_and_val(shape, P1F32::new(0.5f32));
            let img = ArcTensorDR::from_mut_tensor(mut_img);

            let mut img2 = img.clone();
            // assert_eq!(Arc::strong_count(&img.buffer), 2);
            // assert_eq!(Arc::strong_count(&img2.buffer), 2);

            assert_eq!(
                img.view().elem_view().as_slice().unwrap(),
                img2.view().elem_view().as_slice().unwrap()
            );

            let mut_img2 = img2.to_mut_image();
            //  assert_eq!(Arc::strong_count(&img2.buffer), 2);
            assert_ne!(
                mut_img2.view().elem_view().as_slice().unwrap().as_ptr(),
                img2.view().elem_view().as_slice().unwrap().as_ptr()
            );

            img2 = img.clone();
            // assert_eq!(Arc::strong_count(&img.buffer), 2);
            // assert_eq!(Arc::strong_count(&img2.buffer), 2);
        }
        {
            let shape = [4, 6];
            let mut_img = MutTensorDDR::from_shape_and_val(shape, P1F32::new(0.5f32));
            let img = ArcTensorDDR::from_mut_tensor(mut_img);

            let mut img2 = img.clone();
            // assert_eq!(Arc::strong_count(&img.buffer), 2);
            // assert_eq!(Arc::strong_count(&img2.buffer), 2);

            let mut_img2 = img2.to_mut_image();
            //  assert_eq!(Arc::strong_count(&img2.buffer), 2);
            assert_ne!(
                mut_img2.view().elem_view().as_slice().unwrap().as_ptr(),
                img2.view().elem_view().as_slice().unwrap().as_ptr()
            );

            img2 = img.clone();
            // assert_eq!(Arc::strong_count(&img.buffer), 2);
            // assert_eq!(Arc::strong_count(&img2.buffer), 2);
        }
        {
            let shape = [4, 6, 7];
            let mut_img = MutTensorDDDR::from_shape_and_val(shape, P1F32::new(0.5f32));
            let img = ArcTensorDDDR::from_mut_tensor(mut_img);

            let mut img2 = img.clone();
            // assert_eq!(Arc::strong_count(&img.buffer), 2);
            // assert_eq!(Arc::strong_count(&img2.buffer), 2);

            let mut_img2 = img2.to_mut_image();
            //  assert_eq!(Arc::strong_count(&img2.buffer), 2);
            assert_ne!(
                mut_img2.view().elem_view().as_slice().unwrap().as_ptr(),
                img2.view().elem_view().as_slice().unwrap().as_ptr()
            );

            img2 = img.clone();
            // assert_eq!(Arc::strong_count(&img.buffer), 2);
            // assert_eq!(Arc::strong_count(&img2.buffer), 2);
        }
    }

    #[test]
    fn multi_threading() {
        use crate::tensor::arc_tensor::ArcTensorDDRC;
        use crate::tensor::element::P3U16;
        use crate::tensor::mut_tensor::MutTensorDDRC;
        use std::thread;

        let shape = [4, 6];
        let mut_img = MutTensorDDRC::from_shape_and_val(shape, P3U16::new(10, 20, 300));
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
