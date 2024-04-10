use crate::linalg::SMat;
use crate::linalg::SVec;
use crate::prelude::*;
use crate::tensor::mut_tensor::InnerScalarToVec;
use crate::tensor::mut_tensor::InnerVecToMat;
use crate::tensor::MutTensor;
use crate::tensor::TensorView;
use ndarray::Dimension;

use std::marker::PhantomData;

/// Arc tensor - a tensor with shared ownership
///
/// See TensorView for more details of the tensor structure
#[derive(Debug, Clone)]
pub struct ArcTensor<
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar: IsCoreScalar + 'static,
    STensor: IsStaticTensor<Scalar, SRANK, ROWS, COLS> + 'static,
    const ROWS: usize,
    const COLS: usize,
> where
    ndarray::Dim<[ndarray::Ix; DRANK]>: Dimension,
{
    /// ndarray of tensors with shape [D1, D2, ...]
    pub array: ndarray::ArcArray<STensor, ndarray::Dim<[ndarray::Ix; DRANK]>>,
    phantom: PhantomData<(Scalar, STensor)>,
}

/// rank-1 tensor of scalars
pub type ArcTensorX<const DRANK: usize, Scalar> = ArcTensor<DRANK, DRANK, 0, Scalar, Scalar, 1, 1>;

/// rank-2 tensor of vectors with shape R
pub type ArcTensorXR<
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
> = ArcTensor<TOTAL_RANK, DRANK, SRANK, Scalar, SVec<Scalar, R>, R, 1>;

/// rank-2 tensor of matrices with shape [R x C]
pub type ArcTensorXRC<
    const TOTAL_RANK: usize,
    const DRANK: usize,
    const SRANK: usize,
    Scalar,
    const R: usize,
    const C: usize,
> = ArcTensor<TOTAL_RANK, DRANK, SRANK, Scalar, SMat<Scalar, R, C>, R, C>;

/// rank-1 tensor of scalars with shape D0
pub type ArcTensorD<const DRANK: usize, Scalar> = ArcTensorX<DRANK, Scalar>;

/// rank-2 tensor of scalars with shape [D0 x D1]
pub type ArcTensorDD<Scalar> = ArcTensorX<2, Scalar>;

/// rank-2 tensor of vectors with shape [D0 x R]
pub type ArcTensorDR<Scalar, const R: usize> = ArcTensorXR<2, 1, 1, Scalar, R>;

/// rank-3 tensor of scalars with shape [D0 x D1 x D2]
pub type ArcTensorRRR<Scalar> = ArcTensorX<3, Scalar>;

/// rank-3 tensor of vectors with shape [D0 x D1 x R]
pub type ArcTensorDDR<Scalar, const R: usize> = ArcTensorXR<3, 2, 1, Scalar, R>;

/// rank-3 tensor of matrices with shape [D0 x R x C]
pub type ArcTensorDRC<Scalar, const R: usize, const C: usize> = ArcTensorXRC<3, 1, 2, Scalar, R, C>;

/// rank-4 tensor of scalars with shape [D0 x D1 x D2 x D3]
pub type ArcTensorDDDD<Scalar> = ArcTensorX<4, Scalar>;

/// rank-4 tensor of vectors with shape [D0 x D1 x D2 x R]
pub type ArcTensorDDDR<Scalar, const R: usize> = ArcTensorXR<4, 3, 1, Scalar, R>;

/// rank-4 tensor of matrices with shape [D0 x R x C x B]
pub type ArcTensorDDRC<Scalar, const R: usize, const C: usize> =
    ArcTensorXRC<4, 2, 2, Scalar, R, C>;

macro_rules! arc_tensor_is_tensor_view {
    ($scalar_rank:literal, $srank:literal,$drank:literal) => {


        impl<
            'a,
            Scalar: IsCoreScalar + 'static,
            STensor: IsStaticTensor<Scalar, $srank,  ROWS, COLS> + 'static,
            const ROWS: usize,
            const COLS: usize,
        > IsTensorLike<
            'a, $scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS
        > for ArcTensor<$scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS>
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
            ) -> MutTensor<$scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS> {
                MutTensor {
                    mut_array: self.elem_view().to_owned(),
                    phantom: PhantomData::default(),
                }
            }
        }

            impl<
            'a,
            Scalar: IsCoreScalar+ 'static,
            STensor: IsStaticTensor<Scalar, $srank,  ROWS, COLS> +  'static,
            const ROWS: usize,
            const COLS: usize,
        >
            ArcTensor<$scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS>
        {
              /// create a new tensor from a shape - all elements are zero
              pub fn from_shape(size: [usize; $drank]) -> Self {
                Self::from_mut_tensor(
                    MutTensor::<
                        $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS,
                    >::from_shape(size))
            }

              /// create a new tensor from a binary operation applied to two tensor views
              pub fn from_map2<
              'b,
              const OTHER_HRANK: usize, const OTHER_SRANK: usize,
              OtherScalar: IsCoreScalar + 'static,
              OtherSTensor: IsStaticTensor<
                  OtherScalar, OTHER_SRANK, OTHER_ROWS, OTHER_COLS
              > + 'static,
             const OTHER_ROWS: usize, const OTHER_COLS: usize,
              V : IsTensorView::<
                      'b,
                      OTHER_HRANK, $drank, OTHER_SRANK,
                      OtherScalar, OtherSTensor,
                      OTHER_ROWS, OTHER_COLS,
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
                  OTHER_ROWS2, OTHER_COLS2,
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
                      $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS,
                  >::from_map2(view,view2, op),
              )
          }
        }

        impl<
            'a,
            Scalar: IsCoreScalar+ 'static,
            STensor: IsStaticTensor<Scalar, $srank,  ROWS, COLS> + 'static,
            const ROWS: usize,
            const COLS: usize,
        >
            ArcTensor<$scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS>
        {

            /// create a new tensor from a tensor view
            pub fn make_copy_from(
                v: &TensorView<$scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS>
            ) -> Self
            {
                Self::from_mut_tensor(IsTensorLike::to_mut_tensor(v))
            }

            /// create a new tensor from a mutable tensor
           pub fn from_mut_tensor(
                tensor:
                MutTensor<$scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS>,
            ) -> Self {
                Self {
                    array: tensor.mut_array.into(),
                    phantom: PhantomData {},
                }
            }

            /// create a new tensor from a shape and a value
            pub fn from_shape_and_val(
                shape: [usize; $drank],
                val:STensor,
            ) -> Self {
                Self::from_mut_tensor(
                    MutTensor::<
                        $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS
                    >::from_shape_and_val(shape, val),
                )
            }

            /// return a tensor view
            pub fn view<'b: 'a>(&'b self)
                -> TensorView<
                    'a, $scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS>
            {
                TensorView::<
                    'a, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS
                >::new(
                    self.array.view()
                )
            }

            /// create a new tensor from a unary operation applied to a tensor view
            pub fn from_map<
                'b,
                const OTHER_HRANK: usize, const OTHER_SRANK: usize,
                OtherScalar: IsCoreScalar+ 'static,
                OtherSTensor: IsStaticTensor<
                    OtherScalar, OTHER_SRANK, OTHER_ROWS, OTHER_COLS
                > + 'static,
                const OTHER_ROWS: usize, const OTHER_COLS: usize,
                V : IsTensorView::<
                    'b,
                    OTHER_HRANK, $drank, OTHER_SRANK,
                    OtherScalar, OtherSTensor,
                     OTHER_ROWS, OTHER_COLS,
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
                        $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS
                    >::from_map(view, op),
                )
            }
        }
    };
}

impl<Scalar: IsCoreScalar + 'static, const ROWS: usize> InnerVecToMat<3, 1, 2, 4, 2, Scalar, ROWS>
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

impl<Scalar: IsCoreScalar + 'static> InnerScalarToVec<2, 0, 2, 3, 1, Scalar>
    for ArcTensorX<2, Scalar>
{
    fn inner_scalar_to_vec(self) -> ArcTensorXR<3, 2, 1, Scalar, 1> {
        ArcTensorXR::<3, 2, 1, Scalar, 1> {
            array: self
                .array
                .map(|x| SVec::<Scalar, 1>::new(x.clone()))
                .to_shared(),
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
arc_tensor_is_tensor_view!(5, 0, 5);
arc_tensor_is_tensor_view!(5, 1, 4);
arc_tensor_is_tensor_view!(5, 2, 3);

#[test]
fn arc_tensor_tests() {
    //from_mut_tensor

    use crate::tensor::mut_tensor::MutTensorDDDR;
    use crate::tensor::mut_tensor::MutTensorDDR;
    use crate::tensor::mut_tensor::MutTensorDR;

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

    // shared_ownership
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

    // multi_threading
    use crate::tensor::arc_tensor::ArcTensorDDRC;
    use crate::tensor::mut_tensor::MutTensorDDRC;
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
