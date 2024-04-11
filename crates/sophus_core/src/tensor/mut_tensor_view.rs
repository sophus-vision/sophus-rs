use crate::prelude::*;
use crate::tensor::MutTensor;
use crate::tensor::TensorView;
use concat_arrays::concat_arrays;
use std::marker::PhantomData;

/// Mutable tensor view
///
/// See TensorView for more details of the tensor structure
#[derive(Debug, PartialEq, Eq)]
pub struct MutTensorView<
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
    /// mutable ndarray view of the static tensors with shape [D1, D2, ...]
    pub elem_view_mut: ndarray::ArrayViewMut<'a, STensor, ndarray::Dim<[ndarray::Ix; DRANK]>>,
    /// mutable ndarray view of the scalars with shape [D1, D2, ..., S0, S1, ...]
    pub scalar_view_mut: ndarray::ArrayViewMut<'a, Scalar, ndarray::Dim<[ndarray::Ix; TOTAL_RANK]>>,
}

/// A mutable tensor like object
pub trait IsMutTensorLike<
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
    /// mutable ndarray view of the static tensors with shape [D1, D2, ...]
    fn elem_view_mut<'b: 'a>(
        &'b mut self,
    ) -> ndarray::ArrayViewMut<'a, STensor, ndarray::Dim<[ndarray::Ix; DRANK]>>;

    /// mutable ndarray view of the scalars with shape [D1, D2, ..., S0, S1, ...]
    fn scalar_view_mut<'b: 'a>(
        &'b mut self,
    ) -> ndarray::ArrayViewMut<'a, Scalar, ndarray::Dim<[ndarray::Ix; TOTAL_RANK]>>;

    /// mutable reference to the static tensor at index idx
    fn get_mut(&'a mut self, idx: [usize; DRANK]) -> &mut STensor;
}

macro_rules! mut_view_is_view {
    ($scalar_rank:literal, $srank:literal, $drank:literal) => {

        impl<
                'a,
                Scalar: IsCoreScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank,  ROWS, COLS> + 'static,
                const ROWS: usize,
                const COLS: usize,
            > MutTensorView<'a, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS>
        {

            /// Returns a tensor view
            pub fn view(
                & self,
            ) -> TensorView<'_, $scalar_rank, $drank, $srank, Scalar, STensor, ROWS, COLS>
            {
               let v =  TensorView {
                    elem_view: self.elem_view_mut.view(),
                    scalar_view: self.scalar_view_mut.view(),
                };
                v
            }

            /// new mutable tensor view from a mutable ndarray of static tensors
            pub fn new(
                elem_view_mut: ndarray::ArrayViewMut<
                    'a,
                    STensor,
                    ndarray::Dim<[ndarray::Ix; $drank]>,
                >,
            ) -> Self {
                let dims: [usize; $drank] = elem_view_mut.shape().try_into().unwrap();
                #[allow(clippy::drop_non_drop)]
                let shape: [usize; $scalar_rank] = concat_arrays!(dims, STensor::sdims());

                let dstrides: [isize; $drank] = elem_view_mut.strides().try_into().unwrap();
                let mut dstrides: [usize; $drank] = dstrides.map(|x| x as usize);
                let num_scalars = STensor::num_scalars();
                for d in dstrides.iter_mut() {
                    *d *= num_scalars;
                }
                #[allow(clippy::drop_non_drop)]
                let strides = concat_arrays!(dstrides, STensor::strides());

                let ptr = elem_view_mut.as_ptr() as *mut Scalar;
                use ndarray::ShapeBuilder;
                assert_eq!(std::mem::size_of::<STensor>(),
                    std::mem::size_of::<Scalar>() * ROWS * COLS
                );

                let scalar_view_mut =
                    unsafe { ndarray::ArrayViewMut::from_shape_ptr(shape.strides(strides), ptr) };

                Self {
                    elem_view_mut,
                    scalar_view_mut,
                }
            }

            /// get mutable reference to scalar at index idx
            pub fn mut_scalar(&'a mut self, idx: [usize; $scalar_rank]) -> &mut Scalar{
                &mut self.scalar_view_mut[idx]
            }

            /// fills self using a unary operator applied to the elements of another tensor
            pub fn map<
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
                V : IsTensorView::<
                    'b,
                    OTHER_HRANK,
                    $drank,
                    OTHER_SRANK,
                    OtherScalar,
                    OtherSTensor,
                    OTHER_ROWS,
                    OTHER_COLS,
                >,
                F: FnMut(&mut STensor, &OtherSTensor)
            >(
                &'a mut self,
                view: &'b V,
                op: F,
            ) where
                ndarray::Dim<[ndarray::Ix; OTHER_HRANK]>: ndarray::Dimension
            {
                self.elem_view_mut.zip_mut_with(&view.elem_view(),op);
            }
        }




        impl<
        'a,
                Scalar: IsCoreScalar + 'static,
                STensor: IsStaticTensor<Scalar, $srank,  ROWS, COLS> + 'static,
                const ROWS: usize,
                const COLS: usize,
            > IsTensorLike<'a, $scalar_rank, $drank, $srank, Scalar, STensor,  ROWS, COLS>
            for MutTensorView<
                'a,
                $scalar_rank,
                $drank,
                $srank,
                Scalar,
                STensor,
                ROWS,
                COLS
>        {
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
                    mut_array: self.view().elem_view.to_owned(),
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
                $scalar_rank,
                $drank, $srank,
                Scalar, STensor,
                ROWS,
                COLS            >
            for MutTensorView<'a,
                $scalar_rank,
                $drank,
                $srank,
                Scalar,
                STensor,
                ROWS,
                COLS,
            >
        {
            fn elem_view_mut<'b:'a>(
                &'b mut self,
            ) -> ndarray::ArrayViewMut<'a, STensor, ndarray::Dim<[ndarray::Ix; $drank]>>{
                self.elem_view_mut.view_mut()
            }

            fn scalar_view_mut<'b:'a>(
                &'b  mut self,
            ) -> ndarray::ArrayViewMut<'a, Scalar, ndarray::Dim<[ndarray::Ix; $scalar_rank]>>{
                self.scalar_view_mut.view_mut()
            }

            fn get_mut(&'a mut self, idx: [usize; $drank]) -> &mut STensor{
                &mut self.elem_view_mut[idx]
            }


        }
    };
}

mut_view_is_view!(1, 0, 1);
mut_view_is_view!(2, 0, 2);
mut_view_is_view!(2, 1, 1);
mut_view_is_view!(3, 0, 3);
mut_view_is_view!(3, 1, 2);
mut_view_is_view!(3, 2, 1);
mut_view_is_view!(4, 0, 4);
mut_view_is_view!(4, 1, 3);
mut_view_is_view!(4, 2, 2);
mut_view_is_view!(5, 0, 5);
mut_view_is_view!(5, 1, 4);
mut_view_is_view!(5, 2, 3);
