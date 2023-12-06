use std::fmt::Debug;

use simba::simd::AutoSimd;
pub use typenum::generic_const_mappings::Const;

/// Compile time known shape with 0 dimensions
pub type CShape0 = ();
/// Compile time known shape with 1 dimensions
pub type CShape1<M> = (M,);
/// Compile time known shape with 2 dimensions
pub type CShape2<M, N> = (M, N);
/// Compile time known shape with 3 dimensions
pub type CShape3<M, N, O> = (M, N, O);

pub trait IsCShape<const RANK: usize> {}

impl IsCShape<0> for CShape0 {}
impl<A> IsCShape<1> for CShape1<A> {}
impl<A, B> IsCShape<2> for CShape2<A, B> {}
impl<A, B, C> IsCShape<3> for CShape3<A, B, C> {}

pub trait IsTensorScalarLike: Copy + Clone + Debug {
    fn number_category() -> NumberCategory;
}

pub trait IsTensorScalar: IsTensorScalarLike + num_traits::Zero + nalgebra::Scalar {}

impl IsTensorScalarLike for f32 {
    fn number_category() -> NumberCategory {
        NumberCategory::Real
    }
}
impl IsTensorScalarLike for f64 {
    fn number_category() -> NumberCategory {
        NumberCategory::Real
    }
}
impl IsTensorScalarLike for u8 {
    fn number_category() -> NumberCategory {
        NumberCategory::Unsigned
    }
}
impl IsTensorScalarLike for u16 {
    fn number_category() -> NumberCategory {
        NumberCategory::Unsigned
    }
}
impl IsTensorScalarLike for u32 {
    fn number_category() -> NumberCategory {
        NumberCategory::Unsigned
    }
}
impl IsTensorScalarLike for u64 {
    fn number_category() -> NumberCategory {
        NumberCategory::Unsigned
    }
}
impl IsTensorScalarLike for i8 {
    fn number_category() -> NumberCategory {
        NumberCategory::Signed
    }
}
impl IsTensorScalarLike for i16 {
    fn number_category() -> NumberCategory {
        NumberCategory::Signed
    }
}
impl IsTensorScalarLike for i32 {
    fn number_category() -> NumberCategory {
        NumberCategory::Signed
    }
}
impl IsTensorScalarLike for i64 {
    fn number_category() -> NumberCategory {
        NumberCategory::Signed
    }
}

impl IsTensorScalar for f32 {}
impl IsTensorScalar for f64 {}
impl IsTensorScalar for u8 {}
impl IsTensorScalar for u16 {}
impl IsTensorScalar for u32 {}
impl IsTensorScalar for u64 {}
impl IsTensorScalar for i8 {}
impl IsTensorScalar for i16 {}
impl IsTensorScalar for i32 {}
impl IsTensorScalar for i64 {}

pub trait IsBatchScalar: IsTensorScalarLike {}

impl<Scalar: IsTensorScalar + 'static, const BATCHES: usize> IsTensorScalarLike
    for AutoSimd<[Scalar; BATCHES]>
{
    fn number_category() -> NumberCategory {
        Scalar::number_category()
    }
}
impl<Scalar: IsTensorScalar + 'static, const BATCHES: usize> IsBatchScalar
    for AutoSimd<[Scalar; BATCHES]>
{
}

pub type SVec<ScalarLike, const ROWS: usize> = nalgebra::SVector<ScalarLike, ROWS>;
pub type SMat<ScalarLike, const ROWS: usize, const COLS: usize> =
    nalgebra::SMatrix<ScalarLike, ROWS, COLS>;

pub type P1U8 = SVec<u8, 1>;
pub type P1U16 = SVec<u16, 1>;
pub type P1F32 = SVec<f32, 1>;
pub type P3U8 = SVec<u8, 3>;
pub type P3U16 = SVec<u16, 3>;
pub type P3F32 = SVec<f32, 3>;
pub type P4U8 = SVec<u8, 4>;
pub type P4U16 = SVec<u16, 4>;
pub type P4F32 = SVec<f32, 4>;

pub type BatchScalar<ScalarLike, const BATCHES: usize> = AutoSimd<[ScalarLike; BATCHES]>;
pub type BatchVec<ScalarLike, const ROWS: usize, const BATCHES: usize> =
    nalgebra::SVector<AutoSimd<[ScalarLike; BATCHES]>, ROWS>;
pub type BatchMat<ScalarLike, const ROWS: usize, const COLS: usize, const BATCHES: usize> =
    nalgebra::SMatrix<AutoSimd<[ScalarLike; BATCHES]>, ROWS, COLS>;

pub trait IsStaticTensor<
    Scalar: IsTensorScalar + 'static,
    const SRANK: usize,
    const ROWS: usize,
    const COLS: usize,
    const BATCHES: usize,
>: Copy + Clone + Debug
{
    type CShape: IsCShape<SRANK>;

    fn zero() -> Self;

    fn number_category() -> NumberCategory {
        Scalar::number_category()
    }

    fn scalar(&self, idx: [usize; SRANK]) -> &Scalar;

    fn rank(&self) -> usize {
        SRANK
    }

    fn num_batches(&self) -> usize {
        BATCHES
    }

    fn num_rows(&self) -> usize {
        ROWS
    }

    fn num_cols(&self) -> usize {
        COLS
    }

    fn sdims() -> [usize; SRANK];

    fn num_scalars() -> usize {
        BATCHES * ROWS * COLS
    }

    fn strides() -> [usize; SRANK];
}

// Rank 0 tensors
//
// a scalar
impl<Scalar: IsTensorScalar + 'static> IsStaticTensor<Scalar, 0, 1, 1, 1> for Scalar {
    type CShape = ();

    fn zero() -> Self {
        Scalar::zero()
    }

    fn scalar(&self, _idx: [usize; 0]) -> &Scalar {
        self
    }

    fn sdims() -> [usize; 0] {
        []
    }

    fn strides() -> [usize; 0] {
        []
    }
}

// RANK 1 TENSORS
//
// A batch ofBatchScalar scalars
impl<Scalar: IsTensorScalar + 'static, const BATCHES: usize>
    IsStaticTensor<Scalar, 1, 1, 1, BATCHES> for BatchScalar<Scalar, BATCHES>
{
    type CShape = CShape1<Const<BATCHES>>;

    fn zero() -> Self {
        todo!()
    }

    fn scalar(&self, idx: [usize; 1]) -> &Scalar {
        &self.0[idx[0]]
    }

    fn sdims() -> [usize; 1] {
        [BATCHES]
    }

    fn strides() -> [usize; 1] {
        [1]
    }
}

// A vector
impl<Scalar: IsTensorScalar + 'static, const ROWS: usize> IsStaticTensor<Scalar, 1, ROWS, 1, 1>
    for SVec<Scalar, ROWS>
{
    type CShape = CShape1<Const<ROWS>>;

    fn zero() -> Self {
        Self::zeros()
    }

    fn scalar(&self, idx: [usize; 1]) -> &Scalar {
        &self[idx[0]]
    }

    fn sdims() -> [usize; 1] {
        [ROWS]
    }

    fn strides() -> [usize; 1] {
        [1]
    }
}

// RANK 2 TENSORS
//
// A batch of vectors
impl<Scalar: IsTensorScalar + 'static, const BATCHES: usize, const ROWS: usize>
    IsStaticTensor<Scalar, 2, ROWS, 1, BATCHES> for SVec<AutoSimd<[Scalar; BATCHES]>, ROWS>
{
    type CShape = CShape2<Const<BATCHES>, Const<ROWS>>;

    fn zero() -> Self {
        //Self::zeros()
        todo!()
    }

    fn scalar(&self, idx: [usize; 2]) -> &Scalar {
        &self[idx[1]].0[idx[0]]
    }

    fn sdims() -> [usize; 2] {
        [BATCHES, ROWS]
    }

    fn strides() -> [usize; 2] {
        [1, BATCHES]
    }
}

// a matrix
impl<Scalar: IsTensorScalar + 'static, const ROWS: usize, const COLS: usize>
    IsStaticTensor<Scalar, 2, ROWS, COLS, 1> for SMat<Scalar, ROWS, COLS>
{
    type CShape = CShape2<Const<ROWS>, Const<COLS>>;

    fn zero() -> Self {
        Self::zeros()
    }

    fn scalar(&self, idx: [usize; 2]) -> &Scalar {
        &self[(idx[0], idx[1])]
    }

    fn sdims() -> [usize; 2] {
        [ROWS, COLS]
    }

    fn strides() -> [usize; 2] {
        [1, ROWS]
    }
}

// RANK 3 TENSORS

// a batch of matrices
impl<
        Scalar: IsTensorScalar + 'static,
        const BATCHES: usize,
        const ROWS: usize,
        const COLS: usize,
    > IsStaticTensor<Scalar, 3, ROWS, COLS, BATCHES>
    for SMat<AutoSimd<[Scalar; BATCHES]>, ROWS, COLS>
{
    type CShape = CShape3<Const<BATCHES>, Const<ROWS>, Const<COLS>>;
    fn zero() -> Self {
        todo!()
    }
    fn scalar(&self, idx: [usize; 3]) -> &Scalar {
        &self[(idx[1], idx[2])].0[idx[0]]
    }

    fn sdims() -> [usize; 3] {
        [ROWS, COLS, BATCHES]
    }

    fn strides() -> [usize; 3] {
        [1, BATCHES, BATCHES * ROWS]
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NumberCategory {
    Real,
    Unsigned,
    Signed,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct STensorFormat {
    pub number_category: NumberCategory, // unsigned otherwise
    pub num_bytes_per_scalar: usize,
    pub num_batches: usize,
    pub num_rows: usize,
    pub num_cols: usize,
}

impl STensorFormat {
    pub fn new<
        Scalar: IsTensorScalar + 'static,
        const ROWS: usize,
        const COLS: usize,
        const BATCHES: usize,
    >() -> Self {
        STensorFormat {
            number_category: Scalar::number_category(),
            num_rows: ROWS,
            num_cols: COLS,
            num_batches: BATCHES,
            num_bytes_per_scalar: std::mem::size_of::<Scalar>(),
        }
    }

    pub fn num_bytes(&self) -> usize {
        self.num_batches * self.num_rows * self.num_cols * self.num_bytes_per_scalar
    }
}
