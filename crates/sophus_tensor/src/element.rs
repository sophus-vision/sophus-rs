use simba::simd::AutoSimd;
use std::fmt::Debug;

pub use typenum::generic_const_mappings::Const;

/// Number category
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NumberCategory {
    /// Real number such as f32 or f64
    Real,
    /// Unsigned integer such as u8, u16, u32, or u64
    Unsigned,
    /// Signed integer such as i8, i16, i32, or i64
    Signed,
}

/// Trait for scalar and batch scalar types
pub trait IsTensorScalarLike: Copy + Clone + Debug {
    /// Get the number category
    fn number_category() -> NumberCategory;
}

/// Trait for scalar types
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

/// Trait for batch scalar types
pub trait IsBatchScalar: IsTensorScalarLike {}

impl<Scalar: IsTensorScalar + 'static, const BATCH_SIZE: usize> IsTensorScalarLike
    for AutoSimd<[Scalar; BATCH_SIZE]>
{
    fn number_category() -> NumberCategory {
        Scalar::number_category()
    }
}
impl<Scalar: IsTensorScalar + 'static, const BATCH_SIZE: usize> IsBatchScalar
    for AutoSimd<[Scalar; BATCH_SIZE]>
{
}

/// Static vector
pub type SVec<ScalarLike, const ROWS: usize> = nalgebra::SVector<ScalarLike, ROWS>;
/// Static matrix
pub type SMat<ScalarLike, const ROWS: usize, const COLS: usize> =
    nalgebra::SMatrix<ScalarLike, ROWS, COLS>;

/// Batch scalar
pub type BatchScalar<ScalarLike, const BATCH_SIZE: usize> = AutoSimd<[ScalarLike; BATCH_SIZE]>;
/// Batch vector
pub type BatchVec<ScalarLike, const ROWS: usize, const BATCH_SIZE: usize> =
    nalgebra::SVector<AutoSimd<[ScalarLike; BATCH_SIZE]>, ROWS>;
/// Batch matrix
pub type BatchMat<ScalarLike, const ROWS: usize, const COLS: usize, const BATCH_SIZE: usize> =
    nalgebra::SMatrix<AutoSimd<[ScalarLike; BATCH_SIZE]>, ROWS, COLS>;

/// Trait for static tensors
pub trait IsStaticTensor<
    Scalar: IsTensorScalar + 'static,
    const SRANK: usize,
    const ROWS: usize,
    const COLS: usize,
    const BATCH_SIZE: usize,
>: Copy + Clone + Debug
{
    /// Create a tensor from a slice
    fn from_slice(slice: &[Scalar]) -> Self;

    /// Create a zero tensor
    fn zero() -> Self;

    /// Get the number category
    fn number_category() -> NumberCategory {
        Scalar::number_category()
    }

    /// Returns ith scalar element
    fn scalar(&self, idx: [usize; SRANK]) -> &Scalar;

    /// Get the rank
    fn rank(&self) -> usize {
        SRANK
    }

    /// Get the number of batches
    fn num_batches(&self) -> usize {
        BATCH_SIZE
    }

    /// Get the number of rows
    fn num_rows(&self) -> usize {
        ROWS
    }

    /// Get the number of columns
    fn num_cols(&self) -> usize {
        COLS
    }

    /// Get the compile time shape as an array
    fn sdims() -> [usize; SRANK];

    /// Number of scalar elements
    fn num_scalars() -> usize {
        BATCH_SIZE * ROWS * COLS
    }

    /// Get the stride as an array
    fn strides() -> [usize; SRANK];
}

// Rank 0 tensors
//
// a scalar
impl<Scalar: IsTensorScalar + 'static> IsStaticTensor<Scalar, 0, 1, 1, 1> for Scalar {
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

    fn from_slice(slice: &[Scalar]) -> Self {
        slice[0]
    }
}

// RANK 1 TENSORS
//
// A batch ofBatchScalar scalars
impl<Scalar: IsTensorScalar + 'static, const BATCH_SIZE: usize>
    IsStaticTensor<Scalar, 1, 1, 1, BATCH_SIZE> for BatchScalar<Scalar, BATCH_SIZE>
{
    fn zero() -> Self {
        todo!()
    }

    fn scalar(&self, idx: [usize; 1]) -> &Scalar {
        &self.0[idx[0]]
    }

    fn sdims() -> [usize; 1] {
        [BATCH_SIZE]
    }

    fn strides() -> [usize; 1] {
        [1]
    }

    fn from_slice(_slice: &[Scalar]) -> Self {
        todo!("BatchScalar::from_slice")
    }
}

// A vector
impl<Scalar: IsTensorScalar + 'static, const ROWS: usize> IsStaticTensor<Scalar, 1, ROWS, 1, 1>
    for SVec<Scalar, ROWS>
{
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

    fn from_slice(slice: &[Scalar]) -> Self {
        let mut v = Self::zeros();
        v.copy_from_slice(slice);
        v
    }
}

// RANK 2 TENSORS
//
// A batch of vectors
impl<Scalar: IsTensorScalar + 'static, const BATCH_SIZE: usize, const ROWS: usize>
    IsStaticTensor<Scalar, 2, ROWS, 1, BATCH_SIZE> for SVec<AutoSimd<[Scalar; BATCH_SIZE]>, ROWS>
{
    fn zero() -> Self {
        //Self::zeros()
        todo!()
    }

    fn scalar(&self, idx: [usize; 2]) -> &Scalar {
        &self[idx[1]].0[idx[0]]
    }

    fn sdims() -> [usize; 2] {
        [BATCH_SIZE, ROWS]
    }

    fn strides() -> [usize; 2] {
        [1, BATCH_SIZE]
    }

    fn from_slice(_slice: &[Scalar]) -> Self {
        todo!("SVec<AutoSimd<[Scalar; BATCH_SIZE]>, ROWS>::from_slice")
    }
}

// a matrix
impl<Scalar: IsTensorScalar + 'static, const ROWS: usize, const COLS: usize>
    IsStaticTensor<Scalar, 2, ROWS, COLS, 1> for SMat<Scalar, ROWS, COLS>
{
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

    fn from_slice(slice: &[Scalar]) -> Self {
        let mut v = Self::zeros();
        v.copy_from_slice(slice);
        v
    }
}

// RANK 3 TENSORS

// a batch of matrices
impl<
        Scalar: IsTensorScalar + 'static,
        const BATCH_SIZE: usize,
        const ROWS: usize,
        const COLS: usize,
    > IsStaticTensor<Scalar, 3, ROWS, COLS, BATCH_SIZE>
    for SMat<AutoSimd<[Scalar; BATCH_SIZE]>, ROWS, COLS>
{
    fn zero() -> Self {
        todo!()
    }
    fn scalar(&self, idx: [usize; 3]) -> &Scalar {
        &self[(idx[1], idx[2])].0[idx[0]]
    }

    fn sdims() -> [usize; 3] {
        [ROWS, COLS, BATCH_SIZE]
    }

    fn strides() -> [usize; 3] {
        [1, BATCH_SIZE, BATCH_SIZE * ROWS]
    }

    fn from_slice(_slice: &[Scalar]) -> Self {
        todo!("SMat<AutoSimd<[Scalar; BATCH_SIZE]>, ROWS, COLS>::from_slice")
    }
}

/// Format of a static tensor
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct STensorFormat {
    /// Number category
    pub number_category: NumberCategory,
    /// Number of bytes per scalar
    pub num_bytes_per_scalar: usize,
    /// batch size
    pub batch_size: usize,
    /// number of rows
    pub num_rows: usize,
    /// number of columns
    pub num_cols: usize,
}

impl STensorFormat {
    /// Create a new tensor format struct
    pub fn new<
        Scalar: IsTensorScalar + 'static,
        const ROWS: usize,
        const COLS: usize,
        const BATCH_SIZE: usize,
    >() -> Self {
        STensorFormat {
            number_category: Scalar::number_category(),
            num_rows: ROWS,
            num_cols: COLS,
            batch_size: BATCH_SIZE,
            num_bytes_per_scalar: std::mem::size_of::<Scalar>(),
        }
    }

    /// Number of bytes
    pub fn num_bytes(&self) -> usize {
        self.batch_size * self.num_rows * self.num_cols * self.num_bytes_per_scalar
    }
}
