use crate::linalg::scalar::NumberCategory;
use crate::linalg::SMat;
use crate::linalg::SVec;
use crate::prelude::*;
use std::fmt::Debug;
pub use typenum::generic_const_mappings::Const;

/// Trait for static tensors
pub trait IsStaticTensor<
    Scalar: IsCoreScalar + 'static,
    const SRANK: usize,
    const ROWS: usize,
    const COLS: usize,
>: Clone + Debug + num_traits::Zero
{
    /// Returns ith scalar element
    fn scalar(&self, idx: [usize; SRANK]) -> &Scalar;

    /// Get the rank
    fn rank(&self) -> usize {
        SRANK
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
        ROWS * COLS
    }

    /// Get the stride as an array
    fn strides() -> [usize; SRANK];

    /// Create a tensor from a slice
    fn from_slice(slice: &[Scalar]) -> Self;
}

// Rank 0 tensors
//
// a scalar
impl<Scalar: IsCoreScalar + 'static> IsStaticTensor<Scalar, 0, 1, 1> for Scalar {
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
        slice[0].clone()
    }
}

// A vector
impl<Scalar: IsCoreScalar + 'static, const ROWS: usize> IsStaticTensor<Scalar, 1, ROWS, 1>
    for SVec<Scalar, ROWS>
{
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
        SVec::from_iterator(slice.iter().cloned())
    }
}

// a matrix
impl<Scalar: IsCoreScalar + 'static, const ROWS: usize, const COLS: usize>
    IsStaticTensor<Scalar, 2, ROWS, COLS> for SMat<Scalar, ROWS, COLS>
{
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
        SMat::from_iterator(slice.iter().cloned())
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
        Scalar: IsCoreScalar + 'static,
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
        self.num_rows * self.num_cols * self.num_bytes_per_scalar
    }
}

#[test]
fn test_elements() {
    use crate::linalg::scalar::IsScalar;
    use crate::linalg::scalar::NumberCategory;
    use crate::linalg::BatchScalar;
    use crate::linalg::BatchScalarF64;
    use crate::linalg::BatchVecF64;

    use crate::linalg::VecF32;
    use approx::assert_abs_diff_eq;
    assert_eq!(f32::number_category(), NumberCategory::Real);
    assert_eq!(u32::number_category(), NumberCategory::Unsigned);
    assert_eq!(i32::number_category(), NumberCategory::Signed);
    assert_eq!(
        BatchScalar::<f64, 4>::number_category(),
        NumberCategory::Real
    );

    let zeros_vec: VecF32<4> = IsStaticTensor::<f32, 1, 4, 1>::from_slice(&[0.0f32, 0.0, 0.0, 0.0]);
    for elem in zeros_vec.iter() {
        assert_eq!(*elem, 0.0);
    }

    let vec = SVec::<f32, 3>::new(1.0, 2.0, 3.0);
    assert_abs_diff_eq!(vec, SVec::<f32, 3>::new(1.0, 2.0, 3.0));

    let mat = SMat::<f32, 2, 2>::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(mat.scalar([0, 0]), &1.0);
    assert_eq!(mat.scalar([0, 1]), &2.0);
    assert_eq!(mat.scalar([1, 0]), &3.0);
    assert_eq!(mat.scalar([1, 1]), &4.0);
    assert_abs_diff_eq!(mat, SMat::<f32, 2, 2>::new(1.0, 2.0, 3.0, 4.0));

    let batch_vec: BatchVecF64<2, 2> =
        BatchVecF64::from_element(BatchScalarF64::from_real_array([1.0, 2.0]));
    assert_eq!(batch_vec.scalar([0]).extract_single(0), 1.0);
    assert_eq!(batch_vec.scalar([1]).extract_single(1), 2.0);
}
