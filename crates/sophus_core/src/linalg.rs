/// Boolean mask - generalization of bool to SIMD
pub mod bool_mask;

/// Matrix types
pub mod matrix;

/// Scalar types
pub mod scalar;

/// Vector types
pub mod vector;

use std::ops::Add;
use std::simd::cmp::SimdPartialEq;
use std::simd::num::SimdFloat;
use std::simd::LaneCount;
use std::simd::Mask;
use std::simd::Simd;
use std::simd::SimdElement;
use std::simd::SupportedLaneCount;

/// Static vector
pub type SVec<ScalarLike, const ROWS: usize> = nalgebra::SVector<ScalarLike, ROWS>;
/// Static matrix
pub type SMat<ScalarLike, const ROWS: usize, const COLS: usize> =
    nalgebra::SMatrix<ScalarLike, ROWS, COLS>;

/// Batch of scalar
#[derive(Clone, Debug, PartialEq, Copy)]
pub struct BatchScalar<ScalarLike: SimdElement, const BATCH_SIZE: usize>(
    Simd<ScalarLike, BATCH_SIZE>,
)
where
    LaneCount<BATCH_SIZE>: SupportedLaneCount;

/// Batch of vectors
pub type BatchVec<ScalarLike, const ROWS: usize, const BATCH_SIZE: usize> =
    nalgebra::SVector<BatchScalar<ScalarLike, BATCH_SIZE>, ROWS>;

/// Batch of matrices
pub type BatchMat<ScalarLike, const ROWS: usize, const COLS: usize, const BATCH_SIZE: usize> =
    nalgebra::SMatrix<BatchScalar<ScalarLike, BATCH_SIZE>, ROWS, COLS>;

/// f32 vector
pub type VecF32<const ROWS: usize> = nalgebra::SVector<f32, ROWS>;
/// f64 vector
pub type VecF64<const ROWS: usize> = nalgebra::SMatrix<f64, ROWS, 1>;
/// f64 matrix
pub type MatF32<const ROWS: usize, const COLS: usize> = nalgebra::SMatrix<f32, ROWS, COLS>;
/// f64 matrix
pub type MatF64<const ROWS: usize, const COLS: usize> = nalgebra::SMatrix<f64, ROWS, COLS>;

/// batch of f64 scalars
pub type BatchScalarF64<const BATCH: usize> = BatchScalar<f64, BATCH>;
/// batch of f64 vectors
pub type BatchVecF64<const ROWS: usize, const BATCH: usize> = BatchVec<f64, ROWS, BATCH>;
/// batch of f64 matrices
pub type BatchMatF64<const ROWS: usize, const COLS: usize, const BATCH: usize> =
    BatchMat<f64, ROWS, COLS, BATCH>;

impl<S: SimdElement + num_traits::Zero, const BATCH_SIZE: usize> Add for BatchScalar<S, BATCH_SIZE>
where
    LaneCount<BATCH_SIZE>: SupportedLaneCount,
    Simd<S, BATCH_SIZE>: Add<Output = Simd<S, BATCH_SIZE>>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<S: SimdElement + num_traits::Zero, const BATCH_SIZE: usize> num_traits::Zero
    for BatchScalar<S, BATCH_SIZE>
where
    LaneCount<BATCH_SIZE>: SupportedLaneCount,
    Simd<S, BATCH_SIZE>: SimdFloat,
    Simd<S, BATCH_SIZE>:
        SimdPartialEq<Mask = Mask<S::Mask, BATCH_SIZE>> + Add<Output = Simd<S, BATCH_SIZE>>,
{
    fn zero() -> Self {
        Self(Simd::<S, BATCH_SIZE>::splat(S::zero()))
    }

    fn is_zero(&self) -> bool {
        let b = self.0.simd_eq(Simd::<S, BATCH_SIZE>::splat(S::zero()));
        b.all()
    }
}

#[test]
fn test_core() {
    use crate::linalg::scalar::IsScalar;
    use approx::assert_abs_diff_eq;

    let vec = SVec::<f32, 3>::new(1.0, 2.0, 3.0);
    assert_abs_diff_eq!(vec, SVec::<f32, 3>::new(1.0, 2.0, 3.0));

    let batch_scalar = BatchScalar::<f64, 4>(Simd::<f64, 4>::from_array([1.0, 2.0, 3.0, 4.0]));
    assert_abs_diff_eq!(
        batch_scalar,
        BatchScalarF64::<4>::from_real_array([1.0, 2.0, 3.0, 4.0])
    );
}
