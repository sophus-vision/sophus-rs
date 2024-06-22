#[cfg(feature = "simd")]
/// Boolean mask - generalization of bool to SIMD
pub mod batch_mask;
/// Bool and boolean mask traits
pub mod bool_mask;

#[cfg(feature = "simd")]
/// Batch matrix types - require the `simd` feature
pub mod batch_matrix;
/// Matrix types
pub mod matrix;

#[cfg(feature = "simd")]
/// Batch scalar types - require the `simd` feature
pub mod batch_scalar;
/// Scalar types
pub mod scalar;

#[cfg(feature = "simd")]
/// Batch vector types - require the `simd` feature
pub mod batch_vector;
/// Vector types
pub mod vector;

#[cfg(feature = "simd")]
use std::ops::Add;

#[cfg(feature = "simd")]
use std::simd::cmp::SimdPartialEq;
#[cfg(feature = "simd")]
use std::simd::num::SimdFloat;
#[cfg(feature = "simd")]
use std::simd::LaneCount;
#[cfg(feature = "simd")]
use std::simd::Mask;
#[cfg(feature = "simd")]
use std::simd::Simd;
#[cfg(feature = "simd")]
use std::simd::SimdElement;
#[cfg(feature = "simd")]
use std::simd::SupportedLaneCount;

/// Static vector
pub type SVec<ScalarLike, const ROWS: usize> = nalgebra::SVector<ScalarLike, ROWS>;
/// Static matrix
pub type SMat<ScalarLike, const ROWS: usize, const COLS: usize> =
    nalgebra::SMatrix<ScalarLike, ROWS, COLS>;

/// f32 vector
pub type VecF32<const ROWS: usize> = nalgebra::SVector<f32, ROWS>;
/// f64 vector
pub type VecF64<const ROWS: usize> = nalgebra::SMatrix<f64, ROWS, 1>;
/// f64 matrix
pub type MatF32<const ROWS: usize, const COLS: usize> = nalgebra::SMatrix<f32, ROWS, COLS>;
/// f64 matrix
pub type MatF64<const ROWS: usize, const COLS: usize> = nalgebra::SMatrix<f64, ROWS, COLS>;

#[cfg(feature = "simd")]
/// Batch of scalar
#[derive(Clone, Debug, PartialEq, Copy)]
pub struct BatchScalar<ScalarLike: SimdElement, const BATCH_SIZE: usize>(
    Simd<ScalarLike, BATCH_SIZE>,
)
where
    LaneCount<BATCH_SIZE>: SupportedLaneCount;

#[cfg(feature = "simd")]
/// Batch of vectors
pub type BatchVec<ScalarLike, const ROWS: usize, const BATCH_SIZE: usize> =
    nalgebra::SVector<BatchScalar<ScalarLike, BATCH_SIZE>, ROWS>;

#[cfg(feature = "simd")]
/// Batch of matrices
pub type BatchMat<ScalarLike, const ROWS: usize, const COLS: usize, const BATCH_SIZE: usize> =
    nalgebra::SMatrix<BatchScalar<ScalarLike, BATCH_SIZE>, ROWS, COLS>;

#[cfg(feature = "simd")]
/// batch of f64 scalars
pub type BatchScalarF64<const BATCH: usize> = BatchScalar<f64, BATCH>;
/// batch of f64 vectors
#[cfg(feature = "simd")]
pub type BatchVecF64<const ROWS: usize, const BATCH: usize> = BatchVec<f64, ROWS, BATCH>;
/// batch of f64 matrices
#[cfg(feature = "simd")]
pub type BatchMatF64<const ROWS: usize, const COLS: usize, const BATCH: usize> =
    BatchMat<f64, ROWS, COLS, BATCH>;

#[cfg(feature = "simd")]
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

#[cfg(feature = "simd")]
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
    use approx::assert_abs_diff_eq;

    let vec = SVec::<f32, 3>::new(1.0, 2.0, 3.0);
    assert_abs_diff_eq!(vec, SVec::<f32, 3>::new(1.0, 2.0, 3.0));
}

#[test]
#[cfg(feature = "simd")]
fn test_simd_core() {
    use crate::linalg::scalar::IsScalar;
    use approx::assert_abs_diff_eq;

    let batch_scalar = BatchScalar::<f64, 4>(Simd::<f64, 4>::from_array([1.0, 2.0, 3.0, 4.0]));
    assert_abs_diff_eq!(
        batch_scalar,
        BatchScalarF64::<4>::from_real_array([1.0, 2.0, 3.0, 4.0])
    );
}
