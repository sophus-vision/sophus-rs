#[cfg(feature = "simd")]
/// Boolean mask - generalization of bool to SIMD
pub mod batch_mask;
#[cfg(feature = "simd")]
/// Batch matrix types - require the `simd` feature
pub mod batch_matrix;
#[cfg(feature = "simd")]
/// Batch scalar types - require the `simd` feature
pub mod batch_scalar;
#[cfg(feature = "simd")]
/// Batch vector types - require the `simd` feature
pub mod batch_vector;
/// Bool and boolean mask traits
pub mod bool_mask;
/// Matrix types
pub mod matrix;
/// Scalar types
pub mod scalar;
/// Vector types
pub mod vector;

#[cfg(feature = "simd")]
use core::ops::Add;
#[cfg(feature = "simd")]
use core::simd::cmp::SimdPartialEq;
#[cfg(feature = "simd")]
use core::simd::num::SimdFloat;
#[cfg(feature = "simd")]
use core::simd::LaneCount;
#[cfg(feature = "simd")]
use core::simd::Mask;
#[cfg(feature = "simd")]
use core::simd::Simd;
#[cfg(feature = "simd")]
use core::simd::SimdElement;
#[cfg(feature = "simd")]
use core::simd::SupportedLaneCount;

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

/// f64 epsilon.
pub const EPS_F64: f64 = 1e-6;

#[cfg(feature = "simd")]
/// Batch of scalar
#[derive(Clone, Debug, PartialEq, Copy)]
pub struct BatchScalar<ScalarLike: SimdElement, const BATCH: usize>(
    pub(crate) Simd<ScalarLike, BATCH>,
)
where
    LaneCount<BATCH>: SupportedLaneCount;

#[cfg(feature = "simd")]
/// Batch of vectors
pub type BatchVec<ScalarLike, const ROWS: usize, const BATCH: usize> =
    nalgebra::SVector<BatchScalar<ScalarLike, BATCH>, ROWS>;

#[cfg(feature = "simd")]
/// Batch of matrices
pub type BatchMat<ScalarLike, const ROWS: usize, const COLS: usize, const BATCH: usize> =
    nalgebra::SMatrix<BatchScalar<ScalarLike, BATCH>, ROWS, COLS>;

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
impl<S: SimdElement + num_traits::Zero, const BATCH: usize> Add for BatchScalar<S, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
    Simd<S, BATCH>: Add<Output = Simd<S, BATCH>>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

#[cfg(feature = "simd")]
impl<S: SimdElement + num_traits::Zero, const BATCH: usize> num_traits::Zero
    for BatchScalar<S, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
    Simd<S, BATCH>: SimdFloat,
    Simd<S, BATCH>: SimdPartialEq<Mask = Mask<S::Mask, BATCH>> + Add<Output = Simd<S, BATCH>>,
{
    fn zero() -> Self {
        Self(Simd::<S, BATCH>::splat(S::zero()))
    }

    fn is_zero(&self) -> bool {
        let b = self.0.simd_eq(Simd::<S, BATCH>::splat(S::zero()));
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
    use approx::assert_abs_diff_eq;

    use crate::linalg::scalar::IsScalar;

    let batch_scalar = BatchScalar::<f64, 4>(Simd::<f64, 4>::from_array([1.0, 2.0, 3.0, 4.0]));
    assert_abs_diff_eq!(
        batch_scalar,
        BatchScalarF64::<4>::from_real_array([1.0, 2.0, 3.0, 4.0])
    );
}
