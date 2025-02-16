//! Core linear algebra utilities and basic types.
//!
//! This module provides:
//! - **Static vectors** (`SVec`) and **static matrices** (`SMat`) which are type aliases of
//!   `nalgebra` types.
//! - Aliases such as [linalg::VecF64] and [linalg::MatF64] for common real matrix types.
//! - Support for batch (SIMD) scalars, vectors, and matrices when `feature = "simd"` is enabled.

#[cfg(feature = "simd")]
mod batch_mask;
#[cfg(feature = "simd")]
mod batch_matrix;
#[cfg(feature = "simd")]
mod batch_scalar;
#[cfg(feature = "simd")]
mod batch_vector;
mod bool_mask;
mod matrix;
mod scalar;
mod vector;

#[cfg(feature = "simd")]
use core::ops::Add;
#[cfg(feature = "simd")]
use core::simd::{
    cmp::SimdPartialEq,
    num::SimdFloat,
    LaneCount,
    Mask,
    Simd,
    SimdElement,
    SupportedLaneCount,
};

#[cfg(feature = "simd")]
pub use crate::linalg::batch_mask::*;
pub use crate::linalg::{
    bool_mask::*,
    matrix::*,
    scalar::*,
    vector::*,
};

/// A statically sized vector, powered by `nalgebra::SVector`.
///
/// # Type Parameters
/// - `ScalarLike`: The scalar type for each element (e.g., `f64`).
/// - `ROWS`: Dimensionality of the vector.
pub type SVec<ScalarLike, const ROWS: usize> = nalgebra::SVector<ScalarLike, ROWS>;

/// A statically sized matrix, powered by `nalgebra::SMatrix`.
///
/// # Type Parameters
/// - `ScalarLike`: The scalar type for each element (e.g., `f64`).
/// - `ROWS`, `COLS`: Dimensions of the matrix.
pub type SMat<ScalarLike, const ROWS: usize, const COLS: usize> =
    nalgebra::SMatrix<ScalarLike, ROWS, COLS>;

/// A 1D column vector of `f32` elements, dimension given by `ROWS`.
pub type VecF32<const ROWS: usize> = nalgebra::SVector<f32, ROWS>;

/// A 1D column vector of `f64` elements, dimension given by `ROWS`.
///
/// Implemented as an [`SMatrix<f64, ROWS, 1>`][nalgebra::SMatrix].
///
/// This design choice (one column) allows for easy linear algebra operations
/// with [`nalgebra::SMatrix`].
pub type VecF64<const ROWS: usize> = nalgebra::SMatrix<f64, ROWS, 1>;

/// A 2D matrix of `f32` elements, size `[ROWS, COLS]`.
pub type MatF32<const ROWS: usize, const COLS: usize> = nalgebra::SMatrix<f32, ROWS, COLS>;

/// A 2D matrix of `f64` elements, size `[ROWS, COLS]`.
pub type MatF64<const ROWS: usize, const COLS: usize> = nalgebra::SMatrix<f64, ROWS, COLS>;

/// A small epsilon value (`1e-6`) for `f64` computations.
pub const EPS_F64: f64 = 1e-6;

#[cfg(feature = "simd")]
/// A batch scalar type for SIMD usage.
///
/// Each instance holds an underlying [`Simd<ScalarLike, BATCH>`] of length `BATCH`,
/// allowing parallel operations on multiple lanes.
#[derive(Clone, Debug, PartialEq, Copy)]
pub struct BatchScalar<ScalarLike: SimdElement, const BATCH: usize>(
    pub(crate) Simd<ScalarLike, BATCH>,
)
where
    LaneCount<BATCH>: SupportedLaneCount;

#[cfg(feature = "simd")]
/// A batch vector of dimension `ROWS`, where each element is a [`BatchScalar`].
///
/// This allows parallel operations on multiple lanes in each element of the vector.
pub type BatchVec<ScalarLike, const ROWS: usize, const BATCH: usize> =
    nalgebra::SVector<BatchScalar<ScalarLike, BATCH>, ROWS>;

#[cfg(feature = "simd")]
/// A batch matrix of dimension `[ROWS, COLS]`, where each element is a [`BatchScalar`].
///
/// This allows parallel operations on multiple lanes in each element of the matrix.
pub type BatchMat<ScalarLike, const ROWS: usize, const COLS: usize, const BATCH: usize> =
    nalgebra::SMatrix<BatchScalar<ScalarLike, BATCH>, ROWS, COLS>;

#[cfg(feature = "simd")]
/// A batch scalar specialized for `f64` values.
///
/// Each instance is effectively `Simd<f64, BATCH>`, allowing for parallel floating-point ops.
pub type BatchScalarF64<const BATCH: usize> = BatchScalar<f64, BATCH>;

/// A batch vector specialized for `f64`, dimension `[ROWS]` with `BATCH` lanes.
///
/// Each element is `[BatchScalar<f64, BATCH>]`.
#[cfg(feature = "simd")]
pub type BatchVecF64<const ROWS: usize, const BATCH: usize> = BatchVec<f64, ROWS, BATCH>;

/// A batch matrix specialized for `f64`, dimension `[ROWS, COLS]` with `BATCH` lanes.
///
/// Each element is `[BatchScalar<f64, BATCH>]`.
#[cfg(feature = "simd")]
pub type BatchMatF64<const ROWS: usize, const COLS: usize, const BATCH: usize> =
    BatchMat<f64, ROWS, COLS, BATCH>;

#[cfg(feature = "simd")]
impl<S, const BATCH: usize> Add for BatchScalar<S, BATCH>
where
    S: SimdElement + num_traits::Zero,
    LaneCount<BATCH>: SupportedLaneCount,
    Simd<S, BATCH>: Add<Output = Simd<S, BATCH>>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

#[cfg(feature = "simd")]
impl<S, const BATCH: usize> num_traits::Zero for BatchScalar<S, BATCH>
where
    S: SimdElement + num_traits::Zero,
    LaneCount<BATCH>: SupportedLaneCount,
    Simd<S, BATCH>:
        SimdFloat + SimdPartialEq<Mask = Mask<S::Mask, BATCH>> + Add<Output = Simd<S, BATCH>>,
{
    fn zero() -> Self {
        Self(Simd::<S, BATCH>::splat(S::zero()))
    }

    fn is_zero(&self) -> bool {
        let zero_mask = self.0.simd_eq(Simd::<S, BATCH>::splat(S::zero()));
        zero_mask.all()
    }
}

/// A marker trait bounding `f32` or `f64` (and possibly other floats) by additional numeric
/// constraints.
///
/// This trait indicates "floating point" behavior in the sense of `num_traits::Float`, along with
/// some additional conversions. Typically used to ensure an implementor is a standard floating
/// type.
pub trait FloatingPointNumber:
    num_traits::Float + num_traits::FromPrimitive + num_traits::NumCast
{
}

impl FloatingPointNumber for f32 {}
impl FloatingPointNumber for f64 {}

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

    use crate::linalg::IsScalar;

    let batch_scalar = BatchScalar::<f64, 4>(Simd::<f64, 4>::from_array([1.0, 2.0, 3.0, 4.0]));
    // Demonstrate that `BatchScalarF64` can be created via a "real array" (per-lane values).
    assert_abs_diff_eq!(
        batch_scalar,
        BatchScalarF64::<4>::from_real_array([1.0, 2.0, 3.0, 4.0])
    );
}
