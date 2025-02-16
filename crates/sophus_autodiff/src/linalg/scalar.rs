use core::{
    borrow::Borrow,
    fmt::Debug,
    ops::{
        Add,
        AddAssign,
        Div,
        DivAssign,
        Mul,
        MulAssign,
        Neg,
        Sub,
        SubAssign,
    },
};

use approx::{
    assert_abs_diff_eq,
    AbsDiffEq,
    RelativeEq,
};

use crate::{
    dual::{
        DualMatrix,
        DualScalar,
        DualVector,
    },
    linalg::{
        MatF64,
        VecF64,
        EPS_F64,
    },
    prelude::*,
};
extern crate alloc;

/// Specifies the general “kind” of a numeric type, e.g. real, signed integer, or unsigned integer.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NumberCategory {
    /// A real number (e.g., `f32` or `f64`).
    Real,
    /// An unsigned integer (e.g., `u8`, `u16`, `u32`, `u64`).
    Unsigned,
    /// A signed integer (e.g., `i8`, `i16`, `i32`, `i64`).
    Signed,
}

/// A trait for core scalar types, which may be integers, floats, or batch variants thereof.
/// This is a low-level building block in the library, providing minimal numeric requirements
/// such as zero-initialization and debug printing.
///
/// Common implementors include `f64`, `i32`, and batch types via `portable_simd`.
pub trait IsCoreScalar:
    Clone + Debug + nalgebra::Scalar + num_traits::Zero + core::ops::AddAssign
{
    /// Indicates whether the scalar is Real, Unsigned, or Signed integer.
    fn number_category() -> NumberCategory;
}

// Implement `IsCoreScalar` for native Rust scalars:
macro_rules! def_is_tensor_scalar_single {
    ($scalar:ty, $cat:expr) => {
        impl IsCoreScalar for $scalar {
            fn number_category() -> NumberCategory {
                $cat
            }
        }
    };
}

def_is_tensor_scalar_single!(u8, NumberCategory::Unsigned);
def_is_tensor_scalar_single!(u16, NumberCategory::Unsigned);
def_is_tensor_scalar_single!(u32, NumberCategory::Unsigned);
def_is_tensor_scalar_single!(u64, NumberCategory::Unsigned);
def_is_tensor_scalar_single!(i8, NumberCategory::Signed);
def_is_tensor_scalar_single!(i16, NumberCategory::Signed);
def_is_tensor_scalar_single!(i32, NumberCategory::Signed);
def_is_tensor_scalar_single!(i64, NumberCategory::Signed);
def_is_tensor_scalar_single!(f32, NumberCategory::Real);
def_is_tensor_scalar_single!(f64, NumberCategory::Real);

/// The main scalar trait used by sophus-rs, supporting both real and dual-number types,
/// in single or batch (SIMD) form.
///
/// # Generic Parameters
/// - `BATCH`: Number of lanes for batch usage, or 1 for single-scalar usage.
/// - `DM`, `DN`: Dimensions of the derivative if the scalar is dual-based (otherwise 0 for real).
///
/// This trait unifies the behavior of scalars so that higher-level crates can
/// write generic code that handles f64, dual scalars, or batch variations.
pub trait IsScalar<const BATCH: usize, const DM: usize, const DN: usize>:
    PartialEq
    + Debug
    + Clone
    + Copy
    + core::ops::Div<Output = Self>
    + Add<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Sized
    + Neg<Output = Self>
    + AbsDiffEq<Epsilon = f64>
    + RelativeEq<Epsilon = f64>
    + IsCoreScalar
    + 'static
    + Send
    + Sync
{
    /// The same type as `Self`, used to unify generic signatures.
    type Scalar: IsScalar<BATCH, DM, DN>;

    /// The vector type associated with this scalar, e.g. `VecF64<ROWS>` or `DualVector<ROWS>`.
    type Vector<const ROWS: usize>: IsVector<Self, ROWS, BATCH, DM, DN>;
    /// The matrix type associated with this scalar, e.g. `MatF64<ROWS, COLS>` or `DualMatrix<ROWS,
    /// COLS>`.
    type Matrix<const ROWS: usize, const COLS: usize>: IsMatrix<Self, ROWS, COLS, BATCH, DM, DN>;

    /// A “single” form of this scalar, used if we need to drop batch or partial-derivative
    /// components.
    type SingleScalar: IsSingleScalar<DM, DN>;

    /// The real scalar type corresponding to `Self`.
    type RealScalar: IsRealScalar<BATCH>;
    /// The real vector type corresponding to `Self`'s vector form.
    type RealVector<const ROWS: usize>: IsRealVector<Self::RealScalar, ROWS, BATCH>;
    /// The real matrix type corresponding to `Self`'s matrix form.
    type RealMatrix<const ROWS: usize, const COLS: usize>: IsRealMatrix<
        Self::RealScalar,
        ROWS,
        COLS,
        BATCH,
    >;

    /// A dual scalar type for forward-mode AD.
    type DualScalar<const M: usize, const N: usize>: IsDualScalar<BATCH, M, N>;
    /// The corresponding dual vector type.
    type DualVector<const ROWS: usize, const M: usize, const N: usize>: IsDualVector<
        Self::DualScalar<M, N>,
        ROWS,
        BATCH,
        M,
        N,
    >;
    /// The corresponding dual matrix type.
    type DualMatrix<const ROWS: usize, const COLS: usize, const M: usize, const N: usize>:
        IsDualMatrix<Self::DualScalar<M, N>, ROWS, COLS, BATCH, M, N>;

    /// A boolean mask type for lane-wise operations (`true`/`false` per-lane).
    type Mask: IsBoolMask;

    /// Absolute value of the scalar.
    fn abs(&self) -> Self;

    /// Arc-cosine (inverse cosine).
    fn acos(&self) -> Self;

    /// Arc-sine (inverse sine).
    fn asin(&self) -> Self;

    /// Arc-tangent.
    fn atan(&self) -> Self;

    /// Arc-tangent of `self` / `x`.
    fn atan2<S>(&self, x: S) -> Self
    where
        S: Borrow<Self>;

    /// Cosine.
    fn cos(&self) -> Self;

    /// Returns the machine epsilon or small tolerance for this scalar.
    fn eps() -> Self;

    /// Exponential function.
    fn exp(&self) -> Self;

    /// Natural logarithm.
    fn ln(&self) -> Self;

    /// Extracts a single-lane scalar, used when `BATCH > 1`.
    fn extract_single(&self, i: usize) -> Self::SingleScalar;

    /// Returns the floor of the scalar as a real scalar (not necessarily preserving derivative
    /// info).
    fn floor(&self) -> Self::RealScalar;

    /// Returns the fractional part of the scalar.
    fn fract(&self) -> Self;

    /// Creates a new scalar from an `f64` value, zeroing out any dual/batch parts.
    fn from_f64(val: f64) -> Self;

    /// Creates a new scalar from an array of real `f64`, used in batch contexts or single-lane.
    fn from_real_array(arr: [f64; BATCH]) -> Self;

    /// Converts a real scalar into `Self`, zeroing out any dual/batch parts.
    fn from_real_scalar(val: Self::RealScalar) -> Self;

    /// Greater-or-equal comparison, returning a boolean mask.
    fn greater_equal(&self, rhs: &Self) -> Self::Mask;

    /// Less-or-equal comparison, returning a boolean mask.
    fn less_equal(&self, rhs: &Self) -> Self::Mask;

    /// Creates a “ones” scalar (the numeric value 1.0).
    fn ones() -> Self {
        Self::from_f64(1.0)
    }

    /// Returns just the “real part” of the scalar, discarding any derivative or batch info.
    fn real_part(&self) -> Self::RealScalar;

    /// Returns example scalar values for testing or demonstration.
    fn scalar_examples() -> alloc::vec::Vec<Self>;

    /// Selects lane-wise from `self` or `other` based on the given `mask`.
    fn select(&self, mask: &Self::Mask, other: Self) -> Self;

    /// Returns the sign of the scalar, ignoring derivative parts.
    fn signum(&self) -> Self;

    /// Sine.
    fn sin(&self) -> Self;

    /// Square root.
    fn sqrt(&self) -> Self;

    /// Converts `self` to a dual scalar with zero derivative, or merges existing derivatives if
    /// relevant.
    fn to_dual_const<const M: usize, const N: usize>(&self) -> Self::DualScalar<M, N>;

    /// Returns an array of real values for each lane in a batch. If single-lane, returns `[self]`.
    fn to_real_array(&self) -> [f64; BATCH];

    /// Tangent.
    fn tan(&self) -> Self;

    /// Interprets `self` as a single-element vector (1D).
    fn to_vec(&self) -> Self::Vector<1>;

    /// Creates a “zeros” scalar (the numeric value 0.0).
    fn zeros() -> Self {
        Self::from_f64(0.0)
    }

    /// A basic test suite verifying certain trig identities, etc.
    fn test_suite() {
        let examples = Self::scalar_examples();
        for a in &examples {
            let sin_a = a.sin();
            let cos_a = a.cos();
            let val = sin_a * sin_a + cos_a * cos_a;
            let one = Self::ones();

            for i in 0..BATCH {
                assert_abs_diff_eq!(val.extract_single(i), one.extract_single(i));
            }
        }
    }
}

/// Marker trait for real scalars (`f32`, `f64`, or batch forms thereof).
pub trait IsRealScalar<const BATCH: usize>: IsScalar<BATCH, 0, 0> + Copy {}

/// A trait for single-lane scalars (BATCH=1). This trait ensures certain
/// single-scalar operations like floor, integer casting, etc.
pub trait IsSingleScalar<const DM: usize, const DN: usize>:
    IsScalar<1, DM, DN> + PartialEq + Div<Output = Self>
{
    /// The vector type that works with single-lane scalars.
    type SingleVector<const ROWS: usize>: IsSingleVector<Self, ROWS, DM, DN>;

    /// The matrix type that works with single-lane scalars.
    type SingleMatrix<const ROWS: usize, const COLS: usize>: IsSingleMatrix<
        Self,
        ROWS,
        COLS,
        DM,
        DN,
    >;

    /// Returns the real “f64” part if any. For non-real types, you might keep partial derivative
    /// out.
    fn single_real_scalar(&self) -> f64;

    /// Returns a copy of self as a scalar.
    fn single_scalar(&self) -> Self;

    /// Floors this scalar and returns it as an `i64`.
    fn i64_floor(&self) -> i64;
}

/// A trait for batch scalars (BATCH > 1) that can store multiple lanes of real or dual values.
/// Typically used with Rust’s portable SIMD or similar.
#[cfg(feature = "simd")]
pub trait IsBatchScalar<const BATCH: usize, const DM: usize, const DN: usize>:
    IsScalar<BATCH, DM, DN>
{
}

// Mark `f64` as real single-lane scalar:
impl IsRealScalar<1> for f64 {}

/// Example `IsScalar` implementation for `f64`.
impl IsScalar<1, 0, 0> for f64 {
    type Scalar = f64;
    type Vector<const ROWS: usize> = VecF64<ROWS>;
    type Matrix<const ROWS: usize, const COLS: usize> = MatF64<ROWS, COLS>;

    type SingleScalar = f64;

    type RealScalar = f64;
    type RealVector<const ROWS: usize> = VecF64<ROWS>;
    type RealMatrix<const ROWS: usize, const COLS: usize> = MatF64<ROWS, COLS>;

    type DualScalar<const M: usize, const N: usize> = DualScalar<M, N>;
    type DualVector<const ROWS: usize, const M: usize, const N: usize> = DualVector<ROWS, M, N>;
    type DualMatrix<const ROWS: usize, const COLS: usize, const M: usize, const N: usize> =
        DualMatrix<ROWS, COLS, M, N>;

    type Mask = bool;

    fn less_equal(&self, rhs: &Self) -> Self::Mask {
        self <= rhs
    }

    fn greater_equal(&self, rhs: &Self) -> Self::Mask {
        self >= rhs
    }

    fn scalar_examples() -> alloc::vec::Vec<f64> {
        alloc::vec![1.0, 2.0, 3.0]
    }

    fn abs(&self) -> f64 {
        f64::abs(*self)
    }

    fn cos(&self) -> f64 {
        f64::cos(*self)
    }

    fn exp(&self) -> f64 {
        f64::exp(*self)
    }

    fn ln(&self) -> Self {
        f64::ln(*self)
    }

    fn eps() -> f64 {
        EPS_F64
    }

    fn sin(&self) -> f64 {
        f64::sin(*self)
    }

    fn sqrt(&self) -> f64 {
        f64::sqrt(*self)
    }

    fn from_f64(val: f64) -> f64 {
        val
    }

    fn from_real_scalar(val: f64) -> f64 {
        val
    }

    fn atan2<S>(&self, x: S) -> Self
    where
        S: Borrow<f64>,
    {
        f64::atan2(*self, *x.borrow())
    }

    fn from_real_array(arr: [Self::RealScalar; 1]) -> Self {
        arr[0]
    }

    fn to_real_array(&self) -> [Self::RealScalar; 1] {
        [*self]
    }

    fn real_part(&self) -> f64 {
        *self
    }

    fn to_vec(&self) -> VecF64<1> {
        VecF64::<1>::new(*self)
    }

    fn tan(&self) -> Self {
        f64::tan(*self)
    }

    fn acos(&self) -> Self {
        f64::acos(*self)
    }

    fn asin(&self) -> Self {
        f64::asin(*self)
    }

    fn atan(&self) -> Self {
        f64::atan(*self)
    }

    fn fract(&self) -> Self {
        f64::fract(*self)
    }

    fn floor(&self) -> f64 {
        f64::floor(*self)
    }

    fn extract_single(&self, _i: usize) -> f64 {
        // For BATCH=1, we only have one lane, so ignore `i`:
        *self
    }

    fn signum(&self) -> Self {
        f64::signum(*self)
    }

    fn to_dual_const<const DM: usize, const DN: usize>(&self) -> Self::DualScalar<DM, DN> {
        DualScalar::from_f64(*self)
    }

    fn select(&self, mask: &Self::Mask, other: Self) -> Self {
        if *mask {
            *self
        } else {
            other
        }
    }

    fn test_suite() {
        // Basic test verifying sin^2 + cos^2 = 1, etc.
        let examples = Self::scalar_examples();
        for a in &examples {
            let sin_a = a.sin();
            let cos_a = a.cos();
            let val = sin_a * sin_a + cos_a * cos_a;
            let one = Self::ones();

            // Only 1 lane, so:
            for i in 0..1 {
                assert_abs_diff_eq!(val.extract_single(i), one.extract_single(i));
            }
        }
    }
}

impl IsSingleScalar<0, 0> for f64 {
    type SingleMatrix<const ROWS: usize, const COLS: usize> = MatF64<ROWS, COLS>;
    type SingleVector<const ROWS: usize> = VecF64<ROWS>;

    fn single_real_scalar(&self) -> f64 {
        *self
    }

    fn single_scalar(&self) -> Self {
        *self
    }

    fn i64_floor(&self) -> i64 {
        self.floor() as i64
    }
}

#[test]
fn scalar_prop_tests() {
    #[cfg(feature = "simd")]
    use crate::linalg::BatchScalarF64;

    // Test the f64 suite:
    f64::test_suite();

    // If simd is enabled, also test batch forms:
    #[cfg(feature = "simd")]
    {
        BatchScalarF64::<2>::test_suite();
        BatchScalarF64::<4>::test_suite();
        BatchScalarF64::<8>::test_suite();
    }
}
