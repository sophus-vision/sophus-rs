use crate::calculus::dual::DualMatrix;
use crate::calculus::dual::DualScalar;
use crate::calculus::dual::DualVector;
use crate::linalg::MatF64;
use crate::linalg::VecF64;
use crate::linalg::EPS_F64;
use crate::prelude::*;
use approx::assert_abs_diff_eq;
use approx::AbsDiffEq;
use approx::RelativeEq;
use nalgebra::SimdValue;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;
use std::ops::SubAssign;

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

/// Trait for scalar and batch scalar linalg
pub trait IsCoreScalar:
    Clone + Debug + nalgebra::Scalar + num_traits::Zero + std::ops::AddAssign
{
    /// Get the number category
    fn number_category() -> NumberCategory;
}

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

/// Scalar trait
///
///  - either a real (f64) or a dual number
///  - either a single scalar or a batch scalar
pub trait IsScalar<const BATCH_SIZE: usize>:
    PartialEq
    + Debug
    + Clone
    + std::ops::Div<Output = Self>
    + Add<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
    + AddAssign
    + SubAssign
    + Sized
    + Neg<Output = Self>
    + AbsDiffEq<Epsilon = f64>
    + RelativeEq<Epsilon = f64>
    + IsCoreScalar
{
    /// Scalar type
    type Scalar: IsScalar<BATCH_SIZE>;

    /// Single scalar type
    type SingleScalar: IsSingleScalar;

    /// Real scalar type
    type RealScalar: IsRealScalar<BATCH_SIZE>;

    /// Dual scalar type
    type DualScalar: IsDualScalar<BATCH_SIZE>;

    /// Mask type
    type Mask: IsBoolMask;

    /// Vector type
    type RealMatrix<const ROWS: usize, const COLS: usize>: IsRealMatrix<
        Self::RealScalar,
        ROWS,
        COLS,
        BATCH_SIZE,
    >;

    /// Vector type
    type Vector<const ROWS: usize>: IsVector<Self, ROWS, BATCH_SIZE>;

    /// Real vector type
    type RealVector<const ROWS: usize>: IsRealVector<Self::RealScalar, ROWS, BATCH_SIZE>;

    /// Dual vector type
    type DualVector<const ROWS: usize>: IsDualVector<Self::DualScalar, ROWS, BATCH_SIZE>;

    /// Matrix type
    type Matrix<const ROWS: usize, const COLS: usize>: IsMatrix<Self, ROWS, COLS, BATCH_SIZE>;

    /// Dual matrix type
    type DualMatrix<const ROWS: usize, const COLS: usize>: IsDualMatrix<
        Self::DualScalar,
        ROWS,
        COLS,
        BATCH_SIZE,
    >;

    /// absolute value
    fn abs(self) -> Self;

    /// arccosine
    fn acos(self) -> Self;

    /// arcsine
    fn asin(self) -> Self;

    /// arctangent
    fn atan(self) -> Self;

    /// arctangent2
    fn atan2(self, x: Self) -> Self;

    /// cosine
    fn cos(self) -> Self;

    /// eps
    fn eps() -> Self;

    /// Returns value of single lane
    fn extract_single(&self, i: usize) -> Self::SingleScalar;

    /// floor
    fn floor(&self) -> Self::RealScalar;

    /// fractional part
    fn fract(self) -> Self;

    /// Creates a scalar with all real lanes set the given value
    ///
    /// If self is a dual number, the infinitesimal part is set to zero
    fn from_f64(val: f64) -> Self;

    /// Creates a scalar from an array of real values
    ///
    ///  - If self is a single scalar, the array must have one element
    ///  - If self is a batch scalar, the array must have BATCH_SIZE elements
    ///  - If self is a dual number, the infinitesimal part is set to zero
    fn from_real_array(arr: [f64; BATCH_SIZE]) -> Self;

    /// creates scalar from real scalar
    ///
    /// for dual numbers, the infinitesimal part is set to zero
    fn from_real_scalar(val: Self::RealScalar) -> Self;

    /// Greater or equal comparison
    fn greater_equal(&self, rhs: &Self) -> Self::Mask;

    /// Less or equal comparison
    fn less_equal(&self, rhs: &Self) -> Self::Mask;

    /// ones
    fn ones() -> Self {
        Self::from_f64(1.0)
    }

    /// return the real part
    fn real_part(&self) -> Self::RealScalar;

    /// return examples of scalar values
    fn scalar_examples() -> Vec<Self>;

    /// Return the self if the mask is true, otherwise the other value
    ///
    /// This is a lane-wise operation
    fn select(self, mask: &Self::Mask, other: Self) -> Self;

    /// Return the sign of the scalar
    ///
    /// -1 if negative including -0,
    /// 1 if positive, including +0,
    /// NaN if NaN
    fn signum(&self) -> Self;

    /// sine
    fn sin(self) -> Self;

    /// square root
    fn sqrt(self) -> Self;

    /// Returns dual number representation
    ///
    /// If self is a real number, the infinitesimal part is zero: (self, 0ϵ)
    fn to_dual(self) -> Self::DualScalar;

    /// Return as a real array
    ///
    /// If self is a dual number, the infinitesimal part is omitted
    fn to_real_array(&self) -> [f64; BATCH_SIZE];

    /// tangent
    fn tan(self) -> Self;

    /// return as a vector
    fn to_vec(self) -> Self::Vector<1>;

    /// zeros
    fn zeros() -> Self {
        Self::from_f64(0.0)
    }

    /// test suite
    fn test_suite() {
        let examples = Self::scalar_examples();
        for a in &examples {
            let sin_a = a.clone().sin();
            let cos_a = a.clone().cos();
            let val = sin_a.clone() * sin_a + cos_a.clone() * cos_a;
            let one = Self::ones();

            for i in 0..BATCH_SIZE {
                assert_abs_diff_eq!(val.extract_single(i), one.extract_single(i));
            }
        }
    }
}

/// Real scalar
pub trait IsRealScalar<const BATCH_SIZE: usize>: IsScalar<BATCH_SIZE> + Copy {}

/// Scalar
pub trait IsSingleScalar: IsScalar<1> + PartialEq + Div<Output = Self> {
    /// Scalar vector type
    type SingleVector<const ROWS: usize>: IsSingleVector<Self, ROWS>;

    /// Matrix type
    type SingleMatrix<const ROWS: usize, const COLS: usize>: IsSingleMatrix<Self, ROWS, COLS>;

    /// returns single real scalar
    fn single_real_scalar(&self) -> f64;

    /// returns single real scalar
    fn single_scalar(&self) -> Self;

    /// get element
    fn i64_floor(&self) -> i64;
}

/// Batch scalar
pub trait IsBatchScalar<const BATCH_SIZE: usize>: IsScalar<BATCH_SIZE> {}

impl IsRealScalar<1> for f64 {}

impl IsScalar<1> for f64 {
    type Scalar = f64;
    type RealScalar = f64;
    type SingleScalar = f64;
    type DualScalar = DualScalar;
    type Vector<const ROWS: usize> = VecF64<ROWS>;
    type Matrix<const ROWS: usize, const COLS: usize> = MatF64<ROWS, COLS>;
    type RealVector<const ROWS: usize> = VecF64<ROWS>;
    type RealMatrix<const ROWS: usize, const COLS: usize> = MatF64<ROWS, COLS>;

    type Mask = bool;

    fn less_equal(&self, rhs: &Self) -> Self::Mask {
        self <= rhs
    }

    fn greater_equal(&self, rhs: &Self) -> Self::Mask {
        self >= rhs
    }

    fn scalar_examples() -> Vec<f64> {
        vec![1.0, 2.0, 3.0]
    }

    fn abs(self) -> f64 {
        f64::abs(self)
    }

    fn cos(self) -> f64 {
        f64::cos(self)
    }

    fn eps() -> f64 {
        EPS_F64
    }

    fn sin(self) -> f64 {
        f64::sin(self)
    }

    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }

    fn from_f64(val: f64) -> f64 {
        val
    }

    fn from_real_scalar(val: f64) -> f64 {
        val
    }

    fn atan2(self, x: Self) -> Self {
        self.atan2(x)
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

    fn to_vec(self) -> VecF64<1> {
        VecF64::<1>::new(self)
    }

    fn tan(self) -> Self {
        self.tan()
    }

    fn acos(self) -> Self {
        self.acos()
    }

    fn asin(self) -> Self {
        self.asin()
    }

    fn atan(self) -> Self {
        self.atan()
    }

    fn fract(self) -> Self {
        f64::fract(self)
    }

    fn floor(&self) -> f64 {
        f64::floor(*self)
    }

    fn extract_single(&self, i: usize) -> f64 {
        self.extract(i)
    }

    fn signum(&self) -> Self {
        f64::signum(*self)
    }

    type DualVector<const ROWS: usize> = DualVector<ROWS>;

    type DualMatrix<const ROWS: usize, const COLS: usize> = DualMatrix<ROWS, COLS>;

    fn to_dual(self) -> Self::DualScalar {
        DualScalar::from_f64(self)
    }

    fn select(self, mask: &Self::Mask, other: Self) -> Self {
        if *mask {
            self
        } else {
            other
        }
    }
}

impl IsSingleScalar for f64 {
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

    f64::test_suite();

    #[cfg(feature = "simd")]
    BatchScalarF64::<2>::test_suite();
    #[cfg(feature = "simd")]
    BatchScalarF64::<4>::test_suite();
    #[cfg(feature = "simd")]
    BatchScalarF64::<8>::test_suite();
}
