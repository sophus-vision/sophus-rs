use crate::calculus::dual::DualBatchMatrix;
use crate::calculus::dual::DualBatchScalar;
use crate::calculus::dual::DualBatchVector;
use crate::calculus::dual::DualMatrix;
use crate::calculus::dual::DualScalar;
use crate::calculus::dual::DualVector;
use crate::linalg::BatchMatF64;
use crate::linalg::BatchScalar;
use crate::linalg::BatchScalarF64;
use crate::linalg::BatchVecF64;
use crate::linalg::MatF64;
use crate::linalg::VecF64;
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
use std::ops::MulAssign;
use std::ops::Neg;
use std::ops::Sub;
use std::ops::SubAssign;
use std::simd::cmp::SimdPartialOrd;
use std::simd::num::SimdFloat;
use std::simd::LaneCount;
use std::simd::Mask;
use std::simd::Simd;
use std::simd::SimdElement;
use std::simd::StdFloat;
use std::simd::SupportedLaneCount;

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

impl<S: SimdElement + IsCoreScalar, const BATCH_SIZE: usize> IsCoreScalar
    for BatchScalar<S, BATCH_SIZE>
where
    LaneCount<BATCH_SIZE>: SupportedLaneCount,
    Simd<S, BATCH_SIZE>: SimdFloat,
    BatchScalar<S, BATCH_SIZE>:
        Clone + Debug + nalgebra::Scalar + num_traits::Zero + std::ops::AddAssign,
{
    fn number_category() -> NumberCategory {
        NumberCategory::Real
    }
}

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

impl<const BATCH: usize> AbsDiffEq for BatchScalarF64<BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Epsilon = f64;

    fn default_epsilon() -> f64 {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: f64) -> bool {
        for i in 0..BATCH {
            if !self.0[i].abs_diff_eq(&other.0[i], epsilon) {
                return false;
            }
        }
        true
    }
}

impl<const BATCH: usize> RelativeEq for BatchScalarF64<BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        for i in 0..BATCH {
            if !self.0[i].relative_eq(&other.0[i], epsilon, max_relative) {
                return false;
            }
        }
        true
    }
}

impl<const BATCH: usize> AddAssign for BatchScalarF64<BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl<const BATCH: usize> SubAssign for BatchScalarF64<BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl<const BATCH: usize> MulAssign for BatchScalarF64<BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl<const BATCH: usize> Neg for BatchScalarF64<BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = Self;

    fn neg(self) -> Self {
        BatchScalarF64 { 0: -self.0 }
    }
}

impl<const BATCH: usize> Sub for BatchScalarF64<BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        BatchScalarF64 { 0: self.0 - rhs.0 }
    }
}

impl<const BATCH: usize> Mul for BatchScalarF64<BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        BatchScalarF64 { 0: self.0 * rhs.0 }
    }
}

impl<const BATCH: usize> Div for BatchScalarF64<BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        BatchScalarF64 { 0: self.0 / rhs.0 }
    }
}

impl<const BATCH: usize> num_traits::One for BatchScalarF64<BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn one() -> Self {
        Self(Simd::<f64, BATCH>::splat(1.0))
    }
}
impl<const BATCH: usize> IsRealScalar<BATCH> for BatchScalarF64<BATCH> where
    LaneCount<BATCH>: SupportedLaneCount
{
}

impl<const BATCH: usize> IsBatchScalar<BATCH> for BatchScalarF64<BATCH> where
    LaneCount<BATCH>: SupportedLaneCount
{
}

impl<const BATCH: usize> IsScalar<BATCH> for BatchScalarF64<BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Scalar = BatchScalarF64<BATCH>;
    type RealScalar = Self;
    type SingleScalar = f64;
    type DualScalar = DualBatchScalar<BATCH>;

    type RealVector<const ROWS: usize> = BatchVecF64<ROWS, BATCH>;

    type Vector<const ROWS: usize> = BatchVecF64<ROWS, BATCH>;
    type Matrix<const ROWS: usize, const COLS: usize> = BatchMatF64<ROWS, COLS, BATCH>;
    type RealMatrix<const ROWS: usize, const COLS: usize> = BatchMatF64<ROWS, COLS, BATCH>;

    type Mask = Mask<i64, BATCH>;

    fn less_equal(&self, rhs: &Self) -> Self::Mask {
        self.0.simd_le(rhs.0)
    }

    fn greater_equal(&self, rhs: &Self) -> Self::Mask {
        self.0.simd_ge(rhs.0)
    }

    fn scalar_examples() -> Vec<BatchScalarF64<BATCH>> {
        vec![
            BatchScalarF64::<BATCH>::from_f64(1.0),
            BatchScalarF64::<BATCH>::from_f64(2.0),
            BatchScalarF64::<BATCH>::from_f64(3.0),
        ]
    }

    fn extract_single(&self, i: usize) -> f64 {
        self.0[i]
    }

    fn from_real_scalar(val: BatchScalarF64<BATCH>) -> Self {
        val
    }

    fn real_part(&self) -> Self {
        *self
    }

    fn from_f64(val: f64) -> Self {
        BatchScalarF64 {
            0: Simd::<f64, BATCH>::splat(val),
        }
    }

    fn abs(self) -> Self {
        BatchScalarF64 {
            0: SimdFloat::abs(self.0),
        }
    }

    fn from_real_array(arr: [f64; BATCH]) -> Self {
        BatchScalarF64 {
            0: Simd::<f64, BATCH>::from_array(arr),
        }
    }

    fn to_real_array(&self) -> [f64; BATCH] {
        self.0.to_array()
    }

    fn cos(self) -> Self {
        BatchScalarF64 { 0: self.0.cos() }
    }

    fn sin(self) -> Self {
        BatchScalarF64 { 0: self.0.sin() }
    }

    fn tan(self) -> Self {
        BatchScalarF64 {
            0: sleef::Sleef::tan(self.0),
        }
    }

    fn acos(self) -> Self {
        BatchScalarF64 {
            0: sleef::Sleef::acos(self.0),
        }
    }

    fn asin(self) -> Self {
        BatchScalarF64 {
            0: sleef::Sleef::asin(self.0),
        }
    }

    fn atan(self) -> Self {
        BatchScalarF64 {
            0: sleef::Sleef::atan(self.0),
        }
    }

    fn sqrt(self) -> Self {
        BatchScalarF64 { 0: self.0.sqrt() }
    }

    fn atan2(self, x: Self) -> Self {
        BatchScalarF64 {
            0: sleef::Sleef::atan2(self.0, x.0),
        }
    }

    fn to_vec(self) -> Self::Vector<1> {
        BatchVecF64::<1, BATCH>::from_scalar(self)
    }

    fn fract(self) -> Self {
        BatchScalarF64 { 0: self.0.fract() }
    }

    fn floor(&self) -> BatchScalarF64<BATCH> {
        BatchScalarF64 { 0: self.0.floor() }
    }

    fn signum(&self) -> Self {
        BatchScalarF64 { 0: self.0.signum() }
    }

    type DualVector<const ROWS: usize> = DualBatchVector<ROWS, BATCH>;

    type DualMatrix<const ROWS: usize, const COLS: usize> = DualBatchMatrix<ROWS, COLS, BATCH>;

    fn to_dual(self) -> Self::DualScalar {
        DualBatchScalar::from_real_scalar(self)
    }

    fn select(self, mask: &Self::Mask, other: Self) -> Self {
        BatchScalarF64 {
            0: mask.select(self.0, other.0),
        }
    }
}

#[test]
fn scalar_prop_tests() {
    f64::test_suite();
    BatchScalarF64::<2>::test_suite();
    BatchScalarF64::<4>::test_suite();
    BatchScalarF64::<8>::test_suite();
}
