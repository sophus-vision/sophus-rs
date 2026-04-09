use alloc::{
    vec,
    vec::Vec,
};
use core::{
    borrow::Borrow,
    fmt::Debug,
    ops::{
        AddAssign,
        Div,
        DivAssign,
        Mul,
        MulAssign,
        Neg,
        Sub,
        SubAssign,
    },
    simd::{
        Select,
        Simd,
        SimdElement,
        cmp::SimdPartialOrd,
        num::SimdFloat,
    },
};

use approx::{
    AbsDiffEq,
    RelativeEq,
};
use sleef::Sleef;

use super::batch_mask::{
    BatchMask,
    BatchMaskF32,
};
use crate::{
    dual::{
        DualBatchMatrix,
        DualBatchScalar,
        DualBatchVector,
    },
    linalg::{
        BatchMatF32,
        BatchMatF64,
        BatchScalar,
        BatchScalarF32,
        BatchScalarF64,
        BatchVecF32,
        BatchVecF64,
        EPS_F32,
        EPS_F64,
        scalar::{
            IsBatchScalar,
            NumberCategory,
        },
    },
    prelude::{
        IsCoreScalar,
        *,
    },
};

extern crate alloc;

impl<S: SimdElement + IsCoreScalar, const BATCH: usize> IsCoreScalar for BatchScalar<S, BATCH>
where
    Simd<S, BATCH>: SimdFloat,
    BatchScalar<S, BATCH>:
        Clone + Debug + nalgebra::Scalar + num_traits::Zero + core::ops::AddAssign,
{
    fn number_category() -> NumberCategory {
        NumberCategory::Real
    }
}

impl<const BATCH: usize> AbsDiffEq for BatchScalarF64<BATCH> {
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

impl<const BATCH: usize> RelativeEq for BatchScalarF64<BATCH> {
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

impl<const BATCH: usize> AddAssign for BatchScalarF64<BATCH> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl<const BATCH: usize> SubAssign for BatchScalarF64<BATCH> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl<const BATCH: usize> MulAssign for BatchScalarF64<BATCH> {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl<const BATCH: usize> DivAssign for BatchScalarF64<BATCH> {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl<const BATCH: usize> Neg for BatchScalarF64<BATCH> {
    type Output = Self;

    fn neg(self) -> Self {
        BatchScalarF64 { 0: -self.0 }
    }
}

impl<const BATCH: usize> Sub for BatchScalarF64<BATCH> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        BatchScalarF64 { 0: self.0 - rhs.0 }
    }
}

impl<const BATCH: usize> Mul for BatchScalarF64<BATCH> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        BatchScalarF64 { 0: self.0 * rhs.0 }
    }
}

impl<const BATCH: usize> Div for BatchScalarF64<BATCH> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        BatchScalarF64 { 0: self.0 / rhs.0 }
    }
}

impl<const BATCH: usize> num_traits::One for BatchScalarF64<BATCH> {
    fn one() -> Self {
        Self(Simd::<f64, BATCH>::splat(1.0))
    }
}
impl<const BATCH: usize> IsRealScalar<BATCH> for BatchScalarF64<BATCH> {}

impl<const BATCH: usize> IsBatchScalar<BATCH, 0, 0> for BatchScalarF64<BATCH> {}

impl<const BATCH: usize> IsScalar<BATCH, 0, 0> for BatchScalarF64<BATCH> {
    type Scalar = BatchScalarF64<BATCH>;
    type Vector<const ROWS: usize> = BatchVecF64<ROWS, BATCH>;
    type Matrix<const ROWS: usize, const COLS: usize> = BatchMatF64<ROWS, COLS, BATCH>;

    type SingleScalar = f64;

    type RealScalar = Self;
    type RealSingleScalar = f64;
    type RealVector<const ROWS: usize> = BatchVecF64<ROWS, BATCH>;
    type RealMatrix<const ROWS: usize, const COLS: usize> = BatchMatF64<ROWS, COLS, BATCH>;

    type DualScalar<const M: usize, const N: usize> = DualBatchScalar<BATCH, M, N>;
    type DualVector<const ROWS: usize, const M: usize, const N: usize> =
        DualBatchVector<ROWS, BATCH, M, N>;
    type DualMatrix<const ROWS: usize, const COLS: usize, const M: usize, const N: usize> =
        DualBatchMatrix<ROWS, COLS, BATCH, M, N>;

    type Mask = BatchMask<BATCH>;

    fn less_equal(&self, rhs: &Self) -> Self::Mask {
        BatchMask {
            inner: self.0.simd_le(rhs.0),
        }
    }

    fn greater_equal(&self, rhs: &Self) -> Self::Mask {
        BatchMask {
            inner: self.0.simd_ge(rhs.0),
        }
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

    fn eps() -> Self {
        Self::from_f64(EPS_F64)
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

    fn abs(&self) -> Self {
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

    fn cos(&self) -> Self {
        BatchScalarF64 {
            0: sleef::Sleef::cos(self.0),
        }
    }

    fn exp(&self) -> Self {
        BatchScalarF64 {
            0: sleef::Sleef::exp(self.0),
        }
    }

    fn ln(&self) -> Self {
        BatchScalarF64 {
            0: sleef::Sleef::ln(self.0),
        }
    }

    fn sin(&self) -> Self {
        BatchScalarF64 {
            0: sleef::Sleef::sin(self.0),
        }
    }

    fn tan(&self) -> Self {
        BatchScalarF64 {
            0: sleef::Sleef::tan(self.0),
        }
    }

    fn tanh(&self) -> Self {
        BatchScalarF64 {
            0: sleef::Sleef::tanh(self.0),
        }
    }

    fn acos(&self) -> Self {
        BatchScalarF64 {
            0: sleef::Sleef::acos(self.0),
        }
    }

    fn asin(&self) -> Self {
        BatchScalarF64 {
            0: sleef::Sleef::asin(self.0),
        }
    }

    fn atan(&self) -> Self {
        BatchScalarF64 {
            0: sleef::Sleef::atan(self.0),
        }
    }

    fn sqrt(&self) -> Self {
        BatchScalarF64 { 0: self.0.sqrt() }
    }

    fn atan2<S>(&self, rhs: S) -> Self
    where
        S: Borrow<Self>,
    {
        let rhs = rhs.borrow();
        BatchScalarF64 {
            0: sleef::Sleef::atan2(self.0, rhs.0),
        }
    }

    fn to_vec(&self) -> Self::Vector<1> {
        BatchVecF64::<1, BATCH>::from_scalar(self)
    }

    fn fract(&self) -> Self {
        BatchScalarF64 {
            0: self.0.frfrexp(),
        }
    }

    fn floor(&self) -> BatchScalarF64<BATCH> {
        BatchScalarF64 { 0: self.0.floor() }
    }

    fn signum(&self) -> Self {
        BatchScalarF64 { 0: self.0.signum() }
    }

    fn to_dual_const<const M: usize, const N: usize>(&self) -> Self::DualScalar<M, N> {
        DualBatchScalar::from_real_scalar(*self)
    }

    fn select(&self, mask: &Self::Mask, other: Self) -> Self {
        BatchScalarF64 {
            0: mask.inner.select(self.0, other.0),
        }
    }
}

impl<const BATCH: usize> AbsDiffEq for BatchScalarF32<BATCH> {
    type Epsilon = f32;

    fn default_epsilon() -> f32 {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: f32) -> bool {
        for i in 0..BATCH {
            if !self.0[i].abs_diff_eq(&other.0[i], epsilon) {
                return false;
            }
        }
        true
    }
}

impl<const BATCH: usize> RelativeEq for BatchScalarF32<BATCH> {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
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

impl<const BATCH: usize> AddAssign for BatchScalarF32<BATCH> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl<const BATCH: usize> SubAssign for BatchScalarF32<BATCH> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl<const BATCH: usize> MulAssign for BatchScalarF32<BATCH> {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl<const BATCH: usize> DivAssign for BatchScalarF32<BATCH> {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl<const BATCH: usize> Neg for BatchScalarF32<BATCH> {
    type Output = Self;

    fn neg(self) -> Self {
        BatchScalarF32 { 0: -self.0 }
    }
}

impl<const BATCH: usize> Sub for BatchScalarF32<BATCH> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        BatchScalarF32 { 0: self.0 - rhs.0 }
    }
}

impl<const BATCH: usize> Mul for BatchScalarF32<BATCH> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        BatchScalarF32 { 0: self.0 * rhs.0 }
    }
}

impl<const BATCH: usize> Div for BatchScalarF32<BATCH> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        BatchScalarF32 { 0: self.0 / rhs.0 }
    }
}

impl<const BATCH: usize> num_traits::One for BatchScalarF32<BATCH> {
    fn one() -> Self {
        Self(Simd::<f32, BATCH>::splat(1.0_f32))
    }
}

impl<const BATCH: usize> IsRealScalar<BATCH> for BatchScalarF32<BATCH> {}

impl<const BATCH: usize> IsBatchScalar<BATCH, 0, 0> for BatchScalarF32<BATCH> {}

impl<const BATCH: usize> IsScalar<BATCH, 0, 0> for BatchScalarF32<BATCH> {
    type Scalar = BatchScalarF32<BATCH>;
    type Vector<const ROWS: usize> = BatchVecF32<ROWS, BATCH>;
    type Matrix<const ROWS: usize, const COLS: usize> = BatchMatF32<ROWS, COLS, BATCH>;

    type SingleScalar = f32;

    type RealScalar = Self;
    type RealSingleScalar = f32;
    type RealVector<const ROWS: usize> = BatchVecF32<ROWS, BATCH>;
    type RealMatrix<const ROWS: usize, const COLS: usize> = BatchMatF32<ROWS, COLS, BATCH>;

    // f32 AD is unsupported; use f64-based dual types as placeholder.
    type DualScalar<const M: usize, const N: usize> = DualBatchScalar<BATCH, M, N>;
    type DualVector<const ROWS: usize, const M: usize, const N: usize> =
        DualBatchVector<ROWS, BATCH, M, N>;
    type DualMatrix<const ROWS: usize, const COLS: usize, const M: usize, const N: usize> =
        DualBatchMatrix<ROWS, COLS, BATCH, M, N>;

    type Mask = BatchMaskF32<BATCH>;

    fn less_equal(&self, rhs: &Self) -> Self::Mask {
        BatchMaskF32 {
            inner: self.0.simd_le(rhs.0),
        }
    }

    fn greater_equal(&self, rhs: &Self) -> Self::Mask {
        BatchMaskF32 {
            inner: self.0.simd_ge(rhs.0),
        }
    }

    fn scalar_examples() -> Vec<BatchScalarF32<BATCH>> {
        vec![
            BatchScalarF32::<BATCH>::from_f64(1.0),
            BatchScalarF32::<BATCH>::from_f64(2.0),
            BatchScalarF32::<BATCH>::from_f64(3.0),
        ]
    }

    fn extract_single(&self, i: usize) -> f32 {
        self.0[i]
    }

    fn eps() -> Self {
        Self::from_f64(EPS_F32 as f64)
    }

    fn from_real_scalar(val: BatchScalarF32<BATCH>) -> Self {
        val
    }

    fn real_part(&self) -> Self {
        *self
    }

    fn from_f64(val: f64) -> Self {
        BatchScalarF32 {
            0: Simd::<f32, BATCH>::splat(val as f32),
        }
    }

    fn abs(&self) -> Self {
        BatchScalarF32 {
            0: SimdFloat::abs(self.0),
        }
    }

    fn from_real_array(arr: [f32; BATCH]) -> Self {
        BatchScalarF32 {
            0: Simd::<f32, BATCH>::from_array(arr),
        }
    }

    fn to_real_array(&self) -> [f32; BATCH] {
        self.0.to_array()
    }

    fn cos(&self) -> Self {
        BatchScalarF32 {
            0: sleef::Sleef::cos(self.0),
        }
    }

    fn exp(&self) -> Self {
        BatchScalarF32 {
            0: sleef::Sleef::exp(self.0),
        }
    }

    fn ln(&self) -> Self {
        BatchScalarF32 {
            0: sleef::Sleef::ln(self.0),
        }
    }

    fn sin(&self) -> Self {
        BatchScalarF32 {
            0: sleef::Sleef::sin(self.0),
        }
    }

    fn tan(&self) -> Self {
        BatchScalarF32 {
            0: sleef::Sleef::tan(self.0),
        }
    }

    fn tanh(&self) -> Self {
        BatchScalarF32 {
            0: sleef::Sleef::tanh(self.0),
        }
    }

    fn acos(&self) -> Self {
        BatchScalarF32 {
            0: sleef::Sleef::acos(self.0),
        }
    }

    fn asin(&self) -> Self {
        BatchScalarF32 {
            0: sleef::Sleef::asin(self.0),
        }
    }

    fn atan(&self) -> Self {
        BatchScalarF32 {
            0: sleef::Sleef::atan(self.0),
        }
    }

    fn sqrt(&self) -> Self {
        BatchScalarF32 { 0: self.0.sqrt() }
    }

    fn atan2<S>(&self, rhs: S) -> Self
    where
        S: Borrow<Self>,
    {
        let rhs = rhs.borrow();
        BatchScalarF32 {
            0: sleef::Sleef::atan2(self.0, rhs.0),
        }
    }

    fn to_vec(&self) -> Self::Vector<1> {
        BatchVecF32::<1, BATCH>::from_scalar(self)
    }

    fn fract(&self) -> Self {
        BatchScalarF32 {
            0: self.0.frfrexp(),
        }
    }

    fn floor(&self) -> BatchScalarF32<BATCH> {
        BatchScalarF32 { 0: self.0.floor() }
    }

    fn signum(&self) -> Self {
        BatchScalarF32 { 0: self.0.signum() }
    }

    fn to_dual_const<const M: usize, const N: usize>(&self) -> Self::DualScalar<M, N> {
        // f32 AD is not supported; cast to f64 batch scalar for placeholder.
        let f64_val =
            BatchScalarF64::<BATCH>::from_real_array(core::array::from_fn(|i| self.0[i] as f64));
        DualBatchScalar::from_real_scalar(f64_val)
    }

    fn select(&self, mask: &Self::Mask, other: Self) -> Self {
        BatchScalarF32 {
            0: mask.inner.select(self.0, other.0),
        }
    }
}
