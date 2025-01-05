use crate::calculus::dual::DualBatchMatrix;
use crate::calculus::dual::DualBatchScalar;
use crate::calculus::dual::DualBatchVector;
use crate::linalg::scalar::IsBatchScalar;
use crate::linalg::scalar::NumberCategory;
use crate::linalg::BatchMatF64;
use crate::linalg::BatchScalar;
use crate::linalg::BatchScalarF64;
use crate::linalg::BatchVecF64;
use crate::linalg::EPS_F64;
use crate::prelude::IsCoreScalar;
use crate::prelude::*;
use alloc::vec;
use alloc::vec::Vec;
use approx::AbsDiffEq;
use approx::RelativeEq;
use core::borrow::Borrow;
use core::fmt::Debug;
use core::ops::AddAssign;
use core::ops::Div;
use core::ops::DivAssign;
use core::ops::Mul;
use core::ops::MulAssign;
use core::ops::Neg;
use core::ops::Sub;
use core::ops::SubAssign;
use core::simd::cmp::SimdPartialOrd;
use core::simd::num::SimdFloat;
use core::simd::LaneCount;
use core::simd::Simd;
use core::simd::SimdElement;
use core::simd::SupportedLaneCount;
use sleef::Sleef;

use super::batch_mask::BatchMask;

extern crate alloc;

impl<S: SimdElement + IsCoreScalar, const BATCH: usize> IsCoreScalar for BatchScalar<S, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
    Simd<S, BATCH>: SimdFloat,
    BatchScalar<S, BATCH>:
        Clone + Debug + nalgebra::Scalar + num_traits::Zero + core::ops::AddAssign,
{
    fn number_category() -> NumberCategory {
        NumberCategory::Real
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

impl<const BATCH: usize> DivAssign for BatchScalarF64<BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
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

impl<const BATCH: usize> IsBatchScalar<BATCH, 0, 0> for BatchScalarF64<BATCH> where
    LaneCount<BATCH>: SupportedLaneCount
{
}

impl<const BATCH: usize> IsScalar<BATCH, 0, 0> for BatchScalarF64<BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Scalar = BatchScalarF64<BATCH>;
    type Vector<const ROWS: usize> = BatchVecF64<ROWS, BATCH>;
    type Matrix<const ROWS: usize, const COLS: usize> = BatchMatF64<ROWS, COLS, BATCH>;

    type SingleScalar = f64;

    type RealScalar = Self;
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
