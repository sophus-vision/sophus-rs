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
    AbsDiffEq,
    RelativeEq,
};
use nalgebra::SMatrix;
use num_traits::{
    One,
    Zero,
};

use super::{
    dual_matrix::DualMatrix,
    dual_vector::DualVector,
    scalar::IsDualScalarFromCurve,
};
use crate::{
    linalg::{
        NumberCategory,
        SVec,
    },
    prelude::*,
};

extern crate alloc;

/// A dual-number scalar, storing both a real part and an *optional* infinitesimal part
/// (the derivative or Jacobian block).
///
/// # Structure
/// - `real_part` (`R`): The main (real) value.
/// - `infinitesimal_part` (`Option<SMatrix<R, DM, DN>>`): Stores the derivative information, shaped
///   `[DM × DN]`.
///   - If `DM=1, DN=1`, this represents a single partial derivative d/dx.
///   - If `DM>1` or `DN>1`, it can represent more complex derivatives (e.g., gradients, Jacobians).
///
/// # Generic Parameters
/// - `R`: The underlying real scalar type (e.g. `f64` or `f32`).
/// - `DM`: The number of derivative rows.
/// - `DN`: The number of derivative columns.
///
/// For instance, `DualScalar<f64, 3, 1>` might store partials w.r.t. a 3D input, resulting in a
/// 3×1 derivative for each scalar.
///
/// See [crate::dual::IsDualScalar] for more details.
#[derive(Clone, Debug, Copy)]
pub struct DualScalar<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> {
    /// The real (non-infinitesimal) part of the dual number.
    pub real_part: R,

    /// The infinitesimal part, storing derivative information if present.
    ///
    /// If `None`, the derivative is assumed to be zero.
    pub infinitesimal_part: Option<SMatrix<R, DM, DN>>,
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> AbsDiffEq for DualScalar<R, DM, DN>
{
    type Epsilon = R;

    fn default_epsilon() -> Self::Epsilon {
        <R as IsScalar<1, 0, 0>>::eps()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.real_part.abs_diff_eq(&other.real_part, epsilon)
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> RelativeEq for DualScalar<R, DM, DN>
{
    fn default_max_relative() -> Self::Epsilon {
        <R as IsScalar<1, 0, 0>>::eps()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.real_part
            .relative_eq(&other.real_part, epsilon, max_relative)
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> IsCoreScalar for DualScalar<R, DM, DN>
{
    fn number_category() -> NumberCategory {
        // We treat dual numbers as "Real" in the sense that they
        // can represent floating scalars with derivatives.
        NumberCategory::Real
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> SubAssign<DualScalar<R, DM, DN>> for DualScalar<R, DM, DN>
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = (*self).sub(&rhs);
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> IsSingleScalar<DM, DN> for DualScalar<R, DM, DN>
{
    type SingleVector<const ROWS: usize> = DualVector<R, ROWS, DM, DN>;
    type SingleMatrix<const ROWS: usize, const COLS: usize> = DualMatrix<R, ROWS, COLS, DM, DN>;

    fn single_real_scalar(&self) -> R {
        self.real_part
    }

    fn single_scalar(&self) -> Self {
        *self
    }

    fn i64_floor(&self) -> i64 {
        self.real_part.i64_floor()
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> AsRef<DualScalar<R, DM, DN>> for DualScalar<R, DM, DN>
{
    fn as_ref(&self) -> &DualScalar<R, DM, DN> {
        self
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> One for DualScalar<R, DM, DN>
{
    fn one() -> Self {
        Self::from_f64(1.0)
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> Zero for DualScalar<R, DM, DN>
{
    fn zero() -> Self {
        Self::from_f64(0.0)
    }

    fn is_zero(&self) -> bool {
        self.real_part == <R as IsScalar<1, 0, 0>>::from_f64(0.0)
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> IsDualScalar<1, DM, DN> for DualScalar<R, DM, DN>
{
    fn var(val: R) -> Self {
        // Creating a "variable" means setting derivative=identity w.r.t. this scalar.
        Self {
            real_part: val,
            infinitesimal_part: Some(SMatrix::<R, DM, DN>::from_element(<R as IsScalar<
                1,
                0,
                0,
            >>::from_f64(
                1.0
            ))),
        }
    }

    fn vector_var<const ROWS: usize>(val: Self::RealVector<ROWS>) -> Self::Vector<ROWS> {
        DualVector::<R, ROWS, DM, DN>::var(val)
    }

    fn matrix_var<const ROWS: usize, const COLS: usize>(
        val: Self::RealMatrix<ROWS, COLS>,
    ) -> Self::Matrix<ROWS, COLS> {
        DualMatrix::<R, ROWS, COLS, DM, DN>::var(val)
    }

    fn derivative(&self) -> nalgebra::SMatrix<R, DM, DN> {
        self.infinitesimal_part
            .unwrap_or_else(SMatrix::<R, DM, DN>::zeros)
    }
}

impl<R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>>
    IsDualScalarFromCurve<DualScalar<R, 1, 1>, 1> for DualScalar<R, 1, 1>
{
    fn curve_derivative(&self) -> R {
        // The derivative matrix is 1×1, so we just return that single element.
        self.derivative()[0]
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> DualScalar<R, DM, DN>
{
    /// Internal helper to combine derivative parts from two dual scalars during binary ops.
    fn binary_dij(
        lhs_dx: &Option<SMatrix<R, DM, DN>>,
        rhs_dx: &Option<SMatrix<R, DM, DN>>,
        mut left_op: impl FnMut(&SMatrix<R, DM, DN>) -> SMatrix<R, DM, DN>,
        mut right_op: impl FnMut(&SMatrix<R, DM, DN>) -> SMatrix<R, DM, DN>,
    ) -> Option<SMatrix<R, DM, DN>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => Some(right_op(rhs_dij)),
            (Some(lhs_dij), None) => Some(left_op(lhs_dij)),
            (Some(lhs_dij), Some(rhs_dij)) => Some(left_op(lhs_dij) + right_op(rhs_dij)),
        }
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> Neg for DualScalar<R, DM, DN>
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            real_part: -self.real_part,
            infinitesimal_part: self.infinitesimal_part.map(|dij_val| -dij_val),
        }
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> PartialEq for DualScalar<R, DM, DN>
{
    fn eq(&self, other: &Self) -> bool {
        self.real_part == other.real_part
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> PartialOrd for DualScalar<R, DM, DN>
{
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.real_part.partial_cmp(&other.real_part)
    }
}

impl<R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>> From<f64>
    for DualScalar<R, 0, 0>
{
    fn from(value: f64) -> Self {
        Self::from_real_scalar(<R as IsScalar<1, 0, 0>>::from_f64(value))
    }
}

/// Defines the main scalar trait for `DualScalar<R, DM, DN>` in single-lane usage (BATCH=1).
impl<R, const DM: usize, const DN: usize> IsScalar<1, DM, DN> for DualScalar<R, DM, DN>
where
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    R::RealVector<1>: IsRealVector<R, 1, 1>,
{
    type Scalar = Self;
    type Vector<const ROWS: usize> = DualVector<R, ROWS, DM, DN>;
    type Matrix<const ROWS: usize, const COLS: usize> = DualMatrix<R, ROWS, COLS, DM, DN>;

    type RealScalar = R;
    type RealSingleScalar = R;
    type RealMatrix<const ROWS: usize, const COLS: usize> = R::RealMatrix<ROWS, COLS>;
    type RealVector<const ROWS: usize> = R::RealVector<ROWS>;

    type SingleScalar = Self;

    type DualScalar<const M: usize, const N: usize> = DualScalar<R, M, N>;
    type DualVector<const ROWS: usize, const M: usize, const N: usize> = DualVector<R, ROWS, M, N>;
    type DualMatrix<const ROWS: usize, const COLS: usize, const M: usize, const N: usize> =
        DualMatrix<R, ROWS, COLS, M, N>;

    type Mask = bool;

    fn from_real_scalar(val: R) -> Self {
        Self {
            real_part: val,
            infinitesimal_part: None,
        }
    }

    fn from_real_array(arr: [R; 1]) -> Self {
        Self::from_real_scalar(arr[0])
    }

    fn to_real_array(&self) -> [R; 1] {
        [self.real_part]
    }

    fn cos(&self) -> Self {
        Self {
            real_part: self.real_part.cos(),
            // d/dx (cos(x)) = -sin(x)
            infinitesimal_part: self
                .infinitesimal_part
                .map(|dij_val| -dij_val * self.real_part.sin()),
        }
    }

    fn exp(&self) -> Self {
        Self {
            real_part: self.real_part.exp(),
            // d/dx (exp(x)) = exp(x)
            infinitesimal_part: self
                .infinitesimal_part
                .map(|dij_val| dij_val * self.real_part.exp()),
        }
    }

    fn ln(&self) -> Self {
        Self {
            real_part: self.real_part.ln(),
            // d/dx (ln(x)) = 1/x
            infinitesimal_part: self
                .infinitesimal_part
                .map(|dij_val| dij_val.map(|d| d / self.real_part)),
        }
    }

    fn sin(&self) -> Self {
        Self {
            real_part: self.real_part.sin(),
            // d/dx (sin(x)) = cos(x)
            infinitesimal_part: self
                .infinitesimal_part
                .map(|dij_val| dij_val * self.real_part.cos()),
        }
    }

    fn abs(&self) -> Self {
        Self {
            real_part: self.real_part.abs(),
            // derivative wrt x of abs(x) is signum(x), but that is not strictly differentiable at
            // x=0. We'll assume signum-based for x != 0, or break for x=0.
            infinitesimal_part: self
                .infinitesimal_part
                .map(|dij_val| dij_val * self.real_part.signum()),
        }
    }

    fn atan2<S>(&self, rhs: S) -> Self
    where
        S: Borrow<Self>,
    {
        let rhs = rhs.borrow();
        // The usual derivative of atan2(y, x) wrt y => x / (x^2 + y^2), wrt x => -y / (x^2 + y^2)
        let inv_sq_nrm = <R as IsScalar<1, 0, 0>>::from_f64(1.0)
            / (self.real_part * self.real_part + rhs.real_part * rhs.real_part);
        Self {
            real_part: self.real_part.atan2(rhs.real_part),
            infinitesimal_part: Self::binary_dij(
                &self.infinitesimal_part,
                &rhs.infinitesimal_part,
                |l_dij| l_dij * (inv_sq_nrm * rhs.real_part),
                |r_dij| r_dij * (-inv_sq_nrm * self.real_part),
            ),
        }
    }

    fn real_part(&self) -> R {
        self.real_part
    }

    fn sqrt(&self) -> Self {
        let sqrt_val = self.real_part.sqrt();
        // d/dx of sqrt(x) = 1 / (2 sqrt(x))
        let half_inv_sqrt = <R as IsScalar<1, 0, 0>>::from_f64(1.0)
            / (<R as IsScalar<1, 0, 0>>::from_f64(2.0) * sqrt_val);
        Self {
            real_part: sqrt_val,
            infinitesimal_part: self.infinitesimal_part.map(|dij| dij * half_inv_sqrt),
        }
    }

    fn to_vec(&self) -> DualVector<R, 1, DM, DN> {
        DualVector::<R, 1, DM, DN> {
            inner: SVec::<Self, 1>::from_element(*self),
        }
    }

    fn tan(&self) -> Self {
        let cos_val = self.real_part.cos();
        // d/dx (tan x) = sec^2(x) = 1 / cos^2(x)
        let sec_sq = <R as IsScalar<1, 0, 0>>::from_f64(1.0) / (cos_val * cos_val);
        Self {
            real_part: self.real_part.tan(),
            infinitesimal_part: self.infinitesimal_part.map(|dij_val| dij_val * sec_sq),
        }
    }

    fn tanh(&self) -> Self {
        let tanh_x = self.real_part.tanh();
        Self {
            real_part: tanh_x,
            infinitesimal_part: self.infinitesimal_part.map(|dij_val| {
                dij_val * (<R as IsScalar<1, 0, 0>>::from_f64(1.0) - tanh_x * tanh_x)
            }),
        }
    }

    fn acos(&self) -> Self {
        // d/dx (acos(x)) = -1 / sqrt(1 - x^2)
        let one = <R as IsScalar<1, 0, 0>>::from_f64(1.0);
        let denom = one - self.real_part * self.real_part;
        let dval = if denom > <R as IsScalar<1, 0, 0>>::from_f64(0.0) {
            -one / denom.sqrt()
        } else {
            // For out-of-domain or edge cases, we won't handle gracefully here.
            <R as IsScalar<1, 0, 0>>::from_f64(f64::NAN)
        };

        Self {
            real_part: self.real_part.acos(),
            infinitesimal_part: self.infinitesimal_part.map(|dij_val| dij_val * dval),
        }
    }

    fn asin(&self) -> Self {
        // d/dx (asin(x)) = 1 / sqrt(1 - x^2)
        let one = <R as IsScalar<1, 0, 0>>::from_f64(1.0);
        let denom = one - self.real_part * self.real_part;
        let dval = if denom > <R as IsScalar<1, 0, 0>>::from_f64(0.0) {
            one / denom.sqrt()
        } else {
            <R as IsScalar<1, 0, 0>>::from_f64(f64::NAN)
        };

        Self {
            real_part: self.real_part.asin(),
            infinitesimal_part: self.infinitesimal_part.map(|dij_val| dij_val * dval),
        }
    }

    fn atan(&self) -> Self {
        // d/dx (atan(x)) = 1 / (1 + x^2)
        let denom = <R as IsScalar<1, 0, 0>>::from_f64(1.0) + self.real_part * self.real_part;
        let dval = <R as IsScalar<1, 0, 0>>::from_f64(1.0) / denom;
        Self {
            real_part: self.real_part.atan(),
            infinitesimal_part: self.infinitesimal_part.map(|dij_val| dij_val * dval),
        }
    }

    fn fract(&self) -> Self {
        Self {
            real_part: self.real_part.fract(),
            // derivative of fract(x) is derivative( x - floor(x)) => 1 for x not an integer,
            // but it's not well-defined at integers. We'll just keep the same derivative for now,
            // as fract(x) = x - floor(x).
            infinitesimal_part: self.infinitesimal_part,
        }
    }

    fn floor(&self) -> R {
        self.real_part.floor()
    }

    fn from_f64(val: f64) -> Self {
        Self {
            real_part: <R as IsScalar<1, 0, 0>>::from_f64(val),
            infinitesimal_part: None,
        }
    }

    fn scalar_examples() -> alloc::vec::Vec<Self> {
        [1.0, 2.0, 3.0].iter().map(|&v| Self::from_f64(v)).collect()
    }

    fn extract_single(&self, _i: usize) -> Self::SingleScalar {
        *self
    }

    fn signum(&self) -> Self {
        // derivative is not defined at 0, but we'll ignore that for now.
        Self {
            real_part: self.real_part.signum(),
            infinitesimal_part: None,
        }
    }

    fn less_equal(&self, rhs: &Self) -> Self::Mask {
        self.real_part <= rhs.real_part
    }

    fn to_dual_const<const M: usize, const N: usize>(&self) -> Self::DualScalar<M, N> {
        // Copies only the real part, ignoring the derivative
        Self::DualScalar::<M, N> {
            real_part: self.real_part,
            infinitesimal_part: None,
        }
    }

    fn select(&self, mask: &Self::Mask, other: Self) -> Self {
        if *mask { *self } else { other }
    }

    fn greater_equal(&self, rhs: &Self) -> Self::Mask {
        self.real_part >= rhs.real_part
    }

    fn eps() -> Self {
        Self::from_real_scalar(<R as IsScalar<1, 0, 0>>::eps())
    }
}

// Implement additive and multiplicative assignment for convenience:
impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> AddAssign<Self> for DualScalar<R, DM, DN>
{
    fn add_assign(&mut self, rhs: Self) {
        *self = (*self).add(&rhs);
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> MulAssign<Self> for DualScalar<R, DM, DN>
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = (*self).mul(&rhs);
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> DivAssign<Self> for DualScalar<R, DM, DN>
{
    fn div_assign(&mut self, rhs: Self) {
        *self = (*self).div(&rhs);
    }
}

// Basic arithmetic ops to combine derivatives:
impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> Add<DualScalar<R, DM, DN>> for DualScalar<R, DM, DN>
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> Add<&Self> for DualScalar<R, DM, DN>
{
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        let r = self.real_part + rhs.real_part;
        Self {
            real_part: r,
            infinitesimal_part: Self::binary_dij(
                &self.infinitesimal_part,
                &rhs.infinitesimal_part,
                |l_dij| *l_dij,
                |r_dij| *r_dij,
            ),
        }
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> Sub<DualScalar<R, DM, DN>> for DualScalar<R, DM, DN>
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(&rhs)
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> Sub<&Self> for DualScalar<R, DM, DN>
{
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        Self {
            real_part: self.real_part - rhs.real_part,
            infinitesimal_part: Self::binary_dij(
                &self.infinitesimal_part,
                &rhs.infinitesimal_part,
                |l_dij| *l_dij,
                |r_dij| -r_dij,
            ),
        }
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> Mul<DualScalar<R, DM, DN>> for DualScalar<R, DM, DN>
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(&rhs)
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> Mul<&Self> for DualScalar<R, DM, DN>
{
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        let prod = self.real_part * rhs.real_part;
        Self {
            real_part: prod,
            infinitesimal_part: Self::binary_dij(
                &self.infinitesimal_part,
                &rhs.infinitesimal_part,
                |l_dij| *l_dij * rhs.real_part,
                |r_dij| *r_dij * self.real_part,
            ),
        }
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> Div<DualScalar<R, DM, DN>> for DualScalar<R, DM, DN>
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.div(&rhs)
    }
}

impl<
    R: IsRealScalar<1, RealScalar = R> + nalgebra::RealField + IsSingleScalar<0, 0>,
    const DM: usize,
    const DN: usize,
> Div<&Self> for DualScalar<R, DM, DN>
{
    type Output = Self;

    fn div(self, rhs: &Self) -> Self::Output {
        let inv = <R as IsScalar<1, 0, 0>>::from_f64(1.0) / rhs.real_part;
        let new_real = self.real_part * inv;
        Self {
            real_part: new_real,
            infinitesimal_part: Self::binary_dij(
                &self.infinitesimal_part,
                &rhs.infinitesimal_part,
                |l_dij| *l_dij * inv,
                |r_dij| *r_dij * (-self.real_part * inv * inv),
            ),
        }
    }
}
