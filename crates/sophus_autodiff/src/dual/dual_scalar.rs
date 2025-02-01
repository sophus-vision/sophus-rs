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
use num_traits::{
    One,
    Zero,
};

use super::{
    dual_matrix::DualMatrix,
    dual_vector::DualVector,
};
use crate::{
    linalg::{
        scalar::NumberCategory,
        MatF64,
        SVec,
        VecF64,
        EPS_F64,
    },
    prelude::*,
};

extern crate alloc;

/// Dual number - a real number and an infinitesimal number
#[derive(Clone, Debug, Copy)]
pub struct DualScalar<const DM: usize, const DN: usize> {
    /// real part
    pub real_part: f64,

    /// infinitesimal part - represents derivative
    pub infinitesimal_part: Option<MatF64<DM, DN>>,
}

impl<const DM: usize, const DN: usize> AbsDiffEq for DualScalar<DM, DN> {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        EPS_F64
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.real_part.abs_diff_eq(&other.real_part, epsilon)
    }
}

impl<const DM: usize, const DN: usize> RelativeEq for DualScalar<DM, DN> {
    fn default_max_relative() -> Self::Epsilon {
        EPS_F64
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

impl<const DM: usize, const DN: usize> IsCoreScalar for DualScalar<DM, DN> {
    fn number_category() -> NumberCategory {
        NumberCategory::Real
    }
}

impl<const DM: usize, const DN: usize> SubAssign<DualScalar<DM, DN>> for DualScalar<DM, DN> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = (*self).sub(&rhs);
    }
}

impl<const DM: usize, const DN: usize> IsSingleScalar<DM, DN> for DualScalar<DM, DN> {
    type SingleVector<const ROWS: usize> = DualVector<ROWS, DM, DN>;
    type SingleMatrix<const ROWS: usize, const COLS: usize> = DualMatrix<ROWS, COLS, DM, DN>;

    fn single_real_scalar(&self) -> f64 {
        self.real_part
    }

    fn single_scalar(&self) -> Self {
        *self
    }

    fn i64_floor(&self) -> i64 {
        self.real_part.floor() as i64
    }
}

impl<const DM: usize, const DN: usize> AsRef<DualScalar<DM, DN>> for DualScalar<DM, DN> {
    fn as_ref(&self) -> &DualScalar<DM, DN> {
        self
    }
}

impl<const DM: usize, const DN: usize> One for DualScalar<DM, DN> {
    fn one() -> Self {
        <DualScalar<DM, DN>>::from_f64(1.0)
    }
}

impl<const DM: usize, const DN: usize> Zero for DualScalar<DM, DN> {
    fn zero() -> Self {
        <DualScalar<DM, DN>>::from_f64(0.0)
    }

    fn is_zero(&self) -> bool {
        self.real_part == <DualScalar<DM, DN>>::from_f64(0.0).real_part()
    }
}

impl<const DM: usize, const DN: usize> IsDualScalar<1, DM, DN> for DualScalar<DM, DN> {
    fn var(val: f64) -> Self {
        Self {
            real_part: val,
            infinitesimal_part: Some(MatF64::<DM, DN>::from_f64(1.0)),
        }
    }

    fn vector_var<const ROWS: usize>(val: Self::RealVector<ROWS>) -> Self::Vector<ROWS> {
        DualVector::<ROWS, DM, DN>::var(val)
    }

    fn matrix_var<const ROWS: usize, const COLS: usize>(
        val: Self::RealMatrix<ROWS, COLS>,
    ) -> Self::Matrix<ROWS, COLS> {
        DualMatrix::<ROWS, COLS, DM, DN>::var(val)
    }

    fn derivative(&self) -> MatF64<DM, DN> {
        self.infinitesimal_part.unwrap_or(MatF64::<DM, DN>::zeros())
    }
}

impl<const DM: usize, const DN: usize> DualScalar<DM, DN> {
    fn binary_dij(
        lhs_dx: &Option<MatF64<DM, DN>>,
        rhs_dx: &Option<MatF64<DM, DN>>,
        mut left_op: impl FnMut(&MatF64<DM, DN>) -> MatF64<DM, DN>,
        mut right_op: impl FnMut(&MatF64<DM, DN>) -> MatF64<DM, DN>,
    ) -> Option<MatF64<DM, DN>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => Some(right_op(rhs_dij)),
            (Some(lhs_dij), None) => Some(left_op(lhs_dij)),
            (Some(lhs_dij), Some(rhs_dij)) => Some(left_op(lhs_dij) + right_op(rhs_dij)),
        }
    }
}

impl<const DM: usize, const DN: usize> Neg for DualScalar<DM, DN> {
    type Output = DualScalar<DM, DN>;

    fn neg(self) -> Self {
        Self {
            real_part: -self.real_part,
            infinitesimal_part: self.infinitesimal_part.map(|dij_val| -dij_val),
        }
    }
}

impl<const DM: usize, const DN: usize> PartialEq for DualScalar<DM, DN> {
    fn eq(&self, other: &Self) -> bool {
        self.real_part == other.real_part
    }
}

impl<const DM: usize, const DN: usize> PartialOrd for DualScalar<DM, DN> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.real_part.partial_cmp(&other.real_part)
    }
}

impl<const DM: usize, const DN: usize> From<f64> for DualScalar<DM, DN> {
    fn from(value: f64) -> Self {
        Self::from_real_scalar(value)
    }
}

impl<const DM: usize, const DN: usize> IsScalar<1, DM, DN> for DualScalar<DM, DN> {
    type Scalar = DualScalar<DM, DN>;
    type Vector<const ROWS: usize> = DualVector<ROWS, DM, DN>;
    type Matrix<const ROWS: usize, const COLS: usize> = DualMatrix<ROWS, COLS, DM, DN>;

    type RealScalar = f64;
    type RealMatrix<const ROWS: usize, const COLS: usize> = MatF64<ROWS, COLS>;
    type RealVector<const ROWS: usize> = VecF64<ROWS>;

    type SingleScalar = DualScalar<DM, DN>;

    type DualScalar<const M: usize, const N: usize> = DualScalar<M, N>;
    type DualVector<const ROWS: usize, const M: usize, const N: usize> = DualVector<ROWS, M, N>;
    type DualMatrix<const ROWS: usize, const COLS: usize, const M: usize, const N: usize> =
        DualMatrix<ROWS, COLS, M, N>;

    type Mask = bool;

    fn from_real_scalar(val: f64) -> Self {
        Self {
            real_part: val,
            infinitesimal_part: None,
        }
    }

    fn from_real_array(arr: [f64; 1]) -> Self {
        Self::from_f64(arr[0])
    }

    fn to_real_array(&self) -> [f64; 1] {
        [self.real_part]
    }

    fn cos(&self) -> DualScalar<DM, DN> {
        Self {
            real_part: self.real_part.cos(),
            infinitesimal_part: self
                .infinitesimal_part
                .map(|dij_val| -dij_val * self.real_part.sin()),
        }
    }

    fn exp(&self) -> DualScalar<DM, DN> {
        Self {
            real_part: self.real_part.cos(),
            infinitesimal_part: self
                .infinitesimal_part
                .map(|dij_val| dij_val * self.real_part.exp()),
        }
    }

    fn sin(&self) -> DualScalar<DM, DN> {
        Self {
            real_part: self.real_part.sin(),
            infinitesimal_part: self
                .infinitesimal_part
                .map(|dij_val| dij_val * self.real_part.cos()),
        }
    }

    fn abs(&self) -> Self {
        Self {
            real_part: self.real_part.abs(),
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
        let inv_sq_nrm: f64 =
            1.0 / (self.real_part * self.real_part + rhs.real_part * rhs.real_part);
        Self {
            real_part: self.real_part.atan2(rhs.real_part),
            infinitesimal_part: Self::binary_dij(
                &self.infinitesimal_part,
                &rhs.infinitesimal_part,
                |l_dij| inv_sq_nrm * ((*l_dij) * rhs.real_part),
                |r_dij| -inv_sq_nrm * (self.real_part * (*r_dij)),
            ),
        }
    }

    fn real_part(&self) -> f64 {
        self.real_part
    }

    fn sqrt(&self) -> Self {
        let sqrt = self.real_part.sqrt();
        Self {
            real_part: sqrt,
            infinitesimal_part: self.infinitesimal_part.map(|dij| dij * 1.0 / (2.0 * sqrt)),
        }
    }

    fn to_vec(&self) -> DualVector<1, DM, DN> {
        DualVector::<1, DM, DN> {
            inner: SVec::<DualScalar<DM, DN>, 1>::from_element(*self),
        }
    }

    fn tan(&self) -> Self {
        Self {
            real_part: self.real_part.tan(),
            infinitesimal_part: match self.infinitesimal_part {
                Some(dij_val) => {
                    let c = self.real_part.cos();
                    let sec_squared = 1.0 / (c * c);
                    Some(dij_val * sec_squared)
                }
                None => None,
            },
        }
    }

    fn acos(&self) -> Self {
        Self {
            real_part: self.real_part.acos(),
            infinitesimal_part: match self.infinitesimal_part {
                Some(dij_val) => {
                    let dval = -1.0 / (1.0 - self.real_part * self.real_part).sqrt();
                    Some(dij_val * dval)
                }
                None => None,
            },
        }
    }

    fn asin(&self) -> Self {
        Self {
            real_part: self.real_part.asin(),
            infinitesimal_part: match self.infinitesimal_part {
                Some(dij_val) => {
                    let dval = 1.0 / (1.0 - self.real_part * self.real_part).sqrt();
                    Some(dij_val * dval)
                }
                None => None,
            },
        }
    }

    fn atan(&self) -> Self {
        Self {
            real_part: self.real_part.atan(),
            infinitesimal_part: match self.infinitesimal_part {
                Some(dij_val) => {
                    let dval = 1.0 / (1.0 + self.real_part * self.real_part);

                    Some(dij_val * dval)
                }
                None => None,
            },
        }
    }

    fn fract(&self) -> Self {
        Self {
            real_part: self.real_part.fract(),
            infinitesimal_part: self.infinitesimal_part,
        }
    }

    fn floor(&self) -> f64 {
        self.real_part.floor()
    }

    fn from_f64(val: f64) -> Self {
        Self {
            real_part: val,
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
        Self {
            real_part: self.real_part.signum(),
            infinitesimal_part: None,
        }
    }

    fn less_equal(&self, rhs: &Self) -> Self::Mask {
        self.real_part.less_equal(&rhs.real_part)
    }

    fn to_dual_const<const M: usize, const N: usize>(&self) -> Self::DualScalar<M, N> {
        Self::DualScalar::<M, N> {
            real_part: self.real_part,
            infinitesimal_part: None,
        }
    }

    fn select(&self, mask: &Self::Mask, other: Self) -> Self {
        if *mask {
            *self
        } else {
            other
        }
    }

    fn greater_equal(&self, rhs: &Self) -> Self::Mask {
        self.real_part.greater_equal(&rhs.real_part)
    }

    fn eps() -> Self {
        Self::from_real_scalar(EPS_F64)
    }
}

impl<const DM: usize, const DN: usize> AddAssign<DualScalar<DM, DN>> for DualScalar<DM, DN> {
    fn add_assign(&mut self, rhs: Self) {
        // this is a bit inefficient, better to do it in place
        *self = (*self).add(&rhs);
    }
}

impl<const DM: usize, const DN: usize> MulAssign<DualScalar<DM, DN>> for DualScalar<DM, DN> {
    fn mul_assign(&mut self, rhs: Self) {
        // this is a bit inefficient, better to do it in place
        *self = (*self).mul(&rhs);
    }
}

impl<const DM: usize, const DN: usize> DivAssign<DualScalar<DM, DN>> for DualScalar<DM, DN> {
    fn div_assign(&mut self, rhs: Self) {
        // this is a bit inefficient, better to do it in place
        *self = (*self).div(&rhs);
    }
}

impl<const DM: usize, const DN: usize> Add<DualScalar<DM, DN>> for DualScalar<DM, DN> {
    type Output = DualScalar<DM, DN>;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl<const DM: usize, const DN: usize> Add<&DualScalar<DM, DN>> for DualScalar<DM, DN> {
    type Output = DualScalar<DM, DN>;
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

impl<const DM: usize, const DN: usize> Mul<DualScalar<DM, DN>> for DualScalar<DM, DN> {
    type Output = DualScalar<DM, DN>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(&rhs)
    }
}

impl<const DM: usize, const DN: usize> Mul<&DualScalar<DM, DN>> for DualScalar<DM, DN> {
    type Output = DualScalar<DM, DN>;
    fn mul(self, rhs: &Self) -> Self::Output {
        let r = self.real_part * rhs.real_part;

        Self {
            real_part: r,
            infinitesimal_part: Self::binary_dij(
                &self.infinitesimal_part,
                &rhs.infinitesimal_part,
                |l_dij| (*l_dij) * rhs.real_part,
                |r_dij| (*r_dij) * self.real_part,
            ),
        }
    }
}

impl<const DM: usize, const DN: usize> Div<DualScalar<DM, DN>> for DualScalar<DM, DN> {
    type Output = DualScalar<DM, DN>;
    fn div(self, rhs: Self) -> Self::Output {
        self.div(&rhs)
    }
}

impl<const DM: usize, const DN: usize> Div<&DualScalar<DM, DN>> for DualScalar<DM, DN> {
    type Output = DualScalar<DM, DN>;
    fn div(self, rhs: &Self) -> Self::Output {
        let rhs_inv = 1.0 / rhs.real_part;
        Self {
            real_part: self.real_part * rhs_inv,
            infinitesimal_part: Self::binary_dij(
                &self.infinitesimal_part,
                &rhs.infinitesimal_part,
                |l_dij| l_dij * rhs_inv,
                |r_dij| -self.real_part * r_dij * rhs_inv * rhs_inv,
            ),
        }
    }
}

impl<const DM: usize, const DN: usize> Sub<DualScalar<DM, DN>> for DualScalar<DM, DN> {
    type Output = DualScalar<DM, DN>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(&rhs)
    }
}

impl<const DM: usize, const DN: usize> Sub<&DualScalar<DM, DN>> for DualScalar<DM, DN> {
    type Output = DualScalar<DM, DN>;
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
