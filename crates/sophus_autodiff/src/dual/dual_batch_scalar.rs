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
    simd::{
        LaneCount,
        SupportedLaneCount,
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

use super::dual_scalar::DualScalar;
pub use crate::dual::{
    dual_batch_matrix::DualBatchMatrix,
    dual_batch_vector::DualBatchVector,
};
use crate::{
    linalg::{
        batch_mask::BatchMask,
        scalar::NumberCategory,
        BatchMatF64,
        BatchScalarF64,
        BatchVecF64,
        SVec,
        EPS_F64,
    },
    prelude::*,
};

extern crate alloc;

/// Dual number - a real number and an infinitesimal number (batch version)
#[derive(Clone, Debug, Copy)]
pub struct DualBatchScalar<const BATCH: usize, const DM: usize, const DN: usize>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    /// real part
    pub real_part: BatchScalarF64<BATCH>,

    /// infinitesimal part - represents derivative
    pub infinitesimal_part: Option<BatchMatF64<DM, DN, BATCH>>,
}

impl<const BATCH: usize, const DM: usize, const DN: usize> AbsDiffEq
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        EPS_F64
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.real_part.abs_diff_eq(&other.real_part, epsilon)
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> RelativeEq
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
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

impl<const BATCH: usize, const DM: usize, const DN: usize> IsCoreScalar
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn number_category() -> NumberCategory {
        NumberCategory::Real
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> SubAssign<DualBatchScalar<BATCH, DM, DN>>
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = (*self).sub(&rhs);
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> AsRef<DualBatchScalar<BATCH, DM, DN>>
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn as_ref(&self) -> &DualBatchScalar<BATCH, DM, DN> {
        self
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> One for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn one() -> Self {
        <DualBatchScalar<BATCH, DM, DN>>::from_f64(1.0)
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> Zero for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn zero() -> Self {
        <DualBatchScalar<BATCH, DM, DN>>::from_f64(0.0)
    }

    fn is_zero(&self) -> bool {
        self.real_part == <DualBatchScalar<BATCH, DM, DN>>::from_f64(0.0).real_part()
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> IsDualScalar<BATCH, DM, DN>
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn var(val: BatchScalarF64<BATCH>) -> Self {
        Self {
            real_part: val,
            infinitesimal_part: Some(BatchMatF64::<DM, DN, BATCH>::from_f64(1.0)),
        }
    }

    fn vector_var<const ROWS: usize>(val: Self::RealVector<ROWS>) -> Self::Vector<ROWS> {
        DualBatchVector::<ROWS, BATCH, DM, DN>::var(val)
    }

    fn matrix_var<const ROWS: usize, const COLS: usize>(
        val: Self::RealMatrix<ROWS, COLS>,
    ) -> Self::Matrix<ROWS, COLS> {
        DualBatchMatrix::<ROWS, COLS, BATCH, DM, DN>::var(val)
    }

    fn derivative(&self) -> BatchMatF64<DM, DN, BATCH> {
        self.infinitesimal_part
            .unwrap_or(BatchMatF64::<DM, DN, BATCH>::zeros())
    }
}

impl<const BATCH: usize> IsDualScalarFromCurve<DualBatchScalar<BATCH, 1, 1>, BATCH>
    for DualBatchScalar<BATCH, 1, 1>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn curve_derivative(&self) -> BatchScalarF64<BATCH> {
        self.derivative()[0]
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn binary_dij(
        lhs_dx: &Option<BatchMatF64<DM, DN, BATCH>>,
        rhs_dx: &Option<BatchMatF64<DM, DN, BATCH>>,
        mut left_op: impl FnMut(&BatchMatF64<DM, DN, BATCH>) -> BatchMatF64<DM, DN, BATCH>,
        mut right_op: impl FnMut(&BatchMatF64<DM, DN, BATCH>) -> BatchMatF64<DM, DN, BATCH>,
    ) -> Option<BatchMatF64<DM, DN, BATCH>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => Some(right_op(rhs_dij)),
            (Some(lhs_dij), None) => Some(left_op(lhs_dij)),
            (Some(lhs_dij), Some(rhs_dij)) => Some(left_op(lhs_dij) + right_op(rhs_dij)),
        }
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> Neg for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH, DM, DN>;

    fn neg(self) -> Self {
        Self {
            real_part: -self.real_part,
            infinitesimal_part: self.infinitesimal_part.map(|dij_val| -dij_val),
        }
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> PartialEq
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn eq(&self, other: &Self) -> bool {
        self.real_part == other.real_part
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> IsScalar<BATCH, DM, DN>
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Scalar = DualBatchScalar<BATCH, DM, DN>;
    type Vector<const ROWS: usize> = DualBatchVector<ROWS, BATCH, DM, DN>;
    type Matrix<const ROWS: usize, const COLS: usize> = DualBatchMatrix<ROWS, COLS, BATCH, DM, DN>;

    type RealScalar = BatchScalarF64<BATCH>;
    type RealVector<const ROWS: usize> = BatchVecF64<ROWS, BATCH>;
    type RealMatrix<const ROWS: usize, const COLS: usize> = BatchMatF64<ROWS, COLS, BATCH>;

    type SingleScalar = DualScalar<DM, DN>;

    type DualScalar<const M: usize, const N: usize> = DualBatchScalar<BATCH, M, N>;
    type DualVector<const ROWS: usize, const M: usize, const N: usize> =
        DualBatchVector<ROWS, BATCH, M, N>;
    type DualMatrix<const ROWS: usize, const COLS: usize, const M: usize, const N: usize> =
        DualBatchMatrix<ROWS, COLS, BATCH, M, N>;

    type Mask = BatchMask<BATCH>;

    fn from_real_scalar(val: BatchScalarF64<BATCH>) -> Self {
        Self {
            real_part: val,
            infinitesimal_part: None,
        }
    }

    fn from_real_array(_arr: [f64; BATCH]) -> Self {
        todo!()
    }

    fn to_real_array(&self) -> [f64; BATCH] {
        todo!()
    }

    fn cos(&self) -> Self {
        Self {
            real_part: self.real_part.cos(),
            infinitesimal_part: self
                .infinitesimal_part
                .map(|dij_val| -dij_val * self.real_part.sin()),
        }
    }

    fn sin(&self) -> Self {
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
        let inv_sq_nrm = BatchScalarF64::<BATCH>::from_f64(1.0)
            / (self.real_part * self.real_part + rhs.real_part * rhs.real_part);
        Self {
            real_part: self.real_part.atan2(rhs.real_part),
            infinitesimal_part: Self::binary_dij(
                &self.infinitesimal_part,
                &rhs.infinitesimal_part,
                |l_dij| (*l_dij) * (inv_sq_nrm * rhs.real_part),
                |r_dij| (*r_dij) * (-inv_sq_nrm * self.real_part),
            ),
        }
    }

    fn exp(&self) -> DualBatchScalar<BATCH, DM, DN> {
        Self {
            real_part: self.real_part.cos(),
            infinitesimal_part: self
                .infinitesimal_part
                .map(|dij_val| dij_val * self.real_part.exp()),
        }
    }

    fn real_part(&self) -> BatchScalarF64<BATCH> {
        self.real_part
    }

    fn sqrt(&self) -> Self {
        let sqrt = self.real_part.sqrt();
        Self {
            real_part: sqrt,
            infinitesimal_part: self
                .infinitesimal_part
                .map(|dij| dij / (BatchScalarF64::<BATCH>::from_f64(2.0) * sqrt)),
        }
    }

    fn to_vec(&self) -> DualBatchVector<1, BATCH, DM, DN> {
        DualBatchVector::<1, BATCH, DM, DN> {
            inner: SVec::<DualBatchScalar<BATCH, DM, DN>, 1>::from_element(*self),
        }
    }

    fn tan(&self) -> Self {
        Self {
            real_part: self.real_part.tan(),
            infinitesimal_part: match self.infinitesimal_part {
                Some(dij_val) => {
                    let c = self.real_part.cos();
                    let sec_squared = BatchScalarF64::<BATCH>::from_f64(1.0) / (c * c);
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
                    let dval = BatchScalarF64::<BATCH>::from_f64(-1.0)
                        / (BatchScalarF64::<BATCH>::from_f64(1.0)
                            - self.real_part * self.real_part)
                            .sqrt();
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
                    let dval = BatchScalarF64::<BATCH>::from_f64(1.0)
                        / (BatchScalarF64::<BATCH>::from_f64(1.0)
                            - self.real_part * self.real_part)
                            .sqrt();
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
                    let dval = BatchScalarF64::<BATCH>::from_f64(1.0)
                        / (BatchScalarF64::<BATCH>::from_f64(1.0)
                            + self.real_part * self.real_part);

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

    fn floor(&self) -> BatchScalarF64<BATCH> {
        self.real_part.floor()
    }

    fn from_f64(val: f64) -> Self {
        Self {
            real_part: BatchScalarF64::<BATCH>::from_f64(val),
            infinitesimal_part: None,
        }
    }

    fn scalar_examples() -> alloc::vec::Vec<Self> {
        [1.0, 2.0, 3.0].iter().map(|&v| Self::from_f64(v)).collect()
    }

    fn extract_single(&self, _i: usize) -> Self::SingleScalar {
        todo!()
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
        Self {
            real_part: self.real_part.select(mask, other.real_part),
            infinitesimal_part: match (self.infinitesimal_part, other.infinitesimal_part) {
                (Some(lhs), Some(rhs)) => Some(lhs.select(mask, rhs)),
                (Some(lhs), None) => Some(lhs.select(mask, BatchMatF64::<DM, DN, BATCH>::zeros())),
                (None, Some(rhs)) => Some(BatchMatF64::<DM, DN, BATCH>::zeros().select(mask, rhs)),
                (None, None) => None,
            },
        }
    }

    fn greater_equal(&self, rhs: &Self) -> Self::Mask {
        self.real_part.greater_equal(&rhs.real_part)
    }

    fn eps() -> Self {
        Self::from_real_scalar(BatchScalarF64::<BATCH>::from_f64(EPS_F64))
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> AddAssign<DualBatchScalar<BATCH, DM, DN>>
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn add_assign(&mut self, rhs: Self) {
        // this is a bit inefficient, better to do it in place
        *self = (*self).add(&rhs);
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> MulAssign<DualBatchScalar<BATCH, DM, DN>>
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn mul_assign(&mut self, rhs: Self) {
        // this is a bit inefficient, better to do it in place
        *self = (*self).mul(&rhs);
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> DivAssign<DualBatchScalar<BATCH, DM, DN>>
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn div_assign(&mut self, rhs: Self) {
        // this is a bit inefficient, better to do it in place
        *self = (*self).div(&rhs);
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> Add<DualBatchScalar<BATCH, DM, DN>>
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH, DM, DN>;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> Add<&DualBatchScalar<BATCH, DM, DN>>
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH, DM, DN>;
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

impl<const BATCH: usize, const DM: usize, const DN: usize> Mul<DualBatchScalar<BATCH, DM, DN>>
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH, DM, DN>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(&rhs)
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> Mul<&DualBatchScalar<BATCH, DM, DN>>
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH, DM, DN>;
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

impl<const BATCH: usize, const DM: usize, const DN: usize> Div<DualBatchScalar<BATCH, DM, DN>>
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH, DM, DN>;
    fn div(self, rhs: Self) -> Self::Output {
        self.div(&rhs)
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> Div<&DualBatchScalar<BATCH, DM, DN>>
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH, DM, DN>;
    fn div(self, rhs: &Self) -> Self::Output {
        let rhs_inv = BatchScalarF64::<BATCH>::from_f64(1.0) / rhs.real_part;
        Self {
            real_part: self.real_part * rhs_inv,
            infinitesimal_part: Self::binary_dij(
                &self.infinitesimal_part,
                &rhs.infinitesimal_part,
                |l_dij| l_dij * rhs_inv,
                |r_dij| r_dij * (-self.real_part * rhs_inv * rhs_inv),
            ),
        }
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> Sub<DualBatchScalar<BATCH, DM, DN>>
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH, DM, DN>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(&rhs)
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> Sub<&DualBatchScalar<BATCH, DM, DN>>
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH, DM, DN>;
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
