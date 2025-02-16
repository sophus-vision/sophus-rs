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
        BatchMask,
        BatchMatF64,
        BatchScalarF64,
        BatchVecF64,
        NumberCategory,
        SVec,
        EPS_F64,
    },
    prelude::*,
};

extern crate alloc;

/// A batch dual number, storing real parts and partial derivatives for multiple **lanes** (SIMD).
///
/// In forward-mode AD (automatic differentiation), this struct represents a single scalar
/// extended with an infinitesimal part representing derivatives. It can track derivatives
/// w.r.t. one or more variables across multiple lanes in parallel.
///
/// # Fields
/// - `real_part`: A [BatchScalarF64] storing the real values across `BATCH` lanes.
/// - `infinitesimal_part`: An optional [BatchMatF64], i.e., the derivative block.
///   - If `None`, the derivative is zero for all lanes.
///   - If `Some(...)`, it shapes `[DM × DN]` for each lane.
///
/// # Const Parameters
/// - `BATCH`: The number of lanes (e.g., 2,4,8,...).
/// - `DM`, `DN`: The shape of the derivative matrix. For instance, `DM=3, DN=1` can represent
///   partial derivatives w.r.t. 3 variables for each lane.
///
/// See [crate::dual::IsDualScalar] for more details.
#[cfg(feature = "simd")]
#[derive(Clone, Debug, Copy)]
pub struct DualBatchScalar<const BATCH: usize, const DM: usize, const DN: usize>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    /// The real (non-infinitesimal) part for each lane.
    pub real_part: BatchScalarF64<BATCH>,

    /// The derivative (infinitesimal) part for each lane, shaped `[DM x DN]`.
    ///
    /// If absent, the derivative is assumed zero.
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
        // We only compare the real parts for equality checks.
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
        // Similarly, we only compare real parts for relative equality.
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
        // For performance, in-place operations might be better,
        // but here we do a simple approach:
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
        // Real part set to 1.0, no derivative.
        Self::from_f64(1.0)
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> Zero for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn zero() -> Self {
        Self::from_f64(0.0)
    }

    fn is_zero(&self) -> bool {
        // We only check if the real part is zero in all lanes.
        self.real_part == Self::from_f64(0.0).real_part
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> IsDualScalar<BATCH, DM, DN>
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn var(val: BatchScalarF64<BATCH>) -> Self {
        // A "variable" means an identity derivative w.r.t. that variable.
        // So we set derivative = 1.0 for each lane in DM×DN.
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
        // Return the derivative or zero if not present.
        self.infinitesimal_part
            .unwrap_or_else(BatchMatF64::<DM, DN, BATCH>::zeros)
    }
}

impl<const BATCH: usize> IsDualScalarFromCurve<DualBatchScalar<BATCH, 1, 1>, BATCH>
    for DualBatchScalar<BATCH, 1, 1>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn curve_derivative(&self) -> BatchScalarF64<BATCH> {
        // derivative() is DM×DN = 1×1 => single element per lane
        self.derivative()[0]
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    // Internal helper to combine the derivative blocks from two batch dual scalars
    // in binary operations.
    //
    // - `lhs_dx`/`rhs_dx`: Optional derivative blocks.
    // - `left_op`/`right_op`: Functions that produce partial derivatives for the left and right
    //   side respectively, e.g. for multiplication or addition logic.
    fn binary_dij(
        lhs_dx: &Option<BatchMatF64<DM, DN, BATCH>>,
        rhs_dx: &Option<BatchMatF64<DM, DN, BATCH>>,
        mut left_op: impl FnMut(&BatchMatF64<DM, DN, BATCH>) -> BatchMatF64<DM, DN, BATCH>,
        mut right_op: impl FnMut(&BatchMatF64<DM, DN, BATCH>) -> BatchMatF64<DM, DN, BATCH>,
    ) -> Option<BatchMatF64<DM, DN, BATCH>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs)) => Some(right_op(rhs)),
            (Some(lhs), None) => Some(left_op(lhs)),
            (Some(lhs), Some(rhs)) => Some(left_op(lhs) + right_op(rhs)),
        }
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> Neg for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        // - (real + derivative)
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
    // Specialized type definitions for batch dual scalars:
    type Scalar = Self;
    type Vector<const ROWS: usize> = DualBatchVector<ROWS, BATCH, DM, DN>;
    type Matrix<const ROWS: usize, const COLS: usize> = DualBatchMatrix<ROWS, COLS, BATCH, DM, DN>;

    // The "real" portion of the scalar is `BatchScalarF64<BATCH>`.
    type RealScalar = BatchScalarF64<BATCH>;
    // Real (non-dual) vectors / matrices in the batch domain:
    type RealVector<const ROWS: usize> = BatchVecF64<ROWS, BATCH>;
    type RealMatrix<const ROWS: usize, const COLS: usize> = BatchMatF64<ROWS, COLS, BATCH>;

    // Single-scalar version is the non-batch `DualScalar<DM, DN>`.
    type SingleScalar = DualScalar<DM, DN>;

    // Another batch dual version:
    type DualScalar<const M: usize, const N: usize> = DualBatchScalar<BATCH, M, N>;
    // Another batch dual vector / matrix:
    type DualVector<const ROWS: usize, const M: usize, const N: usize> =
        DualBatchVector<ROWS, BATCH, M, N>;
    type DualMatrix<const ROWS: usize, const COLS: usize, const M: usize, const N: usize> =
        DualBatchMatrix<ROWS, COLS, BATCH, M, N>;

    type Mask = BatchMask<BATCH>;

    fn from_real_scalar(val: BatchScalarF64<BATCH>) -> Self {
        // Real part set, derivative = None => zero derivative.
        Self {
            real_part: val,
            infinitesimal_part: None,
        }
    }

    fn from_real_array(_arr: [f64; BATCH]) -> Self {
        // Optional: allow constructing from an array of BATCH f64.
        todo!("Construct from a per-lane array is not implemented yet.")
    }

    fn to_real_array(&self) -> [f64; BATCH] {
        // Optional: convert each lane of real_part to an array.
        todo!("Convert dual batch scalar to real array is not implemented yet.")
    }

    // Basic math ops: cos, sin, etc. We'll only define the real part's derivative if
    // `infinitesimal_part` exists.

    fn cos(&self) -> Self {
        // d/dx cos(x) = -sin(x)
        Self {
            real_part: self.real_part.cos(),
            infinitesimal_part: self
                .infinitesimal_part
                .map(|dij_val| -dij_val * self.real_part.sin()),
        }
    }

    fn sin(&self) -> Self {
        // d/dx sin(x) = cos(x)
        Self {
            real_part: self.real_part.sin(),
            infinitesimal_part: self
                .infinitesimal_part
                .map(|dij_val| dij_val * self.real_part.cos()),
        }
    }

    fn abs(&self) -> Self {
        // d/dx abs(x) is signum(x), ignoring corner at x=0
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
        // d/dy => x / (x^2 + y^2), d/dx => -y / (x^2 + y^2)
        let rhs = rhs.borrow();
        let inv_sq_nrm = BatchScalarF64::<BATCH>::from_f64(1.0)
            / (self.real_part * self.real_part + rhs.real_part * rhs.real_part);

        Self {
            real_part: self.real_part.atan2(rhs.real_part),
            infinitesimal_part: Self::binary_dij(
                &self.infinitesimal_part,
                &rhs.infinitesimal_part,
                |lhs_dij| *lhs_dij * (inv_sq_nrm * rhs.real_part),
                |rhs_dij| *rhs_dij * (-inv_sq_nrm * self.real_part),
            ),
        }
    }

    fn exp(&self) -> Self {
        let exp_val = self.real_part.exp();
        // d/dx e^x = e^x
        Self {
            real_part: exp_val,
            infinitesimal_part: self
                .infinitesimal_part
                .map(|dij_val| dij_val * self.real_part.exp()),
        }
    }

    fn ln(&self) -> Self {
        let ln_val = self.real_part.ln();
        // d/dx ln(x) = 1 / x
        Self {
            real_part: ln_val,
            infinitesimal_part: self
                .infinitesimal_part
                .map(|dij_val| dij_val / self.real_part),
        }
    }

    fn real_part(&self) -> BatchScalarF64<BATCH> {
        self.real_part
    }

    fn sqrt(&self) -> Self {
        // d/dx sqrt(x) = 1 / (2 sqrt(x))
        let sqrt_val = self.real_part.sqrt();
        let denom = BatchScalarF64::<BATCH>::from_f64(2.0) * sqrt_val;
        Self {
            real_part: sqrt_val,
            infinitesimal_part: self.infinitesimal_part.map(|dij| dij / denom),
        }
    }

    fn to_vec(&self) -> DualBatchVector<1, BATCH, DM, DN> {
        DualBatchVector::<1, BATCH, DM, DN> {
            inner: SVec::<Self, 1>::from_element(*self),
        }
    }

    fn tan(&self) -> Self {
        // d/dx tan(x) = sec^2(x) = 1 / cos^2(x)
        let cos_val = self.real_part.cos();
        let sec_sq = BatchScalarF64::<BATCH>::from_f64(1.0) / (cos_val * cos_val);
        Self {
            real_part: self.real_part.tan(),
            infinitesimal_part: self.infinitesimal_part.map(|dij_val| dij_val * sec_sq),
        }
    }

    fn acos(&self) -> Self {
        // d/dx acos(x) = -1 / sqrt(1 - x^2)
        let val = self.real_part.acos();
        let derivative = match self.infinitesimal_part {
            Some(dij_val) => {
                let one = BatchScalarF64::<BATCH>::from_f64(1.0);
                let denom = (one - self.real_part * self.real_part).sqrt();
                let dval = -one / denom;
                Some(dij_val * dval)
            }
            None => None,
        };
        Self {
            real_part: val,
            infinitesimal_part: derivative,
        }
    }

    fn asin(&self) -> Self {
        // d/dx asin(x) = 1 / sqrt(1 - x^2)
        let val = self.real_part.asin();
        let derivative = match self.infinitesimal_part {
            Some(dij_val) => {
                let one = BatchScalarF64::<BATCH>::from_f64(1.0);
                let denom = (one - self.real_part * self.real_part).sqrt();
                let dval = one / denom;
                Some(dij_val * dval)
            }
            None => None,
        };
        Self {
            real_part: val,
            infinitesimal_part: derivative,
        }
    }

    fn atan(&self) -> Self {
        // d/dx atan(x) = 1 / (1 + x^2)
        let val = self.real_part.atan();
        let derivative = match self.infinitesimal_part {
            Some(dij_val) => {
                let one = BatchScalarF64::<BATCH>::from_f64(1.0);
                let denom = one + self.real_part * self.real_part;
                Some(dij_val * (one / denom))
            }
            None => None,
        };
        Self {
            real_part: val,
            infinitesimal_part: derivative,
        }
    }

    fn fract(&self) -> Self {
        // derivative of fract(x) is derivative( x - floor(x) ) => 1 except at discontinuities
        // We'll just copy the derivative as-is
        Self {
            real_part: self.real_part.fract(),
            infinitesimal_part: self.infinitesimal_part,
        }
    }

    fn floor(&self) -> BatchScalarF64<BATCH> {
        self.real_part.floor()
    }

    fn from_f64(val: f64) -> Self {
        // Broadcast the same f64 to all lanes, derivative = None
        Self {
            real_part: BatchScalarF64::<BATCH>::from_f64(val),
            infinitesimal_part: None,
        }
    }

    fn scalar_examples() -> alloc::vec::Vec<Self> {
        // Some sample values with each lane set the same. Real usage might randomize lanes.
        [1.0, 2.0, 3.0].iter().map(|&v| Self::from_f64(v)).collect()
    }

    fn extract_single(&self, _i: usize) -> Self::SingleScalar {
        // Possibly could convert one lane to a single-lane dual scalar.
        // For now, it's unimplemented.
        todo!("Extracting a single lane from a DualBatchScalar is not implemented.")
    }

    fn signum(&self) -> Self {
        // dx signum(x) = 0 except at x=0
        Self {
            real_part: self.real_part.signum(),
            infinitesimal_part: None,
        }
    }

    fn less_equal(&self, rhs: &Self) -> Self::Mask {
        self.real_part.less_equal(&rhs.real_part)
    }

    fn to_dual_const<const M: usize, const N: usize>(&self) -> Self::DualScalar<M, N> {
        // Copy real parts, no derivative
        DualBatchScalar {
            real_part: self.real_part,
            infinitesimal_part: None,
        }
    }

    fn select(&self, mask: &Self::Mask, other: Self) -> Self {
        // Lane-wise select
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
        // returns EPS_F64 broadcast to all lanes
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
        *self = (*self).div(&rhs);
    }
}

impl<const BATCH: usize, const DM: usize, const DN: usize> Add<DualBatchScalar<BATCH, DM, DN>>
    for DualBatchScalar<BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = Self;
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
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        let real_sum = self.real_part + rhs.real_part;
        // d/dx (x + y) = dx + dy
        Self {
            real_part: real_sum,
            infinitesimal_part: Self::binary_dij(
                &self.infinitesimal_part,
                &rhs.infinitesimal_part,
                |l_dij| *l_dij,
                |r_dij| *r_dij,
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
    type Output = Self;
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
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        let real_diff = self.real_part - rhs.real_part;
        // d/dx (x - y) = dx - dy
        Self {
            real_part: real_diff,
            infinitesimal_part: Self::binary_dij(
                &self.infinitesimal_part,
                &rhs.infinitesimal_part,
                |l_dij| *l_dij,
                |r_dij| -*r_dij,
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
    type Output = Self;
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
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        // real: x * y
        let real_prod = self.real_part * rhs.real_part;
        // derivative: dx*y + x*dy
        Self {
            real_part: real_prod,
            infinitesimal_part: Self::binary_dij(
                &self.infinitesimal_part,
                &rhs.infinitesimal_part,
                |lhs_dij| *lhs_dij * rhs.real_part,
                |rhs_dij| *rhs_dij * self.real_part,
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
    type Output = Self;

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
    type Output = Self;

    fn div(self, rhs: &Self) -> Self::Output {
        // (x / y) => derivative is dx / y - x (dy) / y^2
        let inv = BatchScalarF64::<BATCH>::from_f64(1.0) / rhs.real_part;
        let real_div = self.real_part * inv;
        Self {
            real_part: real_div,
            infinitesimal_part: Self::binary_dij(
                &self.infinitesimal_part,
                &rhs.infinitesimal_part,
                |l_dij| *l_dij * inv,
                |r_dij| *r_dij * (-self.real_part * inv * inv),
            ),
        }
    }
}
