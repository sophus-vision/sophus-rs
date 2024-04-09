use super::dual_matrix::DualBatchMatrix;
use super::dual_matrix::DualMatrix;
use super::dual_vector::DualBatchVector;
use super::dual_vector::DualVector;
use crate::linalg::scalar::IsCoreScalar;
use crate::linalg::scalar::IsScalar;
use crate::linalg::scalar::IsSingleScalar;
use crate::linalg::scalar::NumberCategory;
use crate::linalg::BatchMatF64;
use crate::linalg::BatchScalarF64;
use crate::linalg::BatchVecF64;
use crate::linalg::MatF64;
use crate::linalg::VecF64;
use crate::tensor::mut_tensor::InnerScalarToVec;
use crate::tensor::mut_tensor::MutTensorDD;
use crate::tensor::tensor_view::IsTensorLike;
use approx::assert_abs_diff_eq;
use approx::AbsDiffEq;
use approx::RelativeEq;
use num_traits::One;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;
use std::ops::SubAssign;
use std::simd::LaneCount;
use std::simd::Mask;
use std::simd::SupportedLaneCount;

/// Trait for dual numbers
pub trait IsDual {}

/// Dual number - a real number and an infinitesimal number
#[derive(Clone)]
pub struct DualScalar {
    /// value - real number
    pub val: f64,

    /// derivative - infinitesimal number
    pub dij_val: Option<MutTensorDD<f64>>,
}

impl IsDual for DualScalar {}

/// Dual number - a real number and an infinitesimal number (batch version)
#[derive(Clone)]
pub struct DualBatchScalar<const BATCH: usize>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    /// value - real number
    pub val: BatchScalarF64<BATCH>,

    /// derivative - infinitesimal number
    pub dij_val: Option<MutTensorDD<BatchScalarF64<BATCH>>>,
}

impl<const BATCH: usize> IsDual for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
}

/// Trait for scalar dual numbers
pub trait IsDualScalar<const BATCH: usize>: IsScalar<BATCH, DualScalar = Self> + IsDual {
    /// Create a new dual number
    fn new(val: Self::RealScalar) -> Self;

    /// Create a vector of dual numbers
    fn vector_v<const ROWS: usize>(val: Self::RealVector<ROWS>) -> Self::DualVector<ROWS>;

    /// Create a matrix of dual numbers
    fn matrix_v<const ROWS: usize, const COLS: usize>(
        val: Self::RealMatrix<ROWS, COLS>,
    ) -> Self::DualMatrix<ROWS, COLS>;

    /// Get the derivative
    fn dij_val(self) -> Option<MutTensorDD<Self::RealScalar>>;
}

impl AbsDiffEq for DualScalar {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        1e-6
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.val.abs_diff_eq(&other.val, epsilon)
    }
}

impl RelativeEq for DualScalar {
    fn default_max_relative() -> Self::Epsilon {
        1e-6
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.val.relative_eq(&other.val, epsilon, max_relative)
    }
}

impl IsCoreScalar for DualScalar {
    fn number_category() -> NumberCategory {
        NumberCategory::Real
    }
}

impl SubAssign<DualScalar> for DualScalar {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone().sub(&rhs);
    }
}

impl IsSingleScalar for DualScalar {
    type SingleVector<const ROWS: usize> = DualVector<ROWS>;
    type SingleMatrix<const ROWS: usize, const COLS: usize> = DualMatrix<ROWS, COLS>;

    fn single_real_scalar(&self) -> f64 {
        self.val
    }

    fn single_scalar(&self) -> Self {
        self.clone()
    }

    fn i64_floor(&self) -> i64 {
        self.val.floor() as i64
    }
}

impl AsRef<DualScalar> for DualScalar {
    fn as_ref(&self) -> &DualScalar {
        self
    }
}

impl One for DualScalar {
    fn one() -> Self {
        <DualScalar>::from_f64(1.0)
    }
}

impl Zero for DualScalar {
    fn zero() -> Self {
        <DualScalar>::from_f64(0.0)
    }

    fn is_zero(&self) -> bool {
        self.val == <DualScalar>::from_f64(0.0).real_part()
    }
}

impl IsDualScalar<1> for DualScalar {
    fn new(val: f64) -> Self {
        let dij_val = <MutTensorDD<f64>>::from_shape_and_val([1, 1], 1.0);
        Self {
            val,
            dij_val: Some(dij_val),
        }
    }

    fn vector_v<const ROWS: usize>(val: Self::RealVector<ROWS>) -> Self::Vector<ROWS> {
        DualVector::<ROWS>::v(val)
    }

    fn dij_val(self) -> Option<MutTensorDD<f64>> {
        self.dij_val
    }

    fn matrix_v<const ROWS: usize, const COLS: usize>(
        val: Self::RealMatrix<ROWS, COLS>,
    ) -> Self::Matrix<ROWS, COLS> {
        DualMatrix::<ROWS, COLS>::v(val)
    }
}

impl DualScalar {
    /// create a dual number
    fn binary_dij<F: FnMut(&f64) -> f64, G: FnMut(&f64) -> f64>(
        lhs_dx: &Option<MutTensorDD<f64>>,
        rhs_dx: &Option<MutTensorDD<f64>>,
        mut left_op: F,
        mut right_op: G,
    ) -> Option<MutTensorDD<f64>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => {
                let out_dij =
                    <MutTensorDD<f64>>::from_map(&rhs_dij.view(), |r_dij: &f64| right_op(r_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), None) => {
                let out_dij =
                    <MutTensorDD<f64>>::from_map(&lhs_dij.view(), |l_dij: &f64| left_op(l_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), Some(rhs_dij)) => {
                let dyn_mat = <MutTensorDD<f64>>::from_map2(
                    &lhs_dij.view(),
                    &rhs_dij.view(),
                    |l_dij: &f64, r_dij: &f64| left_op(l_dij) + right_op(r_dij),
                );
                Some(dyn_mat)
            }
        }
    }
}

impl Neg for DualScalar {
    type Output = DualScalar;

    fn neg(self) -> Self {
        Self {
            val: -self.val,
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dyn_mat = <MutTensorDD<f64>>::from_map(&dij_val.view(), |v: &f64| -(*v));

                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }
}

impl PartialEq for DualScalar {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl PartialOrd for DualScalar {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl From<f64> for DualScalar {
    fn from(value: f64) -> Self {
        Self::from_real_scalar(value)
    }
}

impl Debug for DualScalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.dij_val.is_some() {
            f.debug_struct("DualScalar")
                .field("val", &self.val)
                .field("dij_val", &self.dij_val.as_ref().unwrap().elem_view())
                .finish()
        } else {
            f.debug_struct("DualScalar")
                .field("val", &self.val)
                .finish()
        }
    }
}

impl IsScalar<1> for DualScalar {
    type Scalar = DualScalar;
    type RealScalar = f64;
    type SingleScalar = DualScalar;

    type RealMatrix<const ROWS: usize, const COLS: usize> = MatF64<ROWS, COLS>;
    type RealVector<const ROWS: usize> = VecF64<ROWS>;

    type Vector<const ROWS: usize> = DualVector<ROWS>;
    type Matrix<const ROWS: usize, const COLS: usize> = DualMatrix<ROWS, COLS>;

    type Mask = bool;

    fn from_real_scalar(val: f64) -> Self {
        Self { val, dij_val: None }
    }

    fn from_real_array(arr: [f64; 1]) -> Self {
        Self::from_f64(arr[0])
    }

    fn real_array(&self) -> [f64; 1] {
        [self.val]
    }

    fn cos(self) -> DualScalar {
        Self {
            val: self.val.cos(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dyn_mat = <MutTensorDD<f64>>::from_map(&dij_val.view(), |dij: &f64| {
                        -(*dij) * self.val.sin()
                    });
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn sin(self) -> DualScalar {
        Self {
            val: self.val.sin(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dyn_mat = <MutTensorDD<f64>>::from_map(&dij_val.view(), |dij: &f64| {
                        *dij * self.val.cos()
                    });
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn abs(self) -> Self {
        Self {
            val: self.val.abs(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dyn_mat = <MutTensorDD<f64>>::from_map(&dij_val.view(), |dij: &f64| {
                        *dij * self.val.signum()
                    });

                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn atan2(self, rhs: Self) -> Self {
        let inv_sq_nrm: f64 = 1.0 / (self.val * self.val + rhs.val * rhs.val);
        Self {
            val: self.val.atan2(rhs.val),
            dij_val: Self::binary_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| inv_sq_nrm * ((*l_dij) * rhs.val),
                |r_dij| -inv_sq_nrm * (self.val * (*r_dij)),
            ),
        }
    }

    fn real_part(&self) -> f64 {
        self.val
    }

    fn sqrt(self) -> Self {
        let sqrt = self.val.sqrt();
        Self {
            val: sqrt,
            dij_val: match self.dij_val {
                Some(dij) => {
                    let out_dij = <MutTensorDD<f64>>::from_map(&dij.view(), |dij: &f64| {
                        (*dij) * 1.0 / (2.0 * sqrt)
                    });
                    Some(out_dij)
                }
                None => None,
            },
        }
    }

    fn to_vec(self) -> DualVector<1> {
        DualVector::<1> {
            val: self.val.real_part().to_vec(),
            dij_val: match self.dij_val {
                Some(dij) => {
                    let tmp = dij.inner_scalar_to_vec();
                    Some(tmp)
                }
                None => None,
            },
        }
    }

    fn tan(self) -> Self {
        Self {
            val: self.val.tan(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let c = self.val.cos();
                    let sec_squared = 1.0 / (c * c);
                    let dyn_mat = <MutTensorDD<f64>>::from_map(&dij_val.view(), |dij: &f64| {
                        *dij * sec_squared
                    });
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn acos(self) -> Self {
        Self {
            val: self.val.acos(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dval = -1.0 / (1.0 - self.val * self.val).sqrt();
                    let dyn_mat =
                        <MutTensorDD<f64>>::from_map(&dij_val.view(), |dij: &f64| *dij * dval);
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn asin(self) -> Self {
        Self {
            val: self.val.asin(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dval = 1.0 / (1.0 - self.val * self.val).sqrt();
                    let dyn_mat =
                        <MutTensorDD<f64>>::from_map(&dij_val.view(), |dij: &f64| *dij * dval);
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn atan(self) -> Self {
        Self {
            val: self.val.atan(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dval = 1.0 / (1.0 + self.val * self.val);
                    let dyn_mat =
                        <MutTensorDD<f64>>::from_map(&dij_val.view(), |dij: &f64| *dij * dval);
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn fract(self) -> Self {
        Self {
            val: self.val.fract(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dyn_mat = <MutTensorDD<f64>>::from_map(&dij_val.view(), |dij: &f64| *dij);
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn floor(&self) -> f64 {
        self.val.floor()
    }

    fn from_f64(val: f64) -> Self {
        Self { val, dij_val: None }
    }

    fn scalar_examples() -> Vec<Self> {
        [1.0, 2.0, 3.0].iter().map(|&v| Self::from_f64(v)).collect()
    }

    fn extract_single(&self, _i: usize) -> Self::SingleScalar {
        self.clone()
    }

    fn signum(&self) -> Self {
        Self {
            val: self.val.signum(),
            dij_val: None,
        }
    }

    type DualScalar = Self;

    type DualVector<const ROWS: usize> = DualVector<ROWS>;

    type DualMatrix<const ROWS: usize, const COLS: usize> = DualMatrix<ROWS, COLS>;

    fn less_equal(&self, rhs: &Self) -> Self::Mask {
        self.val.less_equal(&rhs.val)
    }

    fn to_dual(self) -> Self::DualScalar {
        self
    }

    fn select(self, mask: &Self::Mask, other: Self) -> Self {
        if *mask {
            self
        } else {
            other
        }
    }

    fn greater_equal(&self, rhs: &Self) -> Self::Mask {
        self.val.greater_equal(&rhs.val)
    }
}

impl AddAssign<DualScalar> for DualScalar {
    fn add_assign(&mut self, rhs: Self) {
        // this is a bit inefficient, better to do it in place
        *self = self.clone().add(&rhs);
    }
}

impl Add<DualScalar> for DualScalar {
    type Output = DualScalar;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl Add<&DualScalar> for DualScalar {
    type Output = DualScalar;
    fn add(self, rhs: &Self) -> Self::Output {
        let r = self.val + rhs.val;

        Self {
            val: r,
            dij_val: Self::binary_dij(&self.dij_val, &rhs.dij_val, |l_dij| *l_dij, |r_dij| *r_dij),
        }
    }
}

impl Mul<DualScalar> for DualScalar {
    type Output = DualScalar;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(&rhs)
    }
}

impl Mul<&DualScalar> for DualScalar {
    type Output = DualScalar;
    fn mul(self, rhs: &Self) -> Self::Output {
        let r = self.val * rhs.val;

        Self {
            val: r,
            dij_val: Self::binary_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| (*l_dij) * rhs.val,
                |r_dij| (*r_dij) * self.val,
            ),
        }
    }
}

impl Div<DualScalar> for DualScalar {
    type Output = DualScalar;
    fn div(self, rhs: Self) -> Self::Output {
        self.div(&rhs)
    }
}

impl Div<&DualScalar> for DualScalar {
    type Output = DualScalar;
    fn div(self, rhs: &Self) -> Self::Output {
        let rhs_inv = 1.0 / rhs.val;
        Self {
            val: self.val * rhs_inv,
            dij_val: Self::binary_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| l_dij * rhs_inv,
                |r_dij| -self.val * r_dij * rhs_inv * rhs_inv,
            ),
        }
    }
}

impl Sub<DualScalar> for DualScalar {
    type Output = DualScalar;
    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(&rhs)
    }
}

impl Sub<&DualScalar> for DualScalar {
    type Output = DualScalar;
    fn sub(self, rhs: &Self) -> Self::Output {
        Self {
            val: self.val - rhs.val,
            dij_val: Self::binary_dij(&self.dij_val, &rhs.dij_val, |l_dij| *l_dij, |r_dij| -r_dij),
        }
    }
}

impl<const BATCH: usize> IsDualScalar<BATCH> for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn new(val: BatchScalarF64<BATCH>) -> Self {
        let dij_val = MutTensorDD::from_shape_and_val([1, 1], BatchScalarF64::<BATCH>::ones());
        Self {
            val,
            dij_val: Some(dij_val),
        }
    }

    fn dij_val(self) -> Option<MutTensorDD<BatchScalarF64<BATCH>>> {
        self.dij_val
    }

    fn vector_v<const ROWS: usize>(val: Self::RealVector<ROWS>) -> Self::Vector<ROWS> {
        DualBatchVector::<ROWS, BATCH>::v(val)
    }

    fn matrix_v<const ROWS: usize, const COLS: usize>(
        val: Self::RealMatrix<ROWS, COLS>,
    ) -> Self::Matrix<ROWS, COLS> {
        DualBatchMatrix::<ROWS, COLS, BATCH>::v(val)
    }
}

impl<const BATCH: usize> IsCoreScalar for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn number_category() -> NumberCategory {
        NumberCategory::Real
    }
}

impl<const BATCH: usize> AsRef<DualBatchScalar<BATCH>> for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn as_ref(&self) -> &DualBatchScalar<BATCH>
    where
        BatchScalarF64<BATCH>: IsCoreScalar,
        LaneCount<BATCH>: SupportedLaneCount,
    {
        self
    }
}

impl<const BATCH: usize> One for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn one() -> Self {
        <DualBatchScalar<BATCH>>::from_f64(1.0)
    }
}

impl<const BATCH: usize> Zero for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn zero() -> Self {
        <DualBatchScalar<BATCH>>::from_f64(0.0)
    }

    fn is_zero(&self) -> bool {
        self.val == <DualBatchScalar<BATCH>>::from_f64(0.0).real_part()
    }
}

impl<const BATCH: usize> DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn binary_dij<
        F: FnMut(&BatchScalarF64<BATCH>) -> BatchScalarF64<BATCH>,
        G: FnMut(&BatchScalarF64<BATCH>) -> BatchScalarF64<BATCH>,
    >(
        lhs_dx: &Option<MutTensorDD<BatchScalarF64<BATCH>>>,
        rhs_dx: &Option<MutTensorDD<BatchScalarF64<BATCH>>>,
        mut left_op: F,
        mut right_op: G,
    ) -> Option<MutTensorDD<BatchScalarF64<BATCH>>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => {
                let out_dij =
                    MutTensorDD::from_map(&rhs_dij.view(), |r_dij: &BatchScalarF64<BATCH>| {
                        right_op(r_dij)
                    });
                Some(out_dij)
            }
            (Some(lhs_dij), None) => {
                let out_dij =
                    MutTensorDD::from_map(&lhs_dij.view(), |l_dij: &BatchScalarF64<BATCH>| {
                        left_op(l_dij)
                    });
                Some(out_dij)
            }
            (Some(lhs_dij), Some(rhs_dij)) => {
                let dyn_mat = MutTensorDD::from_map2(
                    &lhs_dij.view(),
                    &rhs_dij.view(),
                    |l_dij: &BatchScalarF64<BATCH>, r_dij: &BatchScalarF64<BATCH>| {
                        left_op(l_dij) + right_op(r_dij)
                    },
                );
                Some(dyn_mat)
            }
        }
    }
}

impl<const BATCH: usize> Neg for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH>;

    fn neg(self) -> Self {
        Self {
            val: -self.val,
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dyn_mat =
                        MutTensorDD::from_map(&dij_val.view(), |v: &BatchScalarF64<BATCH>| -*v);

                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }
}

impl<const BATCH: usize> PartialEq for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl<const BATCH: usize> AbsDiffEq for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        for i in 0..BATCH {
            if !self
                .val
                .extract_single(i)
                .abs_diff_eq(&other.val.extract_single(i), epsilon.extract_single(i))
            {
                return false;
            }
        }
        true
    }
}

impl<const BATCH: usize> RelativeEq for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
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
            if !self.val.extract_single(i).relative_eq(
                &other.val.extract_single(i),
                epsilon.extract_single(i),
                max_relative.extract_single(i),
            ) {
                return false;
            }
        }
        true
    }
}

impl<const BATCH: usize> From<BatchScalarF64<BATCH>> for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn from(value: BatchScalarF64<BATCH>) -> Self {
        Self::from_real_scalar(value)
    }
}

impl<const BATCH: usize> Debug for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.dij_val.is_some() {
            f.debug_struct("DualScalar")
                .field("val", &self.val)
                .field("dij_val", &self.dij_val.as_ref().unwrap().elem_view())
                .finish()
        } else {
            f.debug_struct("DualScalar")
                .field("val", &self.val)
                .finish()
        }
    }
}

impl<const BATCH: usize> IsScalar<BATCH> for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Scalar = DualBatchScalar<BATCH>;
    type RealScalar = BatchScalarF64<BATCH>;
    type SingleScalar = DualScalar;
    type DualVector<const ROWS: usize> = DualBatchVector<ROWS, BATCH>;
    type DualMatrix<const ROWS: usize, const COLS: usize> = DualBatchMatrix<ROWS, COLS, BATCH>;
    type RealVector<const ROWS: usize> = BatchVecF64<ROWS, BATCH>;
    type RealMatrix<const ROWS: usize, const COLS: usize> = BatchMatF64<ROWS, COLS, BATCH>;
    type Vector<const ROWS: usize> = DualBatchVector<ROWS, BATCH>;
    type Matrix<const ROWS: usize, const COLS: usize> = DualBatchMatrix<ROWS, COLS, BATCH>;

    type Mask = Mask<i64, BATCH>;

    fn from_real_scalar(val: BatchScalarF64<BATCH>) -> Self {
        Self { val, dij_val: None }
    }

    fn scalar_examples() -> Vec<Self> {
        [1.0, 2.0, 3.0].iter().map(|&v| Self::from_f64(v)).collect()
    }

    fn from_real_array(arr: [f64; BATCH]) -> Self {
        Self::from_real_scalar(BatchScalarF64::<BATCH>::from_real_array(arr))
    }

    fn real_array(&self) -> [f64; BATCH] {
        self.val.real_array()
    }

    fn extract_single(&self, i: usize) -> Self::SingleScalar {
        Self::SingleScalar {
            val: self.val.extract_single(i),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dyn_mat =
                        MutTensorDD::from_map(&dij_val.view(), |dij: &BatchScalarF64<BATCH>| {
                            dij.extract_single(i)
                        });
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn cos(self) -> DualBatchScalar<BATCH>
    where
        BatchScalarF64<BATCH>: IsCoreScalar,
        LaneCount<BATCH>: SupportedLaneCount,
    {
        Self {
            val: self.val.cos(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dyn_mat =
                        MutTensorDD::from_map(&dij_val.view(), |dij: &BatchScalarF64<BATCH>| {
                            -*dij * self.val.sin()
                        });
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn signum(&self) -> Self {
        Self {
            val: self.val.signum(),
            dij_val: None,
        }
    }

    fn sin(self) -> DualBatchScalar<BATCH>
    where
        BatchScalarF64<BATCH>: IsCoreScalar,
        LaneCount<BATCH>: SupportedLaneCount,
    {
        Self {
            val: self.val.sin(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dyn_mat =
                        MutTensorDD::from_map(&dij_val.view(), |dij: &BatchScalarF64<BATCH>| {
                            *dij * self.val.cos()
                        });
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn abs(self) -> Self {
        Self {
            val: self.val.abs(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dyn_mat =
                        MutTensorDD::from_map(&dij_val.view(), |dij: &BatchScalarF64<BATCH>| {
                            *dij * self.val.signum()
                        });
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn atan2(self, rhs: Self) -> Self {
        let inv_sq_nrm: BatchScalarF64<BATCH> =
            BatchScalarF64::ones() / (self.val * self.val + rhs.val * rhs.val);
        Self {
            val: self.val.atan2(rhs.val),
            dij_val: Self::binary_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| inv_sq_nrm * ((*l_dij) * rhs.val),
                |r_dij| -inv_sq_nrm * (self.val * (*r_dij)),
            ),
        }
    }

    fn real_part(&self) -> BatchScalarF64<BATCH> {
        self.val
    }

    fn sqrt(self) -> Self {
        let sqrt = self.val.sqrt();
        Self {
            val: sqrt,
            dij_val: match self.dij_val {
                Some(dij) => {
                    let out_dij =
                        MutTensorDD::from_map(&dij.view(), |dij: &BatchScalarF64<BATCH>| {
                            *dij * BatchScalarF64::<BATCH>::from_f64(1.0)
                                / (BatchScalarF64::<BATCH>::from_f64(2.0) * sqrt)
                        });
                    Some(out_dij)
                }
                None => None,
            },
        }
    }

    fn to_vec(self) -> DualBatchVector<1, BATCH> {
        DualBatchVector::<1, BATCH> {
            val: self.val.real_part().to_vec(),
            dij_val: match self.dij_val {
                Some(dij) => {
                    let tmp = dij.inner_scalar_to_vec();
                    Some(tmp)
                }
                None => None,
            },
        }
    }

    fn tan(self) -> Self {
        Self {
            val: self.val.tan(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let c = self.val.cos();
                    let sec_squared = BatchScalarF64::<BATCH>::ones() / (c * c);
                    let dyn_mat =
                        MutTensorDD::from_map(&dij_val.view(), |dij: &BatchScalarF64<BATCH>| {
                            *dij * sec_squared
                        });
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn acos(self) -> Self {
        Self {
            val: self.val.acos(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dval = -BatchScalarF64::<BATCH>::ones()
                        / (BatchScalarF64::<BATCH>::ones() - self.val * self.val).sqrt();
                    let dyn_mat =
                        MutTensorDD::from_map(&dij_val.view(), |dij: &BatchScalarF64<BATCH>| {
                            *dij * dval
                        });
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn asin(self) -> Self {
        Self {
            val: self.val.asin(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dval = BatchScalarF64::<BATCH>::ones()
                        / (BatchScalarF64::<BATCH>::ones() - self.val * self.val).sqrt();
                    let dyn_mat =
                        MutTensorDD::from_map(&dij_val.view(), |dij: &BatchScalarF64<BATCH>| {
                            *dij * dval
                        });
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn atan(self) -> Self {
        Self {
            val: self.val.atan(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dval = BatchScalarF64::<BATCH>::ones()
                        / (BatchScalarF64::<BATCH>::ones() + self.val * self.val);
                    let dyn_mat =
                        MutTensorDD::from_map(&dij_val.view(), |dij: &BatchScalarF64<BATCH>| {
                            *dij * dval
                        });
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn fract(self) -> Self {
        Self {
            val: self.val.fract(),
            dij_val: match self.dij_val.clone() {
                Some(dij_val) => {
                    let dyn_mat =
                        MutTensorDD::from_map(&dij_val.view(), |dij: &BatchScalarF64<BATCH>| *dij);
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn floor(&self) -> BatchScalarF64<BATCH> {
        self.val.floor()
    }

    fn from_f64(val: f64) -> Self {
        Self::from_real_scalar(BatchScalarF64::<BATCH>::from_f64(val))
    }

    type DualScalar = Self;

    fn scalar(self) -> Self {
        self
    }

    fn ones() -> Self {
        Self::from_f64(1.0)
    }

    fn zeros() -> Self {
        Self::from_f64(0.0)
    }

    fn test_suite() {
        let examples = Self::scalar_examples();
        for a in &examples {
            let sin_a = a.clone().sin();
            let cos_a = a.clone().cos();
            let val = sin_a.clone() * sin_a + cos_a.clone() * cos_a;
            let one = Self::ones();

            for i in 0..BATCH {
                assert_abs_diff_eq!(val.extract_single(i), one.extract_single(i));
            }
        }
    }

    fn less_equal(&self, rhs: &Self) -> Self::Mask {
        self.val.less_equal(&rhs.val)
    }

    fn to_dual(self) -> Self::DualScalar {
        self
    }

    fn select(self, mask: &Self::Mask, other: Self) -> Self {
        Self {
            val: self.val.select(mask, other.val),
            dij_val: match (self.dij_val, other.dij_val) {
                (Some(lhs), Some(rhs)) => {
                    let dyn_mat = MutTensorDD::from_map2(
                        &lhs.view(),
                        &rhs.view(),
                        |l: &BatchScalarF64<BATCH>, r: &BatchScalarF64<BATCH>| l.select(mask, *r),
                    );
                    Some(dyn_mat)
                }
                _ => None,
            },
        }
    }

    fn greater_equal(&self, rhs: &Self) -> Self::Mask {
        self.val.greater_equal(&rhs.val)
    }
}

impl<const BATCH: usize> AddAssign<DualBatchScalar<BATCH>> for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone().add(&rhs);
    }
}
impl<const BATCH: usize> SubAssign<DualBatchScalar<BATCH>> for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone().sub(&rhs);
    }
}

impl<const BATCH: usize> Add<DualBatchScalar<BATCH>> for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH>;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl<const BATCH: usize> Add<&DualBatchScalar<BATCH>> for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH>;
    fn add(self, rhs: &Self) -> Self::Output {
        let r = self.val + rhs.val;

        Self {
            val: r,
            dij_val: Self::binary_dij(&self.dij_val, &rhs.dij_val, |l_dij| *l_dij, |r_dij| *r_dij),
        }
    }
}

impl<const BATCH: usize> Mul<DualBatchScalar<BATCH>> for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(&rhs)
    }
}

impl<const BATCH: usize> Mul<&DualBatchScalar<BATCH>> for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH>;
    fn mul(self, rhs: &Self) -> Self::Output {
        let r = self.val * rhs.val;

        Self {
            val: r,
            dij_val: Self::binary_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| (*l_dij) * rhs.val,
                |r_dij| (*r_dij) * self.val,
            ),
        }
    }
}

impl<const BATCH: usize> Div<DualBatchScalar<BATCH>> for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH>;
    fn div(self, rhs: Self) -> Self::Output {
        self.div(&rhs)
    }
}

impl<const BATCH: usize> Div<&DualBatchScalar<BATCH>> for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH>;
    fn div(self, rhs: &Self) -> Self::Output {
        let rhs_inv = BatchScalarF64::<BATCH>::ones() / rhs.val;
        Self {
            val: self.val * rhs_inv,
            dij_val: Self::binary_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| *l_dij * rhs_inv,
                |r_dij| -self.val * (*r_dij) * rhs_inv * rhs_inv,
            ),
        }
    }
}

impl<const BATCH: usize> Sub<DualBatchScalar<BATCH>> for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(&rhs)
    }
}

impl<const BATCH: usize> Sub<&DualBatchScalar<BATCH>> for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchScalar<BATCH>;
    fn sub(self, rhs: &Self) -> Self::Output {
        Self {
            val: self.val - rhs.val,
            dij_val: Self::binary_dij(
                &self.dij_val,
                &rhs.dij_val,
                |l_dij| *l_dij,
                |r_dij| -(*r_dij),
            ),
        }
    }
}

#[test]
fn dual_scalar_tests() {
    use crate::calculus::maps::curves::ScalarValuedCurve;

    trait DualScalarTest {
        fn run_dual_scalar_test();
    }
    macro_rules! def_dual_scalar_test_template {
        ($batch:literal, $scalar: ty, $dual_scalar: ty) => {
            impl DualScalarTest for $dual_scalar {
                fn run_dual_scalar_test() {
                    let b = <$scalar>::from_f64(12.0);
                    for i in 1..10 {
                        let a: $scalar = <$scalar>::from_f64(0.1 * (i as f64));

                        // f(x) = x^2
                        fn square_fn(x: $scalar) -> $scalar {
                            x.clone() * x
                        }
                        fn dual_square_fn(x: $dual_scalar) -> $dual_scalar {
                            x.clone() * x
                        }
                        let finite_diff = ScalarValuedCurve::sym_diff_quotient(square_fn, a, 1e-6);
                        let auto_grad = ScalarValuedCurve::fw_autodiff(dual_square_fn, a);
                        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                        {
                            fn add_fn(x: $scalar, y: $scalar) -> $scalar {
                                x + y
                            }
                            fn dual_add_fn(x: $dual_scalar, y: $dual_scalar) -> $dual_scalar {
                                x + y
                            }

                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| add_fn(x, b), a, 1e-6);
                            let auto_grad = ScalarValuedCurve::fw_autodiff(
                                |x| dual_add_fn(x, <$dual_scalar>::from_real_scalar(b)),
                                a,
                            );
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| add_fn(b, x), a, 1e-6);
                            let auto_grad = ScalarValuedCurve::fw_autodiff(
                                |x| dual_add_fn(<$dual_scalar>::from_real_scalar(b), x),
                                a,
                            );
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                        }

                        {
                            fn sub_fn(x: $scalar, y: $scalar) -> $scalar {
                                x - y
                            }
                            fn dual_sub_fn(x: $dual_scalar, y: $dual_scalar) -> $dual_scalar {
                                x - y
                            }
                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| sub_fn(x, b), a, 1e-6);
                            let auto_grad = ScalarValuedCurve::fw_autodiff(
                                |x| dual_sub_fn(x, <$dual_scalar>::from_real_scalar(b)),
                                a,
                            );
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| sub_fn(b, x), a, 1e-6);
                            let auto_grad = ScalarValuedCurve::fw_autodiff(
                                |x| dual_sub_fn(<$dual_scalar>::from_real_scalar(b), x),
                                a,
                            );
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                        }

                        {
                            fn mul_fn(x: $scalar, y: $scalar) -> $scalar {
                                x * y
                            }
                            fn dual_mul_fn(x: $dual_scalar, y: $dual_scalar) -> $dual_scalar {
                                x * y
                            }
                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| mul_fn(x, b), a, 1e-6);
                            let auto_grad = ScalarValuedCurve::fw_autodiff(
                                |x| dual_mul_fn(x, <$dual_scalar>::from_real_scalar(b)),
                                a,
                            );
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                            let finite_diff =
                                ScalarValuedCurve::sym_diff_quotient(|x| mul_fn(x, b), a, 1e-6);
                            let auto_grad = ScalarValuedCurve::fw_autodiff(
                                |x| dual_mul_fn(x, <$dual_scalar>::from_real_scalar(b)),
                                a,
                            );
                            approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                        }

                        fn div_fn(x: $scalar, y: $scalar) -> $scalar {
                            x / y
                        }
                        fn dual_div_fn(x: $dual_scalar, y: $dual_scalar) -> $dual_scalar {
                            x / y
                        }
                        let finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| div_fn(x, b), a, 1e-6);
                        let auto_grad = ScalarValuedCurve::fw_autodiff(
                            |x| dual_div_fn(x, <$dual_scalar>::from_real_scalar(b)),
                            a,
                        );
                        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                        let finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| div_fn(x, b), a, 1e-6);
                        let auto_grad = ScalarValuedCurve::fw_autodiff(
                            |x| dual_div_fn(x, <$dual_scalar>::from_real_scalar(b)),
                            a,
                        );
                        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                        let finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| div_fn(b, x), a, 1e-6);
                        let auto_grad = ScalarValuedCurve::fw_autodiff(
                            |x| dual_div_fn(<$dual_scalar>::from_real_scalar(b), x),
                            a,
                        );
                        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);

                        let finite_diff =
                            ScalarValuedCurve::sym_diff_quotient(|x| div_fn(x, b), a, 1e-6);
                        let auto_grad = ScalarValuedCurve::fw_autodiff(
                            |x| dual_div_fn(x, <$dual_scalar>::from_real_scalar(b)),
                            a,
                        );
                        approx::assert_abs_diff_eq!(finite_diff, auto_grad, epsilon = 0.0001);
                    }
                }
            }
        };
    }

    def_dual_scalar_test_template!(1, f64, DualScalar);
    def_dual_scalar_test_template!(2, BatchScalarF64<2>, DualBatchScalar<2>);
    def_dual_scalar_test_template!(4, BatchScalarF64<4>, DualBatchScalar<4>);
    def_dual_scalar_test_template!(8, BatchScalarF64<8>, DualBatchScalar<8>);

    DualBatchScalar::<2>::run_dual_scalar_test();
    DualBatchScalar::<4>::run_dual_scalar_test();
    DualBatchScalar::<8>::run_dual_scalar_test();
}
