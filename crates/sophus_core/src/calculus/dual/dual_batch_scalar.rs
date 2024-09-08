use super::dual_scalar::DualScalar;
pub use crate::calculus::dual::dual_batch_matrix::DualBatchMatrix;
pub use crate::calculus::dual::dual_batch_vector::DualBatchVector;
use crate::linalg::scalar::NumberCategory;
use crate::linalg::BatchMatF64;
use crate::linalg::BatchScalarF64;
use crate::linalg::BatchVecF64;
use crate::linalg::EPS_F64;
use crate::prelude::*;
use crate::tensor::mut_tensor::InnerScalarToVec;
use crate::tensor::mut_tensor::MutTensorDD;
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

/// Dual number - a real number and an infinitesimal number (batch version)
#[derive(Clone)]
pub struct DualBatchScalar<const BATCH: usize>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    /// real part
    pub real_part: BatchScalarF64<BATCH>,

    /// infinitesimal part - represents derivative
    pub dij_part: Option<MutTensorDD<BatchScalarF64<BATCH>>>,
}

impl<const BATCH: usize> IsDual for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
}

impl<const BATCH: usize> IsDualScalar<BATCH> for DualBatchScalar<BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn new_with_dij(val: BatchScalarF64<BATCH>) -> Self {
        let dij_val = MutTensorDD::from_shape_and_val([1, 1], BatchScalarF64::<BATCH>::ones());
        Self {
            real_part: val,
            dij_part: Some(dij_val),
        }
    }

    fn dij_val(self) -> Option<MutTensorDD<BatchScalarF64<BATCH>>> {
        self.dij_part
    }

    fn vector_with_dij<const ROWS: usize>(val: Self::RealVector<ROWS>) -> Self::Vector<ROWS> {
        DualBatchVector::<ROWS, BATCH>::new_with_dij(val)
    }

    fn matrix_with_dij<const ROWS: usize, const COLS: usize>(
        val: Self::RealMatrix<ROWS, COLS>,
    ) -> Self::Matrix<ROWS, COLS> {
        DualBatchMatrix::<ROWS, COLS, BATCH>::new_with_dij(val)
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
        self.real_part == <DualBatchScalar<BATCH>>::from_f64(0.0).real_part()
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
            real_part: -self.real_part,
            dij_part: match self.dij_part.clone() {
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
        self.real_part == other.real_part
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
            if !self.real_part.extract_single(i).abs_diff_eq(
                &other.real_part.extract_single(i),
                epsilon.extract_single(i),
            ) {
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
            if !self.real_part.extract_single(i).relative_eq(
                &other.real_part.extract_single(i),
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
        if self.dij_part.is_some() {
            f.debug_struct("DualScalar")
                .field("val", &self.real_part)
                .field("dij_val", &self.dij_part.as_ref().unwrap().elem_view())
                .finish()
        } else {
            f.debug_struct("DualScalar")
                .field("val", &self.real_part)
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
        Self {
            real_part: val,
            dij_part: None,
        }
    }

    fn scalar_examples() -> Vec<Self> {
        [1.0, 2.0, 3.0].iter().map(|&v| Self::from_f64(v)).collect()
    }

    fn from_real_array(arr: [f64; BATCH]) -> Self {
        Self::from_real_scalar(BatchScalarF64::<BATCH>::from_real_array(arr))
    }

    fn to_real_array(&self) -> [f64; BATCH] {
        self.real_part.to_real_array()
    }

    fn eps() -> Self {
        Self::from_f64(EPS_F64)
    }

    fn extract_single(&self, i: usize) -> Self::SingleScalar {
        Self::SingleScalar {
            real_part: self.real_part.extract_single(i),
            dij_part: match self.dij_part.clone() {
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
            real_part: self.real_part.cos(),
            dij_part: match self.dij_part.clone() {
                Some(dij_val) => {
                    let dyn_mat =
                        MutTensorDD::from_map(&dij_val.view(), |dij: &BatchScalarF64<BATCH>| {
                            -*dij * self.real_part.sin()
                        });
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn signum(&self) -> Self {
        Self {
            real_part: self.real_part.signum(),
            dij_part: None,
        }
    }

    fn sin(self) -> DualBatchScalar<BATCH>
    where
        BatchScalarF64<BATCH>: IsCoreScalar,
        LaneCount<BATCH>: SupportedLaneCount,
    {
        Self {
            real_part: self.real_part.sin(),
            dij_part: match self.dij_part.clone() {
                Some(dij_val) => {
                    let dyn_mat =
                        MutTensorDD::from_map(&dij_val.view(), |dij: &BatchScalarF64<BATCH>| {
                            *dij * self.real_part.cos()
                        });
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn abs(self) -> Self {
        Self {
            real_part: self.real_part.abs(),
            dij_part: match self.dij_part.clone() {
                Some(dij_val) => {
                    let dyn_mat =
                        MutTensorDD::from_map(&dij_val.view(), |dij: &BatchScalarF64<BATCH>| {
                            *dij * self.real_part.signum()
                        });
                    Some(dyn_mat)
                }
                None => None,
            },
        }
    }

    fn atan2(self, rhs: Self) -> Self {
        let inv_sq_nrm: BatchScalarF64<BATCH> = BatchScalarF64::ones()
            / (self.real_part * self.real_part + rhs.real_part * rhs.real_part);
        Self {
            real_part: self.real_part.atan2(rhs.real_part),
            dij_part: Self::binary_dij(
                &self.dij_part,
                &rhs.dij_part,
                |l_dij| inv_sq_nrm * ((*l_dij) * rhs.real_part),
                |r_dij| -inv_sq_nrm * (self.real_part * (*r_dij)),
            ),
        }
    }

    fn real_part(&self) -> BatchScalarF64<BATCH> {
        self.real_part
    }

    fn sqrt(self) -> Self {
        let sqrt = self.real_part.sqrt();
        Self {
            real_part: sqrt,
            dij_part: match self.dij_part {
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
            real_part: self.real_part.real_part().to_vec(),
            dij_part: match self.dij_part {
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
            real_part: self.real_part.tan(),
            dij_part: match self.dij_part.clone() {
                Some(dij_val) => {
                    let c = self.real_part.cos();
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
            real_part: self.real_part.acos(),
            dij_part: match self.dij_part.clone() {
                Some(dij_val) => {
                    let dval = -BatchScalarF64::<BATCH>::ones()
                        / (BatchScalarF64::<BATCH>::ones() - self.real_part * self.real_part)
                            .sqrt();
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
            real_part: self.real_part.asin(),
            dij_part: match self.dij_part.clone() {
                Some(dij_val) => {
                    let dval = BatchScalarF64::<BATCH>::ones()
                        / (BatchScalarF64::<BATCH>::ones() - self.real_part * self.real_part)
                            .sqrt();
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
            real_part: self.real_part.atan(),
            dij_part: match self.dij_part.clone() {
                Some(dij_val) => {
                    let dval = BatchScalarF64::<BATCH>::ones()
                        / (BatchScalarF64::<BATCH>::ones() + self.real_part * self.real_part);
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
            real_part: self.real_part.fract(),
            dij_part: match self.dij_part.clone() {
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
        self.real_part.floor()
    }

    fn from_f64(val: f64) -> Self {
        Self::from_real_scalar(BatchScalarF64::<BATCH>::from_f64(val))
    }

    type DualScalar = Self;

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
        self.real_part.less_equal(&rhs.real_part)
    }

    fn to_dual(self) -> Self::DualScalar {
        self
    }

    fn select(self, mask: &Self::Mask, other: Self) -> Self {
        Self {
            real_part: self.real_part.select(mask, other.real_part),
            dij_part: match (self.dij_part, other.dij_part) {
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
        self.real_part.greater_equal(&rhs.real_part)
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
        let r = self.real_part + rhs.real_part;

        Self {
            real_part: r,
            dij_part: Self::binary_dij(
                &self.dij_part,
                &rhs.dij_part,
                |l_dij| *l_dij,
                |r_dij| *r_dij,
            ),
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
        let r = self.real_part * rhs.real_part;

        Self {
            real_part: r,
            dij_part: Self::binary_dij(
                &self.dij_part,
                &rhs.dij_part,
                |l_dij| (*l_dij) * rhs.real_part,
                |r_dij| (*r_dij) * self.real_part,
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
        let rhs_inv = BatchScalarF64::<BATCH>::ones() / rhs.real_part;
        Self {
            real_part: self.real_part * rhs_inv,
            dij_part: Self::binary_dij(
                &self.dij_part,
                &rhs.dij_part,
                |l_dij| *l_dij * rhs_inv,
                |r_dij| -self.real_part * (*r_dij) * rhs_inv * rhs_inv,
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
            real_part: self.real_part - rhs.real_part,
            dij_part: Self::binary_dij(
                &self.dij_part,
                &rhs.dij_part,
                |l_dij| *l_dij,
                |r_dij| -(*r_dij),
            ),
        }
    }
}
