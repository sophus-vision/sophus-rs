use super::dual_matrix::DualMatrix;
use super::dual_scalar::DualBatchScalar;
use super::dual_scalar::DualScalar;
use crate::calculus::dual::DualBatchMatrix;
use crate::linalg::BatchScalarF64;
use crate::linalg::BatchVecF64;
use crate::linalg::VecF64;
use crate::prelude::*;
use crate::tensor::mut_tensor::InnerVecToMat;
use crate::tensor::mut_tensor::MutTensorDD;
use crate::tensor::mut_tensor::MutTensorDDR;
use approx::AbsDiffEq;
use approx::RelativeEq;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Neg;
use std::ops::Sub;
use std::simd::LaneCount;
use std::simd::Mask;
use std::simd::SupportedLaneCount;

/// Dual vector
#[derive(Clone)]
pub struct DualVector<const ROWS: usize> {
    /// real part
    pub real_part: VecF64<ROWS>,
    /// infinitesimal part - represents derivative
    pub dij_part: Option<MutTensorDDR<f64, ROWS>>,
}

/// Trait for scalar dual numbers
pub trait IsDualVector<S: IsDualScalar<BATCH>, const ROWS: usize, const BATCH: usize>:
    IsVector<S, ROWS, BATCH> + IsDual
{
    /// Create a new dual vector from a real vector for auto-differentiation with respect to self
    ///
    /// Typically this is not called directly, but through using a map auto-differentiation call:
    ///
    ///  - ScalarValuedMapFromVector::fw_autodiff(...);
    ///  - VectorValuedMapFromVector::fw_autodiff(...);
    ///  - MatrixValuedMapFromVector::fw_autodiff(...);
    fn new_with_dij(val: S::RealVector<ROWS>) -> Self;

    /// Get the derivative
    fn dij_val(self) -> Option<MutTensorDDR<S::RealScalar, ROWS>>;
}

/// Dual vector (batch version)
#[derive(Clone, Debug)]
pub struct DualBatchVector<const ROWS: usize, const BATCH: usize>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    /// real part
    pub real_part: BatchVecF64<ROWS, BATCH>,
    /// infinitesimal part - represents derivative
    pub dij_part: Option<MutTensorDDR<BatchScalarF64<BATCH>, ROWS>>,
}

impl<const ROWS: usize> IsDual for DualVector<ROWS> {}

impl<const ROWS: usize> IsDualVector<DualScalar, ROWS, 1> for DualVector<ROWS> {
    fn new_with_dij(val: VecF64<ROWS>) -> Self {
        let mut dij_val = MutTensorDDR::<f64, ROWS>::from_shape([ROWS, 1]);
        for i in 0..ROWS {
            dij_val.mut_view().get_mut([i, 0])[(i, 0)] = 1.0;
        }

        Self {
            real_part: val,
            dij_part: Some(dij_val),
        }
    }

    fn dij_val(self) -> Option<MutTensorDDR<f64, ROWS>> {
        self.dij_part
    }
}

impl<const ROWS: usize> num_traits::Zero for DualVector<ROWS> {
    fn zero() -> Self {
        DualVector {
            real_part: VecF64::zeros(),
            dij_part: None,
        }
    }

    fn is_zero(&self) -> bool {
        self.real_part == VecF64::<ROWS>::zeros()
    }
}

impl<const ROWS: usize> IsSingleVector<DualScalar, ROWS> for DualVector<ROWS>
where
    DualVector<ROWS>: IsVector<DualScalar, ROWS, 1>,
{
    fn set_real_scalar(&mut self, idx: usize, v: f64) {
        self.real_part[idx] = v;
    }
}

struct DijPair<S: IsCoreScalar, const R0: usize, const R1: usize> {
    lhs: MutTensorDDR<S, R0>,
    rhs: MutTensorDDR<S, R1>,
}

impl<S: IsCoreScalar, const R0: usize, const R1: usize> DijPair<S, R0, R1> {
    fn shape(&self) -> [usize; 2] {
        self.lhs.dims()
    }
}

impl<const ROWS: usize> DualVector<ROWS> {
    fn binary_dij<
        const R0: usize,
        const R1: usize,
        F: FnMut(&VecF64<R0>) -> VecF64<ROWS>,
        G: FnMut(&VecF64<R1>) -> VecF64<ROWS>,
    >(
        lhs_dx: &Option<MutTensorDDR<f64, R0>>,
        rhs_dx: &Option<MutTensorDDR<f64, R1>>,
        mut left_op: F,
        mut right_op: G,
    ) -> Option<MutTensorDDR<f64, ROWS>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => {
                let out_dij = MutTensorDDR::from_map(&rhs_dij.view(), |r_dij| right_op(r_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), None) => {
                let out_dij = MutTensorDDR::from_map(&lhs_dij.view(), |l_dij| left_op(l_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), Some(rhs_dij)) => {
                let dyn_mat =
                    MutTensorDDR::from_map2(&lhs_dij.view(), &rhs_dij.view(), |l_dij, r_dij| {
                        left_op(l_dij) + right_op(r_dij)
                    });
                Some(dyn_mat)
            }
        }
    }

    fn binary_vs_dij<
        const R0: usize,
        F: FnMut(&VecF64<R0>) -> VecF64<ROWS>,
        G: FnMut(&f64) -> VecF64<ROWS>,
    >(
        lhs_dx: &Option<MutTensorDDR<f64, R0>>,
        rhs_dx: &Option<MutTensorDD<f64>>,
        mut left_op: F,
        mut right_op: G,
    ) -> Option<MutTensorDDR<f64, ROWS>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => {
                let out_dij = MutTensorDDR::from_map(&rhs_dij.view(), |r_dij| right_op(r_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), None) => {
                let out_dij = MutTensorDDR::from_map(&lhs_dij.view(), |l_dij| left_op(l_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), Some(rhs_dij)) => {
                let dyn_mat =
                    MutTensorDDR::from_map2(&lhs_dij.view(), &rhs_dij.view(), |l_dij, r_dij| {
                        left_op(l_dij) + right_op(r_dij)
                    });
                Some(dyn_mat)
            }
        }
    }

    fn two_dx<const R0: usize, const R1: usize>(
        mut lhs_dx: Option<MutTensorDDR<f64, R0>>,
        mut rhs_dx: Option<MutTensorDDR<f64, R1>>,
    ) -> Option<DijPair<f64, R0, R1>> {
        if lhs_dx.is_none() && rhs_dx.is_none() {
            return None;
        }

        if lhs_dx.is_some() && rhs_dx.is_some() {
            assert_eq!(
                lhs_dx.clone().unwrap().dims(),
                rhs_dx.clone().unwrap().dims()
            );
        }

        if lhs_dx.is_none() {
            lhs_dx = Some(MutTensorDDR::from_shape(rhs_dx.clone().unwrap().dims()))
        } else if rhs_dx.is_none() {
            rhs_dx = Some(MutTensorDDR::from_shape(lhs_dx.clone().unwrap().dims()))
        }

        Some(DijPair {
            lhs: lhs_dx.unwrap(),
            rhs: rhs_dx.unwrap(),
        })
    }
}

impl<const ROWS: usize> Neg for DualVector<ROWS> {
    type Output = DualVector<ROWS>;

    fn neg(self) -> Self::Output {
        DualVector {
            real_part: -self.real_part,
            dij_part: self
                .dij_part
                .clone()
                .map(|dij_val| MutTensorDDR::from_map(&dij_val.view(), |v| -v)),
        }
    }
}

impl<const ROWS: usize> Sub for DualVector<ROWS> {
    type Output = DualVector<ROWS>;

    fn sub(self, rhs: Self) -> Self::Output {
        DualVector {
            real_part: self.real_part - rhs.real_part,
            dij_part: Self::binary_dij(
                &self.dij_part,
                &rhs.dij_part,
                |l_dij| *l_dij,
                |r_dij| -r_dij,
            ),
        }
    }
}

impl<const ROWS: usize> Add for DualVector<ROWS> {
    type Output = DualVector<ROWS>;

    fn add(self, rhs: Self) -> Self::Output {
        DualVector {
            real_part: self.real_part + rhs.real_part,
            dij_part: Self::binary_dij(
                &self.dij_part,
                &rhs.dij_part,
                |l_dij| *l_dij,
                |r_dij| *r_dij,
            ),
        }
    }
}

impl<const ROWS: usize> Debug for DualVector<ROWS> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.dij_part.is_some() {
            f.debug_struct("DualScalarLike")
                .field("val", &self.real_part)
                .field("dij_val", &self.dij_part.as_ref().unwrap().elem_view())
                .finish()
        } else {
            f.debug_struct("DualScalarLike")
                .field("val", &self.real_part)
                .finish()
        }
    }
}

impl<const ROWS: usize> PartialEq for DualVector<ROWS> {
    fn eq(&self, other: &Self) -> bool {
        self.real_part == other.real_part && self.dij_part == other.dij_part
    }
}

impl<const ROWS: usize> AbsDiffEq for DualVector<ROWS> {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.real_part.abs_diff_eq(&other.real_part, epsilon)
    }
}

impl<const ROWS: usize> RelativeEq for DualVector<ROWS> {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
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

impl<const ROWS: usize> IsVector<DualScalar, ROWS, 1> for DualVector<ROWS> {
    fn from_f64(val: f64) -> Self {
        DualVector {
            real_part: VecF64::<ROWS>::from_scalar(val),
            dij_part: None,
        }
    }

    fn norm(&self) -> DualScalar {
        self.clone().dot(self.clone()).sqrt()
    }

    fn squared_norm(&self) -> DualScalar {
        self.clone().dot(self.clone())
    }

    fn get_elem(&self, idx: usize) -> DualScalar {
        DualScalar {
            real_part: self.real_part[idx],
            dij_part: self
                .dij_part
                .clone()
                .map(|dij_val| MutTensorDD::from_map(&dij_val.view(), |v| v[idx])),
        }
    }

    fn from_array(duals: [DualScalar; ROWS]) -> Self {
        let mut shape = None;
        let mut val_v = VecF64::<ROWS>::zeros();
        for i in 0..duals.len() {
            let d = duals.clone()[i].clone();

            val_v[i] = d.real_part;
            if d.dij_part.is_some() {
                shape = Some(d.dij_part.clone().unwrap().dims());
            }
        }

        if shape.is_none() {
            return DualVector {
                real_part: val_v,
                dij_part: None,
            };
        }
        let shape = shape.unwrap();

        let mut r = MutTensorDDR::<f64, ROWS>::from_shape(shape);

        for i in 0..duals.len() {
            let d = duals.clone()[i].clone();
            if d.dij_part.is_some() {
                for d0 in 0..shape[0] {
                    for d1 in 0..shape[1] {
                        r.mut_view().get_mut([d0, d1])[(i, 0)] =
                            d.dij_part.clone().unwrap().get([d0, d1]);
                    }
                }
            }
        }
        DualVector {
            real_part: val_v,
            dij_part: Some(r),
        }
    }

    fn from_real_array(vals: [f64; ROWS]) -> Self {
        DualVector {
            real_part: VecF64::from_real_array(vals),
            dij_part: None,
        }
    }

    fn from_real_vector(val: VecF64<ROWS>) -> Self {
        Self {
            real_part: val,
            dij_part: None,
        }
    }

    fn real_vector(&self) -> &VecF64<ROWS> {
        &self.real_part
    }

    fn to_mat(self) -> DualMatrix<ROWS, 1> {
        DualMatrix {
            real_part: self.real_part,
            dij_part: self.dij_part.map(|dij| dij.inner_vec_to_mat()),
        }
    }

    fn block_vec2<const R0: usize, const R1: usize>(
        top_row: DualVector<R0>,
        bot_row: DualVector<R1>,
    ) -> Self {
        assert_eq!(R0 + R1, ROWS);

        let maybe_dij = Self::two_dx(top_row.dij_part, bot_row.dij_part);
        Self {
            real_part: VecF64::<ROWS>::block_vec2(top_row.real_part, bot_row.real_part),
            dij_part: match maybe_dij {
                Some(dij_val) => {
                    let mut r = MutTensorDDR::<f64, ROWS>::from_shape(dij_val.shape());
                    for d0 in 0..dij_val.shape()[0] {
                        for d1 in 0..dij_val.shape()[1] {
                            *r.mut_view().get_mut([d0, d1]) = VecF64::<ROWS>::block_vec2(
                                dij_val.lhs.get([d0, d1]),
                                dij_val.rhs.get([d0, d1]),
                            );
                        }
                    }
                    Some(r)
                }
                None => None,
            },
        }
    }

    fn scaled(&self, s: DualScalar) -> Self {
        DualVector {
            real_part: self.real_part * s.real_part,
            dij_part: Self::binary_vs_dij(
                &self.dij_part,
                &s.dij_part,
                |l_dij| l_dij * s.real_part,
                |r_dij| self.real_part * *r_dij,
            ),
        }
    }

    fn dot(self, rhs: Self) -> DualScalar {
        let mut sum = <DualScalar>::from_f64(0.0);

        for i in 0..ROWS {
            sum += self.get_elem(i) * rhs.get_elem(i);
        }

        sum
    }

    fn normalized(&self) -> Self {
        self.clone()
            .scaled(<DualScalar>::from_f64(1.0) / self.norm())
    }

    fn from_f64_array(vals: [f64; ROWS]) -> Self {
        DualVector {
            real_part: VecF64::from_f64_array(vals),
            dij_part: None,
        }
    }

    fn from_scalar_array(vals: [DualScalar; ROWS]) -> Self {
        let mut shape = None;
        let mut val_v = VecF64::<ROWS>::zeros();
        for i in 0..vals.len() {
            let d = vals.clone()[i].clone();

            val_v[i] = d.real_part;
            if d.dij_part.is_some() {
                shape = Some(d.dij_part.clone().unwrap().dims());
            }
        }

        if shape.is_none() {
            return DualVector {
                real_part: val_v,
                dij_part: None,
            };
        }
        let shape = shape.unwrap();

        let mut r = MutTensorDDR::<f64, ROWS>::from_shape(shape);

        for i in 0..vals.len() {
            let d = vals.clone()[i].clone();
            if d.dij_part.is_some() {
                for d0 in 0..shape[0] {
                    for d1 in 0..shape[1] {
                        r.mut_view().get_mut([d0, d1])[(i, 0)] =
                            d.dij_part.clone().unwrap().get([d0, d1]);
                    }
                }
            }
        }
        DualVector {
            real_part: val_v,
            dij_part: Some(r),
        }
    }

    fn set_elem(&mut self, idx: usize, v: DualScalar) {
        self.real_part[idx] = v.real_part;
        if self.dij_part.is_some() {
            let dij = &mut self.dij_part.as_mut().unwrap();
            for i in 0..dij.dims()[0] {
                for j in 0..dij.dims()[1] {
                    dij.mut_view().get_mut([i, j])[idx] = v.dij_part.clone().unwrap().get([i, j]);
                }
            }
        }
    }

    fn to_dual(self) -> <DualScalar as IsScalar<1>>::DualVector<ROWS> {
        self
    }

    fn outer<const R2: usize>(
        self,
        rhs: DualVector<R2>,
    ) -> <DualScalar as IsScalar<1>>::Matrix<ROWS, R2> {
        let mut out = DualMatrix::<ROWS, R2>::zeros();
        for i in 0..ROWS {
            for j in 0..R2 {
                out.set_elem([i, j], self.get_elem(i) * rhs.get_elem(j));
            }
        }
        out
    }

    fn select(self, mask: &bool, other: Self) -> Self {
        if *mask {
            self
        } else {
            other
        }
    }

    fn get_fixed_subvec<const R: usize>(&self, start_r: usize) -> DualVector<R> {
        DualVector {
            real_part: self.real_part.fixed_rows::<R>(start_r).into(),
            dij_part: self.dij_part.clone().map(|dij_val| {
                MutTensorDDR::from_map(&dij_val.view(), |v| v.fixed_rows::<R>(start_r).into())
            }),
        }
    }
}

impl<const ROWS: usize, const BATCH: usize> num_traits::Zero for DualBatchVector<ROWS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn zero() -> Self {
        DualBatchVector {
            real_part: BatchVecF64::<ROWS, BATCH>::zeros(),
            dij_part: None,
        }
    }

    fn is_zero(&self) -> bool {
        self.real_part == BatchVecF64::<ROWS, BATCH>::zeros()
    }
}

impl<const ROWS: usize, const BATCH: usize> IsDual for DualBatchVector<ROWS, BATCH> where
    LaneCount<BATCH>: SupportedLaneCount
{
}

impl<const ROWS: usize, const BATCH: usize> PartialEq for DualBatchVector<ROWS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn eq(&self, other: &Self) -> bool {
        self.real_part == other.real_part && self.dij_part == other.dij_part
    }
}

impl<const ROWS: usize, const BATCH: usize> AbsDiffEq for DualBatchVector<ROWS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.real_part.abs_diff_eq(&other.real_part, epsilon)
    }
}

impl<const ROWS: usize, const BATCH: usize> RelativeEq for DualBatchVector<ROWS, BATCH>
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
        self.real_part
            .relative_eq(&other.real_part, epsilon, max_relative)
    }
}

impl<const ROWS: usize, const BATCH: usize> IsDualVector<DualBatchScalar<BATCH>, ROWS, BATCH>
    for DualBatchVector<ROWS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn new_with_dij(val: BatchVecF64<ROWS, BATCH>) -> Self {
        let mut dij_val = MutTensorDDR::<BatchScalarF64<BATCH>, ROWS>::from_shape([ROWS, 1]);
        for i in 0..ROWS {
            dij_val.mut_view().get_mut([i, 0])[(i, 0)] = BatchScalarF64::<BATCH>::ones();
        }

        Self {
            real_part: val,
            dij_part: Some(dij_val),
        }
    }

    fn dij_val(self) -> Option<MutTensorDDR<BatchScalarF64<BATCH>, ROWS>> {
        self.dij_part
    }
}

impl<const ROWS: usize, const BATCH: usize> DualBatchVector<ROWS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn binary_dij<
        const R0: usize,
        const R1: usize,
        F: FnMut(&BatchVecF64<R0, BATCH>) -> BatchVecF64<ROWS, BATCH>,
        G: FnMut(&BatchVecF64<R1, BATCH>) -> BatchVecF64<ROWS, BATCH>,
    >(
        lhs_dx: &Option<MutTensorDDR<BatchScalarF64<BATCH>, R0>>,
        rhs_dx: &Option<MutTensorDDR<BatchScalarF64<BATCH>, R1>>,
        mut left_op: F,
        mut right_op: G,
    ) -> Option<MutTensorDDR<BatchScalarF64<BATCH>, ROWS>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => {
                let out_dij = MutTensorDDR::from_map(&rhs_dij.view(), |r_dij| right_op(r_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), None) => {
                let out_dij = MutTensorDDR::from_map(&lhs_dij.view(), |l_dij| left_op(l_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), Some(rhs_dij)) => {
                let dyn_mat =
                    MutTensorDDR::from_map2(&lhs_dij.view(), &rhs_dij.view(), |l_dij, r_dij| {
                        left_op(l_dij) + right_op(r_dij)
                    });
                Some(dyn_mat)
            }
        }
    }

    fn binary_vs_dij<
        const R0: usize,
        F: FnMut(&BatchVecF64<R0, BATCH>) -> BatchVecF64<ROWS, BATCH>,
        G: FnMut(&BatchScalarF64<BATCH>) -> BatchVecF64<ROWS, BATCH>,
    >(
        lhs_dx: &Option<MutTensorDDR<BatchScalarF64<BATCH>, R0>>,
        rhs_dx: &Option<MutTensorDD<BatchScalarF64<BATCH>>>,
        mut left_op: F,
        mut right_op: G,
    ) -> Option<MutTensorDDR<BatchScalarF64<BATCH>, ROWS>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => {
                let out_dij = MutTensorDDR::from_map(&rhs_dij.view(), |r_dij| right_op(r_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), None) => {
                let out_dij = MutTensorDDR::from_map(&lhs_dij.view(), |l_dij| left_op(l_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), Some(rhs_dij)) => {
                let dyn_mat =
                    MutTensorDDR::from_map2(&lhs_dij.view(), &rhs_dij.view(), |l_dij, r_dij| {
                        left_op(l_dij) + right_op(r_dij)
                    });
                Some(dyn_mat)
            }
        }
    }

    fn two_dx<const R0: usize, const R1: usize>(
        mut lhs_dx: Option<MutTensorDDR<BatchScalarF64<BATCH>, R0>>,
        mut rhs_dx: Option<MutTensorDDR<BatchScalarF64<BATCH>, R1>>,
    ) -> Option<DijPair<BatchScalarF64<BATCH>, R0, R1>> {
        if lhs_dx.is_none() && rhs_dx.is_none() {
            return None;
        }

        if lhs_dx.is_some() && rhs_dx.is_some() {
            assert_eq!(
                lhs_dx.clone().unwrap().dims(),
                rhs_dx.clone().unwrap().dims()
            );
        }

        if lhs_dx.is_none() {
            lhs_dx = Some(MutTensorDDR::from_shape(rhs_dx.clone().unwrap().dims()))
        } else if rhs_dx.is_none() {
            rhs_dx = Some(MutTensorDDR::from_shape(lhs_dx.clone().unwrap().dims()))
        }

        Some(DijPair {
            lhs: lhs_dx.unwrap(),
            rhs: rhs_dx.unwrap(),
        })
    }
}

impl<const ROWS: usize, const BATCH: usize> Neg for DualBatchVector<ROWS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchVector<ROWS, BATCH>;

    fn neg(self) -> Self::Output {
        DualBatchVector {
            real_part: -self.real_part,
            dij_part: self
                .dij_part
                .clone()
                .map(|dij_val| MutTensorDDR::from_map(&dij_val.view(), |v| -v)),
        }
    }
}

impl<const ROWS: usize, const BATCH: usize> Sub for DualBatchVector<ROWS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchVector<ROWS, BATCH>;

    fn sub(self, rhs: Self) -> Self::Output {
        DualBatchVector {
            real_part: self.real_part - rhs.real_part,
            dij_part: Self::binary_dij(
                &self.dij_part,
                &rhs.dij_part,
                |l_dij| *l_dij,
                |r_dij| -r_dij,
            ),
        }
    }
}

impl<const ROWS: usize, const BATCH: usize> Add for DualBatchVector<ROWS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchVector<ROWS, BATCH>;

    fn add(self, rhs: Self) -> Self::Output {
        DualBatchVector {
            real_part: self.real_part + rhs.real_part,
            dij_part: Self::binary_dij(
                &self.dij_part,
                &rhs.dij_part,
                |l_dij| *l_dij,
                |r_dij| *r_dij,
            ),
        }
    }
}

impl<const ROWS: usize, const BATCH: usize> IsVector<DualBatchScalar<BATCH>, ROWS, BATCH>
    for DualBatchVector<ROWS, BATCH>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn from_f64(val: f64) -> Self {
        DualBatchVector {
            real_part:
                <BatchVecF64<ROWS, BATCH> as IsVector<BatchScalarF64<BATCH>, ROWS, BATCH>>::from_f64(
                    val,
                ),
            dij_part: None,
        }
    }

    fn outer<const R2: usize>(
        self,
        rhs: DualBatchVector<R2, BATCH>,
    ) -> DualBatchMatrix<ROWS, R2, BATCH> {
        let mut result = DualBatchMatrix::zeros();
        for i in 0..ROWS {
            for j in 0..R2 {
                result.set_elem([i, j], self.get_elem(i) * rhs.get_elem(j));
            }
        }
        result
    }

    fn norm(&self) -> DualBatchScalar<BATCH> {
        self.clone().dot(self.clone()).sqrt()
    }

    fn squared_norm(&self) -> DualBatchScalar<BATCH> {
        self.clone().dot(self.clone())
    }

    fn get_elem(&self, idx: usize) -> DualBatchScalar<BATCH> {
        DualBatchScalar {
            real_part: self.real_part[idx],
            dij_part: self
                .dij_part
                .clone()
                .map(|dij_val| MutTensorDD::from_map(&dij_val.view(), |v| v[idx])),
        }
    }

    fn from_array(duals: [DualBatchScalar<BATCH>; ROWS]) -> Self {
        let mut shape = None;
        let mut val_v = BatchVecF64::<ROWS, BATCH>::zeros();
        for i in 0..duals.len() {
            let d = duals.clone()[i].clone();

            val_v[i] = d.real_part;
            if d.dij_part.is_some() {
                shape = Some(d.dij_part.clone().unwrap().dims());
            }
        }

        if shape.is_none() {
            return DualBatchVector {
                real_part: val_v,
                dij_part: None,
            };
        }
        let shape = shape.unwrap();

        let mut r = MutTensorDDR::<BatchScalarF64<BATCH>, ROWS>::from_shape(shape);

        for i in 0..duals.len() {
            let d = duals.clone()[i].clone();
            if d.dij_part.is_some() {
                for d0 in 0..shape[0] {
                    for d1 in 0..shape[1] {
                        r.mut_view().get_mut([d0, d1])[(i, 0)] =
                            d.dij_part.clone().unwrap().get([d0, d1]);
                    }
                }
            }
        }
        DualBatchVector {
            real_part: val_v,
            dij_part: Some(r),
        }
    }

    fn from_real_array(vals: [BatchScalarF64<BATCH>; ROWS]) -> Self {
        DualBatchVector {
            real_part: BatchVecF64::from_real_array(vals),
            dij_part: None,
        }
    }

    fn from_real_vector(val: BatchVecF64<ROWS, BATCH>) -> Self {
        Self {
            real_part: val,
            dij_part: None,
        }
    }

    fn real_vector(&self) -> &BatchVecF64<ROWS, BATCH> {
        &self.real_part
    }

    fn to_mat(self) -> DualBatchMatrix<ROWS, 1, BATCH> {
        DualBatchMatrix {
            real_part: self.real_part,
            dij_part: self.dij_part.map(|dij| dij.inner_vec_to_mat()),
        }
    }

    fn block_vec2<const R0: usize, const R1: usize>(
        top_row: DualBatchVector<R0, BATCH>,
        bot_row: DualBatchVector<R1, BATCH>,
    ) -> Self {
        assert_eq!(R0 + R1, ROWS);

        let maybe_dij = Self::two_dx(top_row.dij_part, bot_row.dij_part);
        Self {
            real_part: BatchVecF64::<ROWS, BATCH>::block_vec2(top_row.real_part, bot_row.real_part),
            dij_part: match maybe_dij {
                Some(dij_val) => {
                    let mut r =
                        MutTensorDDR::<BatchScalarF64<BATCH>, ROWS>::from_shape(dij_val.shape());
                    for d0 in 0..dij_val.shape()[0] {
                        for d1 in 0..dij_val.shape()[1] {
                            *r.mut_view().get_mut([d0, d1]) =
                                BatchVecF64::<ROWS, BATCH>::block_vec2(
                                    dij_val.lhs.get([d0, d1]),
                                    dij_val.rhs.get([d0, d1]),
                                );
                        }
                    }
                    Some(r)
                }
                None => None,
            },
        }
    }

    fn scaled(&self, s: DualBatchScalar<BATCH>) -> Self {
        DualBatchVector {
            real_part: self.real_part * s.real_part,
            dij_part: Self::binary_vs_dij(
                &self.dij_part,
                &s.dij_part,
                |l_dij| l_dij * s.real_part,
                |r_dij| self.real_part * *r_dij,
            ),
        }
    }

    fn dot(self, rhs: Self) -> DualBatchScalar<BATCH> {
        let mut sum = DualBatchScalar::from_f64(0.0);

        for i in 0..ROWS {
            sum += self.get_elem(i) * rhs.get_elem(i);
        }

        sum
    }

    fn normalized(&self) -> Self {
        self.clone()
            .scaled(DualBatchScalar::<BATCH>::from_f64(1.0) / self.norm())
    }

    fn from_f64_array(vals: [f64; ROWS]) -> Self {
        DualBatchVector {
            real_part: BatchVecF64::from_f64_array(vals),
            dij_part: None,
        }
    }

    fn from_scalar_array(vals: [DualBatchScalar<BATCH>; ROWS]) -> Self {
        let mut shape = None;
        let mut val_v = BatchVecF64::<ROWS, BATCH>::zeros();
        for i in 0..vals.len() {
            let d = vals.clone()[i].clone();

            val_v[i] = d.real_part;
            if d.dij_part.is_some() {
                shape = Some(d.dij_part.clone().unwrap().dims());
            }
        }

        if shape.is_none() {
            return DualBatchVector {
                real_part: val_v,
                dij_part: None,
            };
        }
        let shape = shape.unwrap();

        let mut r = MutTensorDDR::<BatchScalarF64<BATCH>, ROWS>::from_shape(shape);

        for i in 0..vals.len() {
            let d = vals.clone()[i].clone();
            if d.dij_part.is_some() {
                for d0 in 0..shape[0] {
                    for d1 in 0..shape[1] {
                        r.mut_view().get_mut([d0, d1])[(i, 0)] =
                            d.dij_part.clone().unwrap().get([d0, d1]);
                    }
                }
            }
        }
        DualBatchVector {
            real_part: val_v,
            dij_part: Some(r),
        }
    }

    fn set_elem(&mut self, idx: usize, v: DualBatchScalar<BATCH>) {
        self.real_part[idx] = v.real_part;
        if self.dij_part.is_some() {
            let dij = &mut self.dij_part.as_mut().unwrap();
            for i in 0..dij.dims()[0] {
                for j in 0..dij.dims()[1] {
                    dij.mut_view().get_mut([i, j])[idx] = v.dij_part.clone().unwrap().get([i, j]);
                }
            }
        }
    }

    fn to_dual(self) -> <DualBatchScalar<BATCH> as IsScalar<BATCH>>::DualVector<ROWS> {
        self
    }

    fn select(self, mask: &Mask<i64, BATCH>, other: Self) -> Self {
        let maybe_dij = Self::two_dx(self.dij_part, other.dij_part);

        Self {
            real_part: IsVector::select(self.real_part, mask, other.real_part),
            dij_part: match maybe_dij {
                Some(dij) => {
                    let mut r =
                        MutTensorDDR::<BatchScalarF64<BATCH>, ROWS>::from_shape(dij.shape());
                    for i in 0..dij.shape()[0] {
                        for j in 0..dij.shape()[1] {
                            *r.get_mut([i, j]) =
                                IsVector::select(dij.lhs.get([i, j]), mask, dij.rhs.get([i, j]));
                        }
                    }
                    Some(r)
                }
                _ => None,
            },
        }
    }

    fn get_fixed_subvec<const R: usize>(&self, start_r: usize) -> DualBatchVector<R, BATCH> {
        DualBatchVector {
            real_part: self.real_part.fixed_rows::<R>(start_r).into(),
            dij_part: self.dij_part.clone().map(|dij_val| {
                MutTensorDDR::from_map(&dij_val.view(), |v| v.fixed_rows::<R>(start_r).into())
            }),
        }
    }
}

#[test]
fn dual_vector_tests() {
    use crate::calculus::dual::dual_scalar::DualBatchScalar;
    use crate::calculus::dual::dual_scalar::DualScalar;
    use crate::calculus::maps::scalar_valued_maps::ScalarValuedMapFromVector;
    use crate::calculus::maps::vector_valued_maps::VectorValuedMapFromVector;
    use crate::linalg::vector::IsVector;
    use crate::linalg::BatchScalarF64;
    use crate::points::example_points;

    #[cfg(test)]
    trait Test {
        fn run();
    }

    macro_rules! def_test_template {
        ( $scalar:ty, $dual_scalar: ty, $batch:literal
    ) => {
            #[cfg(test)]
            impl Test for $scalar {
                fn run() {
                    let points = example_points::<$scalar, 4, $batch>();

                    for p in points.clone() {
                        for p1 in points.clone() {
                            {
                                fn dot_fn<S: IsScalar<BATCH>, const BATCH: usize>(
                                    x: S::Vector<4>,
                                    y: S::Vector<4>,
                                ) -> S {
                                    x.dot(y)
                                }
                                let finite_diff =
                                    ScalarValuedMapFromVector::<$scalar, $batch>::sym_diff_quotient(
                                        |x| dot_fn(x, p1),
                                        p,
                                        1e-6,
                                    );
                                let auto_grad =
                                    ScalarValuedMapFromVector::<$dual_scalar, $batch>::fw_autodiff(
                                        |x| {
                                            dot_fn(
                                                x,
                                                <$dual_scalar as IsScalar<$batch>>::Vector::<4>::from_real_vector(p1),
                                            )
                                        },
                                        p,
                                    );
                                approx::assert_abs_diff_eq!(
                                    finite_diff,
                                    auto_grad,
                                    epsilon = 0.0001
                                );
                            }

                            fn dot_fn<S: IsScalar<BATCH>, const BATCH: usize>(x: S::Vector<4>, s: S) -> S::Vector<4> {
                                x.scaled(s)
                            }
                            let finite_diff = VectorValuedMapFromVector::<$scalar, $batch>::sym_diff_quotient(
                                |x| dot_fn::<$scalar, $batch>(x, <$scalar>::from_f64(0.99)),
                                p,
                                1e-6,
                            );
                            let auto_grad = VectorValuedMapFromVector::<$dual_scalar, $batch>::fw_autodiff(
                                |x| dot_fn::<$dual_scalar, $batch>(x, <$dual_scalar>::from_f64(0.99)),
                                p,
                            );
                            for i in 0..finite_diff.dims()[0] {
                                approx::assert_abs_diff_eq!(
                                    finite_diff.get([i]),
                                    auto_grad.get([i]),
                                    epsilon = 0.0001
                                );
                            }

                            let finite_diff = VectorValuedMapFromVector::<$scalar, $batch>::sym_diff_quotient(
                                |x| dot_fn::<$scalar, $batch>(p1, x[0]),
                                p,
                                1e-6,
                            );
                            let auto_grad = VectorValuedMapFromVector::<$dual_scalar, $batch>::fw_autodiff(
                                |x| {
                                    dot_fn::<$dual_scalar, $batch>(
                                        <$dual_scalar as IsScalar<$batch>>::Vector::from_real_vector(p1),
                                        x.get_elem(0),
                                    )
                                },
                                p,
                            );
                            for i in 0..finite_diff.dims()[0] {
                                approx::assert_abs_diff_eq!(
                                    finite_diff.get([i]),
                                    auto_grad.get([i]),
                                    epsilon = 0.0001
                                );
                            }
                        }
                    }
                }
            }
        };
    }

    def_test_template!(f64, DualScalar, 1);
    def_test_template!(BatchScalarF64<2>, DualBatchScalar<2>, 2);
    def_test_template!(BatchScalarF64<4>, DualBatchScalar<4>, 4);

    f64::run();
    BatchScalarF64::<2>::run();
    BatchScalarF64::<4>::run();
}
