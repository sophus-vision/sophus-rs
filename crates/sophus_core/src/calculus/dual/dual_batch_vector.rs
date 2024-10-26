pub use crate::calculus::dual::dual_batch_scalar::DualBatchScalar;
use crate::calculus::dual::dual_vector::DijPair;
use crate::calculus::dual::DualBatchMatrix;
use crate::linalg::BatchScalarF64;
use crate::linalg::BatchVecF64;
use crate::prelude::*;
use crate::tensor::mut_tensor::InnerVecToMat;
use crate::tensor::mut_tensor::MutTensorDD;
use crate::tensor::mut_tensor::MutTensorDDR;
use approx::AbsDiffEq;
use approx::RelativeEq;
use core::fmt::Debug;
use core::ops::Add;
use core::ops::Neg;
use core::ops::Sub;
use core::simd::LaneCount;
use core::simd::Mask;
use core::simd::SupportedLaneCount;

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
