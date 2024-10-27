use crate::calculus::dual::dual_batch_scalar::DualBatchScalar;
use crate::calculus::dual::dual_matrix::DijPairMV;
use crate::calculus::dual::DualBatchVector;
use crate::linalg::BatchMatF64;
use crate::linalg::BatchScalarF64;
use crate::linalg::BatchVecF64;
use crate::prelude::*;
use crate::tensor::mut_tensor::MutTensorDD;
use crate::tensor::mut_tensor::MutTensorDDR;
use crate::tensor::mut_tensor::MutTensorDDRC;
use approx::AbsDiffEq;
use approx::RelativeEq;
use core::fmt::Debug;
use core::ops::Add;
use core::ops::Mul;
use core::ops::Neg;
use core::ops::Sub;
use core::simd::LaneCount;
use core::simd::Mask;
use core::simd::SupportedLaneCount;
use num_traits::Zero;

use crate::calculus::dual::dual_matrix::DijPairM;

/// DualScalarLike matrix
#[derive(Clone)]
pub struct DualBatchMatrix<const ROWS: usize, const COLS: usize, const BATCH: usize>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    /// real part
    pub real_part: BatchMatF64<ROWS, COLS, BATCH>,
    /// infinitesimal part - represents derivative
    pub dij_part: Option<MutTensorDDRC<BatchScalarF64<BATCH>, ROWS, COLS>>,
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> IsDual
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize>
    IsDualMatrix<DualBatchScalar<BATCH>, ROWS, COLS, BATCH> for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    /// Create a new dual number
    fn new_with_dij(val: BatchMatF64<ROWS, COLS, BATCH>) -> Self {
        let mut dij_val =
            MutTensorDDRC::<BatchScalarF64<BATCH>, ROWS, COLS>::from_shape([ROWS, COLS]);
        for i in 0..ROWS {
            for j in 0..COLS {
                dij_val.mut_view().get_mut([i, j])[(i, j)] = BatchScalarF64::<BATCH>::from_f64(1.0);
            }
        }

        Self {
            real_part: val,
            dij_part: Some(dij_val),
        }
    }

    /// Get the derivative
    fn dij_val(self) -> Option<MutTensorDDRC<BatchScalarF64<BATCH>, ROWS, COLS>> {
        self.dij_part
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn binary_mm_dij<
        const R0: usize,
        const R1: usize,
        const C0: usize,
        const C1: usize,
        F: FnMut(&BatchMatF64<R0, C0, BATCH>) -> BatchMatF64<ROWS, COLS, BATCH>,
        G: FnMut(&BatchMatF64<R1, C1, BATCH>) -> BatchMatF64<ROWS, COLS, BATCH>,
    >(
        lhs_dx: &Option<MutTensorDDRC<BatchScalarF64<BATCH>, R0, C0>>,
        rhs_dx: &Option<MutTensorDDRC<BatchScalarF64<BATCH>, R1, C1>>,
        mut left_op: F,
        mut right_op: G,
    ) -> Option<MutTensorDDRC<BatchScalarF64<BATCH>, ROWS, COLS>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => {
                let out_dij = MutTensorDDRC::from_map(&rhs_dij.view(), |r_dij| right_op(r_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), None) => {
                let out_dij = MutTensorDDRC::from_map(&lhs_dij.view(), |l_dij| left_op(l_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), Some(rhs_dij)) => {
                let dyn_mat =
                    MutTensorDDRC::from_map2(&lhs_dij.view(), &rhs_dij.view(), |l_dij, r_dij| {
                        left_op(l_dij) + right_op(r_dij)
                    });
                Some(dyn_mat)
            }
        }
    }

    fn binary_mv_dij<
        const R0: usize,
        const R1: usize,
        const C0: usize,
        F: FnMut(&BatchMatF64<R0, C0, BATCH>) -> BatchVecF64<ROWS, BATCH>,
        G: FnMut(&BatchVecF64<R1, BATCH>) -> BatchVecF64<ROWS, BATCH>,
    >(
        lhs_dx: &Option<MutTensorDDRC<BatchScalarF64<BATCH>, R0, C0>>,
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

    fn binary_ms_dij<
        const R0: usize,
        const C0: usize,
        F: FnMut(&BatchMatF64<R0, C0, BATCH>) -> BatchMatF64<ROWS, COLS, BATCH>,
        G: FnMut(&BatchScalarF64<BATCH>) -> BatchMatF64<ROWS, COLS, BATCH>,
    >(
        lhs_dx: &Option<MutTensorDDRC<BatchScalarF64<BATCH>, R0, C0>>,
        rhs_dx: &Option<MutTensorDD<BatchScalarF64<BATCH>>>,
        mut left_op: F,
        mut right_op: G,
    ) -> Option<MutTensorDDRC<BatchScalarF64<BATCH>, ROWS, COLS>> {
        match (lhs_dx, rhs_dx) {
            (None, None) => None,
            (None, Some(rhs_dij)) => {
                let out_dij = MutTensorDDRC::from_map(&rhs_dij.view(), |r_dij| right_op(r_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), None) => {
                let out_dij = MutTensorDDRC::from_map(&lhs_dij.view(), |l_dij| left_op(l_dij));
                Some(out_dij)
            }
            (Some(lhs_dij), Some(rhs_dij)) => {
                let dyn_mat =
                    MutTensorDDRC::from_map2(&lhs_dij.view(), &rhs_dij.view(), |l_dij, r_dij| {
                        left_op(l_dij) + right_op(r_dij)
                    });
                Some(dyn_mat)
            }
        }
    }

    /// derivatives
    pub fn two_dx<const R1: usize, const C1: usize, const R2: usize, const C2: usize>(
        mut lhs_dx: Option<MutTensorDDRC<BatchScalarF64<BATCH>, R1, C1>>,
        mut rhs_dx: Option<MutTensorDDRC<BatchScalarF64<BATCH>, R2, C2>>,
    ) -> Option<DijPairM<BatchScalarF64<BATCH>, R1, C1, R2, C2>> {
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
            lhs_dx = Some(MutTensorDDRC::<BatchScalarF64<BATCH>, R1, C1>::from_shape(
                rhs_dx.clone().unwrap().dims(),
            ))
        } else if rhs_dx.is_none() {
            rhs_dx = Some(MutTensorDDRC::<BatchScalarF64<BATCH>, R2, C2>::from_shape(
                lhs_dx.clone().unwrap().dims(),
            ))
        }

        Some(DijPairM {
            lhs: lhs_dx.unwrap(),
            rhs: rhs_dx.unwrap(),
        })
    }

    /// derivatives
    pub fn two_dx_from_vec(
        mut lhs_dx: Option<MutTensorDDRC<BatchScalarF64<BATCH>, ROWS, COLS>>,
        mut rhs_dx: Option<MutTensorDDR<BatchScalarF64<BATCH>, COLS>>,
    ) -> Option<DijPairMV<BatchScalarF64<BATCH>, ROWS, COLS>> {
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
            lhs_dx = Some(
                MutTensorDDRC::<BatchScalarF64<BATCH>, ROWS, COLS>::from_shape(
                    rhs_dx.clone().unwrap().dims(),
                ),
            )
        } else if rhs_dx.is_none() {
            rhs_dx = Some(MutTensorDDR::<BatchScalarF64<BATCH>, COLS>::from_shape(
                lhs_dx.clone().unwrap().dims(),
            ))
        }

        Some(DijPairMV::<BatchScalarF64<BATCH>, ROWS, COLS> {
            lhs: lhs_dx.unwrap(),
            rhs: rhs_dx.unwrap(),
        })
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> PartialEq
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn eq(&self, other: &Self) -> bool {
        self.real_part == other.real_part && self.dij_part == other.dij_part
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> AbsDiffEq
    for DualBatchMatrix<ROWS, COLS, BATCH>
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

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> RelativeEq
    for DualBatchMatrix<ROWS, COLS, BATCH>
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

impl<const ROWS: usize, const COLS: usize, const BATCH: usize>
    IsMatrix<DualBatchScalar<BATCH>, ROWS, COLS, BATCH> for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn from_f64(val: f64) -> Self {
        DualBatchMatrix {
            real_part: BatchMatF64::<ROWS, COLS, BATCH>::from_f64(val),
            dij_part: None,
        }
    }

    fn set_elem(&mut self, idx: [usize; 2], val: DualBatchScalar<BATCH>) {
        self.real_part.set_elem(idx, val.real_part);
        if self.dij_part.is_some() {
            let dij = &mut self.dij_part.as_mut().unwrap();
            for i in 0..dij.dims()[0] {
                for j in 0..dij.dims()[1] {
                    dij.mut_view().get_mut([i, j])[(idx[0], idx[1])] =
                        val.dij_part.clone().unwrap().get([i, j]);
                }
            }
        }
    }

    fn from_scalar(val: DualBatchScalar<BATCH>) -> Self {
        DualBatchMatrix {
            real_part: BatchMatF64::<ROWS, COLS, BATCH>::from_scalar(val.real_part),
            dij_part: val.dij_part.map(|dij_val| {
                MutTensorDDRC::from_map(&dij_val.view(), |v| {
                    BatchMatF64::<ROWS, COLS, BATCH>::from_scalar(*v)
                })
            }),
        }
    }

    fn mat_mul<const COLS2: usize>(
        &self,
        rhs: DualBatchMatrix<COLS, COLS2, BATCH>,
    ) -> DualBatchMatrix<ROWS, COLS2, BATCH> {
        DualBatchMatrix {
            real_part: self.real_part * rhs.real_part,
            dij_part: DualBatchMatrix::<ROWS, COLS2, BATCH>::binary_mm_dij(
                &self.dij_part,
                &rhs.dij_part,
                |l_dij| l_dij * rhs.real_part,
                |r_dij| self.real_part * r_dij,
            ),
        }
    }

    fn from_real_matrix(val: BatchMatF64<ROWS, COLS, BATCH>) -> Self {
        Self {
            real_part: val,
            dij_part: None,
        }
    }

    fn scaled(&self, s: DualBatchScalar<BATCH>) -> Self {
        DualBatchMatrix {
            real_part: self.real_part * s.real_part,
            dij_part: DualBatchMatrix::<ROWS, COLS, BATCH>::binary_ms_dij(
                &self.dij_part,
                &s.dij_part,
                |l_dij| l_dij * s.real_part,
                |r_dij| self.real_part * *r_dij,
            ),
        }
    }

    fn identity() -> Self {
        DualBatchMatrix::from_real_matrix(BatchMatF64::<ROWS, COLS, BATCH>::identity())
    }

    fn get_elem(&self, idx: [usize; 2]) -> DualBatchScalar<BATCH> {
        DualBatchScalar::<BATCH> {
            real_part: self.real_part.get_elem(idx),
            dij_part: self
                .dij_part
                .clone()
                .map(|dij_val| MutTensorDD::from_map(&dij_val.view(), |v| v[(idx[0], idx[1])])),
        }
    }

    fn from_array2(duals: [[DualBatchScalar<BATCH>; COLS]; ROWS]) -> Self {
        let mut shape = None;
        let mut val_mat = BatchMatF64::<ROWS, COLS, BATCH>::zeros();
        for i in 0..duals.len() {
            let d_rows = duals[i].clone();
            for j in 0..d_rows.len() {
                let d = d_rows.clone()[j].clone();

                val_mat[(i, j)] = d.real_part;
                if d.dij_part.is_some() {
                    shape = Some(d.dij_part.clone().unwrap().dims());
                }
            }
        }

        if shape.is_none() {
            return DualBatchMatrix {
                real_part: val_mat,
                dij_part: None,
            };
        }
        let shape = shape.unwrap();

        let mut r = MutTensorDDRC::<BatchScalarF64<BATCH>, ROWS, COLS>::from_shape(shape);
        for i in 0..duals.len() {
            let d_rows = duals[i].clone();
            for j in 0..d_rows.len() {
                let d = d_rows.clone()[j].clone();
                if d.dij_part.is_some() {
                    for d0 in 0..shape[0] {
                        for d1 in 0..shape[1] {
                            r.mut_view().get_mut([d0, d1])[(i, j)] =
                                d.dij_part.clone().unwrap().get([d0, d1]);
                        }
                    }
                }
            }
        }
        DualBatchMatrix {
            real_part: val_mat,
            dij_part: Some(r),
        }
    }

    fn real_matrix(&self) -> &BatchMatF64<ROWS, COLS, BATCH> {
        &self.real_part
    }

    fn block_mat2x2<const R0: usize, const R1: usize, const C0: usize, const C1: usize>(
        top_row: (
            DualBatchMatrix<R0, C0, BATCH>,
            DualBatchMatrix<R0, C1, BATCH>,
        ),
        bot_row: (
            DualBatchMatrix<R1, C0, BATCH>,
            DualBatchMatrix<R1, C1, BATCH>,
        ),
    ) -> Self {
        assert_eq!(R0 + R1, ROWS);
        assert_eq!(C0 + C1, COLS);

        Self::block_mat2x1(
            DualBatchMatrix::<R0, COLS, BATCH>::block_mat1x2(top_row.0, top_row.1),
            DualBatchMatrix::<R1, COLS, BATCH>::block_mat1x2(bot_row.0, bot_row.1),
        )
    }

    fn block_mat2x1<const R0: usize, const R1: usize>(
        top_row: DualBatchMatrix<R0, COLS, BATCH>,
        bot_row: DualBatchMatrix<R1, COLS, BATCH>,
    ) -> Self {
        assert_eq!(R0 + R1, ROWS);
        let maybe_dij = Self::two_dx(top_row.dij_part, bot_row.dij_part);

        Self {
            real_part: BatchMatF64::<ROWS, COLS, BATCH>::block_mat2x1(
                top_row.real_part,
                bot_row.real_part,
            ),
            dij_part: match maybe_dij {
                Some(dij_val) => {
                    let mut r = MutTensorDDRC::<BatchScalarF64<BATCH>, ROWS, COLS>::from_shape(
                        dij_val.shape(),
                    );
                    for d0 in 0..dij_val.shape()[0] {
                        for d1 in 0..dij_val.shape()[1] {
                            *r.mut_view().get_mut([d0, d1]) =
                                BatchMatF64::<ROWS, COLS, BATCH>::block_mat2x1(
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

    fn block_mat1x2<const C0: usize, const C1: usize>(
        left_col: DualBatchMatrix<ROWS, C0, BATCH>,
        righ_col: DualBatchMatrix<ROWS, C1, BATCH>,
    ) -> Self {
        assert_eq!(C0 + C1, COLS);
        let maybe_dij = Self::two_dx(left_col.dij_part, righ_col.dij_part);

        Self {
            real_part: BatchMatF64::<ROWS, COLS, BATCH>::block_mat1x2(
                left_col.real_part,
                righ_col.real_part,
            ),
            dij_part: match maybe_dij {
                Some(dij_val) => {
                    let mut r = MutTensorDDRC::<BatchScalarF64<BATCH>, ROWS, COLS>::from_shape(
                        dij_val.shape(),
                    );
                    for d0 in 0..dij_val.shape()[0] {
                        for d1 in 0..dij_val.shape()[1] {
                            *r.mut_view().get_mut([d0, d1]) =
                                BatchMatF64::<ROWS, COLS, BATCH>::block_mat1x2(
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

    fn get_fixed_submat<const R: usize, const C: usize>(
        &self,
        start_r: usize,
        start_c: usize,
    ) -> DualBatchMatrix<R, C, BATCH> {
        DualBatchMatrix {
            real_part: self.real_part.get_fixed_submat(start_r, start_c),
            dij_part: self.dij_part.clone().map(|dij_val| {
                MutTensorDDRC::from_map(&dij_val.view(), |v| v.get_fixed_submat(start_r, start_c))
            }),
        }
    }

    fn get_col_vec(&self, start_r: usize) -> DualBatchVector<ROWS, BATCH> {
        DualBatchVector {
            real_part: self.real_part.get_col_vec(start_r),
            dij_part: self
                .dij_part
                .clone()
                .map(|dij_val| MutTensorDDR::from_map(&dij_val.view(), |v| v.get_col_vec(start_r))),
        }
    }

    fn get_row_vec(&self, c: usize) -> DualBatchVector<COLS, BATCH> {
        DualBatchVector {
            real_part: self.real_part.get_row_vec(c),
            dij_part: self
                .dij_part
                .clone()
                .map(|dij_val| MutTensorDDR::from_map(&dij_val.view(), |v| v.get_row_vec(c))),
        }
    }

    fn from_real_scalar_array2(vals: [[BatchScalarF64<BATCH>; COLS]; ROWS]) -> Self {
        DualBatchMatrix {
            real_part: BatchMatF64::from_real_scalar_array2(vals),
            dij_part: None,
        }
    }

    fn from_f64_array2(vals: [[f64; COLS]; ROWS]) -> Self {
        DualBatchMatrix {
            real_part: BatchMatF64::from_f64_array2(vals),
            dij_part: None,
        }
    }

    fn set_col_vec(
        &mut self,
        c: usize,
        v: <DualBatchScalar<BATCH> as IsScalar<BATCH>>::Vector<ROWS>,
    ) {
        self.real_part.set_col_vec(c, v.real_part);
        todo!();
    }

    fn to_dual(self) -> <DualBatchScalar<BATCH> as IsScalar<BATCH>>::DualMatrix<ROWS, COLS> {
        self
    }

    fn select(self, mask: &Mask<i64, BATCH>, other: Self) -> Self {
        let maybe_dij = Self::two_dx(self.dij_part, other.dij_part);

        DualBatchMatrix {
            real_part: self.real_part.select(mask, other.real_part),
            dij_part: match maybe_dij {
                Some(dij) => {
                    let mut r =
                        MutTensorDDRC::<BatchScalarF64<BATCH>, ROWS, COLS>::from_shape(dij.shape());
                    for i in 0..dij.shape()[0] {
                        for j in 0..dij.shape()[1] {
                            *r.get_mut([i, j]) =
                                dij.lhs.get([i, j]).select(mask, dij.rhs.get([i, j]));
                        }
                    }
                    Some(r)
                }
                _ => None,
            },
        }
    }

    fn transposed(self) -> DualBatchMatrix<COLS, ROWS, BATCH> {
        DualBatchMatrix {
            real_part: self.real_part.transpose(),
            dij_part: self.dij_part.clone().map(|_dij_val| todo!()),
        }
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> Add
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchMatrix<ROWS, COLS, BATCH>;

    fn add(self, rhs: Self) -> Self::Output {
        DualBatchMatrix {
            real_part: self.real_part + rhs.real_part,
            dij_part: Self::binary_mm_dij(
                &self.dij_part,
                &rhs.dij_part,
                |l_dij| *l_dij,
                |r_dij| *r_dij,
            ),
        }
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> Sub
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchMatrix<ROWS, COLS, BATCH>;

    fn sub(self, rhs: Self) -> Self::Output {
        DualBatchMatrix {
            real_part: self.real_part - rhs.real_part,
            dij_part: Self::binary_mm_dij(
                &self.dij_part,
                &rhs.dij_part,
                |l_dij| *l_dij,
                |r_dij| -r_dij,
            ),
        }
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> Neg
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchMatrix<ROWS, COLS, BATCH>;

    fn neg(self) -> Self::Output {
        DualBatchMatrix {
            real_part: -self.real_part,
            dij_part: self
                .dij_part
                .clone()
                .map(|dij_val| MutTensorDDRC::from_map(&dij_val.view(), |v| -v)),
        }
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> Zero
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn zero() -> Self {
        Self::from_real_matrix(BatchMatF64::zeros())
    }

    fn is_zero(&self) -> bool {
        self.real_part.is_zero()
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> Mul<DualBatchVector<COLS, BATCH>>
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchVector<ROWS, BATCH>;

    fn mul(self, rhs: DualBatchVector<COLS, BATCH>) -> Self::Output {
        Self::Output {
            real_part: self.real_part * rhs.real_part,
            dij_part: Self::binary_mv_dij(
                &self.dij_part,
                &rhs.dij_part,
                |l_dij| l_dij * rhs.real_part,
                |r_dij| self.real_part * r_dij,
            ),
        }
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize> Debug
    for DualBatchMatrix<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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
