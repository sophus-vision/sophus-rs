use core::{
    borrow::Borrow,
    simd::{
        LaneCount,
        SupportedLaneCount,
    },
};

use super::batch_mask::BatchMask;
use crate::{
    dual::DualBatchMatrix,
    linalg::{
        BatchMatF64,
        BatchScalarF64,
        BatchVecF64,
    },
    prelude::*,
};

impl<const ROWS: usize, const COLS: usize, const BATCH: usize>
    IsMatrix<BatchScalarF64<BATCH>, ROWS, COLS, BATCH, 0, 0> for BatchMatF64<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn from_scalar<S>(val: S) -> Self
    where
        S: Borrow<BatchScalarF64<BATCH>>,
    {
        let val = val.borrow();
        Self::from_element(*val)
    }

    fn from_real_matrix<M>(val: M) -> Self
    where
        M: Borrow<BatchMatF64<ROWS, COLS, BATCH>>,
    {
        *val.borrow()
    }

    fn real_matrix(&self) -> Self {
        *self
    }

    fn scaled<S>(&self, v: S) -> Self
    where
        S: Borrow<BatchScalarF64<BATCH>>,
    {
        *self * *v.borrow()
    }

    fn identity() -> Self {
        nalgebra::SMatrix::<BatchScalarF64<BATCH>, ROWS, COLS>::identity()
    }

    fn from_array2<A>(vals: A) -> Self
    where
        A: Borrow<[[BatchScalarF64<BATCH>; COLS]; ROWS]>,
    {
        let vals = vals.borrow();
        Self::from_fn(|r, c| vals[r][c])
    }

    fn from_real_scalar_array2<A>(vals: A) -> Self
    where
        A: Borrow<[[BatchScalarF64<BATCH>; COLS]; ROWS]>,
    {
        let vals = vals.borrow();
        Self::from_fn(|r, c| vals[r][c])
    }

    fn from_f64_array2<A>(vals: A) -> Self
    where
        A: Borrow<[[f64; COLS]; ROWS]>,
    {
        let vals = vals.borrow();
        Self::from_fn(|r, c| BatchScalarF64::<BATCH>::from_f64(vals[r][c]))
    }

    fn get_elem(&self, idx: [usize; 2]) -> BatchScalarF64<BATCH> {
        self[(idx[0], idx[1])]
    }

    fn block_mat2x1<const R0: usize, const R1: usize>(
        top_row: BatchMatF64<R0, COLS, BATCH>,
        bot_row: BatchMatF64<R1, COLS, BATCH>,
    ) -> Self {
        Self::from_fn(|r, c| {
            if r < R0 {
                top_row[(r, c)]
            } else {
                bot_row[(r - R0, c)]
            }
        })
    }

    fn block_mat1x2<const C0: usize, const C1: usize>(
        left_col: BatchMatF64<ROWS, C0, BATCH>,
        righ_col: BatchMatF64<ROWS, C1, BATCH>,
    ) -> Self {
        Self::from_fn(|r, c| {
            if c < C0 {
                left_col[(r, c)]
            } else {
                righ_col[(r, c - C0)]
            }
        })
    }

    fn block_mat2x2<const R0: usize, const R1: usize, const C0: usize, const C1: usize>(
        top_row: (BatchMatF64<R0, C0, BATCH>, BatchMatF64<R0, C1, BATCH>),
        bot_row: (BatchMatF64<R1, C0, BATCH>, BatchMatF64<R1, C1, BATCH>),
    ) -> Self {
        Self::from_fn(|r, c| {
            if r < R0 {
                if c < C0 {
                    top_row.0[(r, c)]
                } else {
                    top_row.1[(r, c - C0)]
                }
            } else if c < C0 {
                bot_row.0[(r - R0, c)]
            } else {
                bot_row.1[(r - R0, c - C0)]
            }
        })
    }

    fn mat_mul<const C2: usize, M>(&self, other: M) -> BatchMatF64<ROWS, C2, BATCH>
    where
        M: Borrow<BatchMatF64<COLS, C2, BATCH>>,
    {
        self * other.borrow()
    }

    fn set_col_vec(&mut self, c: usize, v: BatchVecF64<ROWS, BATCH>) {
        self.fixed_columns_mut::<1>(c).copy_from(&v);
    }

    fn get_fixed_submat<const R: usize, const C: usize>(
        &self,
        start_r: usize,
        start_c: usize,
    ) -> BatchMatF64<R, C, BATCH> {
        self.fixed_view::<R, C>(start_r, start_c).into()
    }

    fn get_col_vec(&self, c: usize) -> BatchVecF64<ROWS, BATCH> {
        self.fixed_view::<ROWS, 1>(0, c).into()
    }

    fn get_row_vec(&self, r: usize) -> BatchVecF64<COLS, BATCH> {
        self.fixed_view::<1, COLS>(r, 0).transpose()
    }

    fn from_f64(val: f64) -> Self {
        Self::from_element(BatchScalarF64::<BATCH>::from_f64(val))
    }

    fn to_dual_const<const M: usize, const N: usize>(
        &self,
    ) -> DualBatchMatrix<ROWS, COLS, BATCH, M, N> {
        DualBatchMatrix::from_real_matrix(self)
    }

    fn select<Q>(&self, mask: &BatchMask<BATCH>, other: Q) -> Self
    where
        Q: Borrow<Self>,
    {
        self.zip_map(other.borrow(), |a, b| a.select(mask, b))
    }

    fn set_elem(&mut self, idx: [usize; 2], val: BatchScalarF64<BATCH>) {
        self[(idx[0], idx[1])] = val;
    }

    fn transposed(&self) -> BatchMatF64<COLS, ROWS, BATCH> {
        self.transpose()
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize>
    IsRealMatrix<BatchScalarF64<BATCH>, ROWS, COLS, BATCH> for BatchMatF64<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
}
