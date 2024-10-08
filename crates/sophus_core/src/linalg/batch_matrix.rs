use crate::calculus::dual::DualBatchMatrix;

use crate::linalg::BatchMatF64;
use crate::linalg::BatchScalarF64;
use crate::linalg::BatchVecF64;

use crate::prelude::*;

use std::simd::LaneCount;
use std::simd::Mask;
use std::simd::SupportedLaneCount;

impl<const ROWS: usize, const COLS: usize, const BATCH: usize>
    IsMatrix<BatchScalarF64<BATCH>, ROWS, COLS, BATCH> for BatchMatF64<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn from_scalar(val: BatchScalarF64<BATCH>) -> Self {
        Self::from_element(val)
    }

    fn from_real_matrix(val: Self) -> Self {
        val
    }

    fn real_matrix(&self) -> &Self {
        self
    }

    fn scaled(&self, v: BatchScalarF64<BATCH>) -> Self {
        self * v
    }

    fn identity() -> Self {
        nalgebra::SMatrix::<BatchScalarF64<BATCH>, ROWS, COLS>::identity()
    }

    fn from_array2(vals: [[BatchScalarF64<BATCH>; COLS]; ROWS]) -> Self {
        Self::from_fn(|r, c| vals[r][c])
    }

    fn from_real_scalar_array2(vals: [[BatchScalarF64<BATCH>; COLS]; ROWS]) -> Self {
        Self::from_fn(|r, c| vals[r][c])
    }

    fn from_f64_array2(vals: [[f64; COLS]; ROWS]) -> Self {
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

    fn mat_mul<const C2: usize>(
        &self,
        other: BatchMatF64<COLS, C2, BATCH>,
    ) -> BatchMatF64<ROWS, C2, BATCH> {
        self * other
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

    fn to_dual(self) -> <BatchScalarF64<BATCH> as IsScalar<BATCH>>::DualMatrix<ROWS, COLS> {
        DualBatchMatrix::from_real_matrix(self)
    }

    fn select(self, mask: &Mask<i64, BATCH>, other: Self) -> Self {
        self.zip_map(&other, |a, b| a.select(mask, b))
    }

    fn set_elem(&mut self, idx: [usize; 2], val: BatchScalarF64<BATCH>) {
        self[(idx[0], idx[1])] = val;
    }

    fn transposed(self) -> BatchMatF64<COLS, ROWS, BATCH> {
        self.transpose()
    }
}

impl<const ROWS: usize, const COLS: usize, const BATCH: usize>
    IsRealMatrix<BatchScalarF64<BATCH>, ROWS, COLS, BATCH> for BatchMatF64<ROWS, COLS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
}
