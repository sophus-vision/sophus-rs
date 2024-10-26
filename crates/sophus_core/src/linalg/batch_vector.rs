use crate::calculus::dual::DualBatchVector;
use crate::linalg::BatchMatF64;
use crate::linalg::BatchScalarF64;
use crate::linalg::BatchVecF64;
use crate::prelude::*;
use core::simd::LaneCount;
use core::simd::Mask;
use core::simd::SupportedLaneCount;

impl<const ROWS: usize, const BATCH: usize> IsVector<BatchScalarF64<BATCH>, ROWS, BATCH>
    for BatchVecF64<ROWS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn block_vec2<const R0: usize, const R1: usize>(
        top_row: BatchVecF64<R0, BATCH>,
        bot_row: BatchVecF64<R1, BATCH>,
    ) -> Self {
        assert_eq!(ROWS, R0 + R1);
        let mut m = Self::zeros();

        m.fixed_view_mut::<R0, 1>(0, 0).copy_from(&top_row);
        m.fixed_view_mut::<R1, 1>(R0, 0).copy_from(&bot_row);
        m
    }

    fn dot(self, rhs: Self) -> BatchScalarF64<BATCH> {
        (self.transpose() * rhs)[0]
    }

    fn from_array(vals: [BatchScalarF64<BATCH>; ROWS]) -> Self {
        Self::from_fn(|i, _| vals[i])
    }

    fn from_real_array(vals: [BatchScalarF64<BATCH>; ROWS]) -> Self {
        Self::from_fn(|i, _| vals[i])
    }

    fn from_f64_array(vals: [f64; ROWS]) -> Self {
        Self::from_fn(|i, _| BatchScalarF64::<BATCH>::from_f64(vals[i]))
    }

    fn from_scalar_array(vals: [BatchScalarF64<BATCH>; ROWS]) -> Self {
        Self::from_fn(|i, _| vals[i])
    }

    fn from_real_vector(val: BatchVecF64<ROWS, BATCH>) -> Self {
        val
    }

    fn get_elem(&self, idx: usize) -> BatchScalarF64<BATCH> {
        self[idx]
    }

    fn norm(&self) -> BatchScalarF64<BATCH> {
        self.squared_norm().sqrt()
    }

    fn normalized(&self) -> Self {
        let norm = self.norm();
        if norm == <BatchScalarF64<BATCH> as IsScalar<BATCH>>::zeros() {
            return *self;
        }
        let factor = BatchScalarF64::<BATCH>::ones() / norm;
        self * factor
    }

    fn real_vector(&self) -> &BatchVecF64<ROWS, BATCH> {
        self
    }

    fn scaled(&self, v: BatchScalarF64<BATCH>) -> Self {
        self * v
    }

    fn set_elem(&mut self, idx: usize, v: BatchScalarF64<BATCH>) {
        self[idx] = v;
    }

    fn squared_norm(&self) -> BatchScalarF64<BATCH> {
        let mut squared_norm = <BatchScalarF64<BATCH> as IsScalar<BATCH>>::zeros();
        for i in 0..ROWS {
            let val = IsVector::get_elem(self, i);
            squared_norm += val * val;
        }
        squared_norm
    }

    fn to_mat(self) -> BatchMatF64<ROWS, 1, BATCH> {
        self
    }

    fn from_f64(val: f64) -> Self {
        Self::from_element(BatchScalarF64::<BATCH>::from_f64(val))
    }

    fn to_dual(self) -> <BatchScalarF64<BATCH> as IsScalar<BATCH>>::DualVector<ROWS> {
        DualBatchVector::from_real_vector(self)
    }

    fn outer<const R2: usize>(self, rhs: BatchVecF64<R2, BATCH>) -> BatchMatF64<ROWS, R2, BATCH> {
        self * rhs.transpose()
    }

    fn select(self, mask: &Mask<i64, BATCH>, other: Self) -> Self {
        self.zip_map(&other, |a, b| a.select(mask, b))
    }

    fn get_fixed_subvec<const R: usize>(&self, start_r: usize) -> BatchVecF64<R, BATCH> {
        self.fixed_rows::<R>(start_r).into()
    }
}

impl<const ROWS: usize, const BATCH: usize> IsRealVector<BatchScalarF64<BATCH>, ROWS, BATCH>
    for BatchVecF64<ROWS, BATCH>
where
    LaneCount<BATCH>: SupportedLaneCount,
{
}
