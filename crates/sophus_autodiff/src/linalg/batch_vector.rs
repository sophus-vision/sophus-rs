use core::borrow::Borrow;

use super::batch_mask::{
    BatchMask,
    BatchMaskF32,
};
use crate::{
    dual::DualBatchVector,
    linalg::{
        BatchMatF32,
        BatchMatF64,
        BatchScalarF32,
        BatchScalarF64,
        BatchVecF32,
        BatchVecF64,
    },
    prelude::*,
};

impl<const ROWS: usize, const BATCH: usize> IsVector<BatchScalarF64<BATCH>, ROWS, BATCH, 0, 0>
    for BatchVecF64<ROWS, BATCH>
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

    fn block_vec3<const R0: usize, const R1: usize, const R2: usize>(
        top_row: BatchVecF64<R0, BATCH>,
        mid_row: BatchVecF64<R1, BATCH>,
        bot_row: BatchVecF64<R2, BATCH>,
    ) -> Self {
        assert_eq!(ROWS, R0 + R1 + R2);
        let mut m = Self::zeros();

        m.fixed_view_mut::<R0, 1>(0, 0).copy_from(&top_row);
        m.fixed_view_mut::<R1, 1>(R0, 0).copy_from(&mid_row);
        m.fixed_view_mut::<R2, 1>(R1, 0).copy_from(&bot_row);
        m
    }

    fn dot<V>(&self, rhs: V) -> BatchScalarF64<BATCH>
    where
        V: Borrow<Self>,
    {
        (self.transpose() * rhs.borrow())[0]
    }

    fn from_array<A>(vals: A) -> Self
    where
        A: Borrow<[BatchScalarF64<BATCH>; ROWS]>,
    {
        let vals = vals.borrow();
        Self::from_fn(|i, _| vals[i])
    }

    fn from_real_array<A>(vals: A) -> Self
    where
        A: Borrow<[BatchScalarF64<BATCH>; ROWS]>,
    {
        let vals = vals.borrow();
        Self::from_fn(|i, _| vals[i])
    }

    fn from_f64_array<A>(vals: A) -> Self
    where
        A: Borrow<[f64; ROWS]>,
    {
        let vals = vals.borrow();
        Self::from_fn(|i, _| BatchScalarF64::<BATCH>::from_f64(vals[i]))
    }

    fn from_scalar_array<A>(vals: A) -> Self
    where
        A: Borrow<[BatchScalarF64<BATCH>; ROWS]>,
    {
        let vals = vals.borrow();
        Self::from_fn(|i, _| vals[i])
    }

    fn from_real_vector<A>(val: A) -> Self
    where
        A: Borrow<BatchVecF64<ROWS, BATCH>>,
    {
        *val.borrow()
    }

    fn elem(&self, idx: usize) -> BatchScalarF64<BATCH> {
        self[idx]
    }

    fn elem_mut(&mut self, idx: usize) -> &mut BatchScalarF64<BATCH> {
        &mut self[idx]
    }

    fn norm(&self) -> BatchScalarF64<BATCH> {
        self.squared_norm().sqrt()
    }

    fn normalized(&self) -> Self {
        let norm = self.norm();
        if norm == <BatchScalarF64<BATCH> as IsScalar<BATCH, 0, 0>>::zeros() {
            return *self;
        }
        let factor = BatchScalarF64::<BATCH>::ones() / norm;
        self * factor
    }

    fn real_vector(&self) -> BatchVecF64<ROWS, BATCH> {
        *self
    }

    fn scaled(&self, v: BatchScalarF64<BATCH>) -> Self {
        self * v
    }

    fn squared_norm(&self) -> BatchScalarF64<BATCH> {
        let mut squared_norm = <BatchScalarF64<BATCH> as IsScalar<BATCH, 0, 0>>::zeros();
        for i in 0..ROWS {
            let val = IsVector::elem(self, i);
            squared_norm += val * val;
        }
        squared_norm
    }

    fn to_mat(&self) -> BatchMatF64<ROWS, 1, BATCH> {
        *self
    }

    fn from_f64(val: f64) -> Self {
        Self::from_element(BatchScalarF64::<BATCH>::from_f64(val))
    }

    fn to_dual_const<const M: usize, const N: usize>(
        &self,
    ) -> <BatchScalarF64<BATCH> as IsScalar<BATCH, 0, 0>>::DualVector<ROWS, M, N> {
        DualBatchVector::from_real_vector(self)
    }

    fn outer<const R2: usize, V>(&self, rhs: V) -> BatchMatF64<ROWS, R2, BATCH>
    where
        V: Borrow<BatchVecF64<R2, BATCH>>,
    {
        self * rhs.borrow().transpose()
    }

    fn select<Q>(&self, mask: &BatchMask<BATCH>, other: Q) -> Self
    where
        Q: Borrow<Self>,
    {
        self.zip_map(other.borrow(), |a, b| a.select(mask, b))
    }

    fn get_fixed_subvec<const R: usize>(&self, start_r: usize) -> BatchVecF64<R, BATCH> {
        self.fixed_rows::<R>(start_r).into()
    }
}

impl<const ROWS: usize, const BATCH: usize> IsRealVector<BatchScalarF64<BATCH>, ROWS, BATCH>
    for BatchVecF64<ROWS, BATCH>
{
}

impl<const ROWS: usize, const BATCH: usize> IsVector<BatchScalarF32<BATCH>, ROWS, BATCH, 0, 0>
    for BatchVecF32<ROWS, BATCH>
{
    fn block_vec2<const R0: usize, const R1: usize>(
        top_row: BatchVecF32<R0, BATCH>,
        bot_row: BatchVecF32<R1, BATCH>,
    ) -> Self {
        assert_eq!(ROWS, R0 + R1);
        let mut m = Self::zeros();
        m.fixed_view_mut::<R0, 1>(0, 0).copy_from(&top_row);
        m.fixed_view_mut::<R1, 1>(R0, 0).copy_from(&bot_row);
        m
    }

    fn block_vec3<const R0: usize, const R1: usize, const R2: usize>(
        top_row: BatchVecF32<R0, BATCH>,
        mid_row: BatchVecF32<R1, BATCH>,
        bot_row: BatchVecF32<R2, BATCH>,
    ) -> Self {
        assert_eq!(ROWS, R0 + R1 + R2);
        let mut m = Self::zeros();
        m.fixed_view_mut::<R0, 1>(0, 0).copy_from(&top_row);
        m.fixed_view_mut::<R1, 1>(R0, 0).copy_from(&mid_row);
        m.fixed_view_mut::<R2, 1>(R1, 0).copy_from(&bot_row);
        m
    }

    fn dot<V>(&self, rhs: V) -> BatchScalarF32<BATCH>
    where
        V: Borrow<Self>,
    {
        (self.transpose() * rhs.borrow())[0]
    }

    fn from_array<A>(vals: A) -> Self
    where
        A: Borrow<[BatchScalarF32<BATCH>; ROWS]>,
    {
        let vals = vals.borrow();
        Self::from_fn(|i, _| vals[i])
    }

    fn from_real_array<A>(vals: A) -> Self
    where
        A: Borrow<[BatchScalarF32<BATCH>; ROWS]>,
    {
        let vals = vals.borrow();
        Self::from_fn(|i, _| vals[i])
    }

    fn from_f64_array<A>(vals: A) -> Self
    where
        A: Borrow<[f64; ROWS]>,
    {
        let vals = vals.borrow();
        Self::from_fn(|i, _| BatchScalarF32::<BATCH>::from_f64(vals[i]))
    }

    fn from_scalar_array<A>(vals: A) -> Self
    where
        A: Borrow<[BatchScalarF32<BATCH>; ROWS]>,
    {
        let vals = vals.borrow();
        Self::from_fn(|i, _| vals[i])
    }

    fn from_real_vector<A>(val: A) -> Self
    where
        A: Borrow<BatchVecF32<ROWS, BATCH>>,
    {
        *val.borrow()
    }

    fn elem(&self, idx: usize) -> BatchScalarF32<BATCH> {
        self[idx]
    }

    fn elem_mut(&mut self, idx: usize) -> &mut BatchScalarF32<BATCH> {
        &mut self[idx]
    }

    fn norm(&self) -> BatchScalarF32<BATCH> {
        self.squared_norm().sqrt()
    }

    fn normalized(&self) -> Self {
        let norm = self.norm();
        if norm == <BatchScalarF32<BATCH> as IsScalar<BATCH, 0, 0>>::zeros() {
            return *self;
        }
        let factor = BatchScalarF32::<BATCH>::ones() / norm;
        self * factor
    }

    fn real_vector(&self) -> BatchVecF32<ROWS, BATCH> {
        *self
    }

    fn scaled(&self, v: BatchScalarF32<BATCH>) -> Self {
        self * v
    }

    fn squared_norm(&self) -> BatchScalarF32<BATCH> {
        let mut squared_norm = <BatchScalarF32<BATCH> as IsScalar<BATCH, 0, 0>>::zeros();
        for i in 0..ROWS {
            let val = IsVector::elem(self, i);
            squared_norm += val * val;
        }
        squared_norm
    }

    fn to_mat(&self) -> BatchMatF32<ROWS, 1, BATCH> {
        *self
    }

    fn from_f64(val: f64) -> Self {
        Self::from_element(BatchScalarF32::<BATCH>::from_f64(val))
    }

    fn to_dual_const<const M: usize, const N: usize>(
        &self,
    ) -> <BatchScalarF32<BATCH> as IsScalar<BATCH, 0, 0>>::DualVector<ROWS, M, N> {
        DualBatchVector::from_real_vector(&self.map(|x| {
            BatchScalarF64::<BATCH>::from_real_array(core::array::from_fn(|i| x.0[i] as f64))
        }))
    }

    fn outer<const R2: usize, V>(&self, rhs: V) -> BatchMatF32<ROWS, R2, BATCH>
    where
        V: Borrow<BatchVecF32<R2, BATCH>>,
    {
        self * rhs.borrow().transpose()
    }

    fn select<Q>(&self, mask: &BatchMaskF32<BATCH>, other: Q) -> Self
    where
        Q: Borrow<Self>,
    {
        self.zip_map(other.borrow(), |a, b| a.select(mask, b))
    }

    fn get_fixed_subvec<const R: usize>(&self, start_r: usize) -> BatchVecF32<R, BATCH> {
        self.fixed_rows::<R>(start_r).into()
    }
}

impl<const ROWS: usize, const BATCH: usize> IsRealVector<BatchScalarF32<BATCH>, ROWS, BATCH>
    for BatchVecF32<ROWS, BATCH>
{
}
