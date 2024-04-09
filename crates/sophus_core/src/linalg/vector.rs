use approx::AbsDiffEq;
use approx::RelativeEq;

use super::scalar::IsRealScalar;
use super::scalar::IsScalar;
use super::scalar::IsSingleScalar;
use crate::calculus::dual::dual_vector::DualBatchVector;
use crate::calculus::dual::dual_vector::DualVector;
use crate::linalg::BatchMatF64;
use crate::linalg::BatchScalarF64;
use crate::linalg::BatchVecF64;
use crate::linalg::MatF64;
use crate::linalg::VecF64;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Index;
use std::ops::IndexMut;
use std::ops::Neg;
use std::ops::Sub;
use std::simd::LaneCount;
use std::simd::Mask;
use std::simd::SupportedLaneCount;

/// Vector - either a real (f64) or a dual number vector
pub trait IsVector<S: IsScalar<BATCH_SIZE>, const ROWS: usize, const BATCH_SIZE: usize>:
    Clone
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg
    + Debug
    + AbsDiffEq<Epsilon = f64>
    + RelativeEq<Epsilon = f64>
{
    fn vector(self) -> Self;

    /// create a block vector
    fn block_vec2<const R0: usize, const R1: usize>(
        top_row: S::Vector<R0>,
        bot_row: S::Vector<R1>,
    ) -> Self;

    fn to_dual(
        self,
    ) -> <<S as IsScalar<BATCH_SIZE>>::DualScalar as IsScalar<BATCH_SIZE>>::Vector<ROWS>;

    /// dot product
    fn dot(self, rhs: Self) -> S;

    fn outer<const R2: usize>(self, rhs: S::Vector<R2>) -> S::Matrix<ROWS, R2>;

    /// create a vector from an array
    fn from_array(vals: [S; ROWS]) -> Self;

    fn select(self, mask: &S::Mask, other: Self) -> Self;

    /// create a constant vector from an array
    fn from_real_array(vals: [S::RealScalar; ROWS]) -> Self;

    /// create a constant vector
    fn from_real_vector(val: S::RealVector<ROWS>) -> Self;

    /// create a constant scalar
    fn from_f64(val: f64) -> Self;

    /// create a constant vector from an array
    fn from_f64_array(vals: [f64; ROWS]) -> Self;

    /// create a constant vector from an array
    fn from_scalar_array(vals: [S; ROWS]) -> Self;

    /// get ith element
    fn get_elem(&self, idx: usize) -> S;

    /// get fixed rows
    fn get_fixed_rows<const R: usize>(&self, start: usize) -> S::Vector<R>;

    /// norm
    fn norm(&self) -> S;

    /// return normalized vector
    fn normalized(&self) -> Self;

    /// return the real part
    fn real_vector(&self) -> &S::RealVector<ROWS>;

    /// return scaled vector
    fn scaled(&self, v: S) -> Self;

    /// set ith element as constant
    fn set_elem(&mut self, idx: usize, v: S);

    /// set ith element as constant
    fn set_real_elem(&mut self, idx: usize, v: S::RealScalar);

    /// squared norm
    fn squared_norm(&self) -> S;

    /// return the matrix representation
    fn to_mat(self) -> S::Matrix<ROWS, 1>;

    /// ones
    fn ones() -> Self {
        Self::from_f64(1.0)
    }

    /// zeros
    fn zeros() -> Self {
        Self::from_f64(0.0)
    }

    /// get fixed submatrix
    fn get_fixed_subvec<const R: usize>(&self, start_r: usize) -> S::Vector<R>;
}

/// is real vector like
pub trait IsRealVector<
    S: IsRealScalar<BATCH_SIZE> + IsScalar<BATCH_SIZE>,
    const ROWS: usize,
    const BATCH_SIZE: usize,
>:
    IsVector<S, ROWS, BATCH_SIZE> + Index<usize, Output = S> + IndexMut<usize, Output = S> + Copy
{
}

/// Batch scalar
pub trait IsBatchVector<const ROWS: usize, const BATCH_SIZE: usize>: IsScalar<BATCH_SIZE> {
    /// get item
    fn extract_single(&self, i: usize) -> Self::SingleScalar;
}

/// is scalar vector
pub trait IsSingleVector<S: IsSingleScalar, const ROWS: usize>: IsVector<S, ROWS, 1> {
    /// set real scalar
    fn set_real_scalar(&mut self, idx: usize, v: f64);
}

impl<const BATCH: usize> IsSingleVector<f64, BATCH> for VecF64<BATCH> {
    fn set_real_scalar(&mut self, idx: usize, v: f64) {
        self[idx] = v;
    }
}

impl<const ROWS: usize> IsRealVector<f64, ROWS, 1> for VecF64<ROWS> {}

impl<const ROWS: usize> IsVector<f64, ROWS, 1> for VecF64<ROWS> {
    fn vector(self) -> Self {
        self
    }

    fn block_vec2<const R0: usize, const R1: usize>(
        top_row: VecF64<R0>,
        bot_row: VecF64<R1>,
    ) -> Self {
        assert_eq!(ROWS, R0 + R1);
        let mut m = Self::zeros();

        m.fixed_view_mut::<R0, 1>(0, 0).copy_from(&top_row);
        m.fixed_view_mut::<R1, 1>(R0, 0).copy_from(&bot_row);
        m
    }

    fn from_array(vals: [f64; ROWS]) -> VecF64<ROWS> {
        VecF64::<ROWS>::from_row_slice(&vals[..])
    }

    fn from_real_array(vals: [f64; ROWS]) -> Self {
        VecF64::<ROWS>::from_row_slice(&vals[..])
    }

    fn from_real_vector(val: VecF64<ROWS>) -> Self {
        val
    }

    fn from_f64_array(vals: [f64; ROWS]) -> Self {
        VecF64::<ROWS>::from_row_slice(&vals[..])
    }

    fn from_scalar_array(vals: [f64; ROWS]) -> Self {
        VecF64::<ROWS>::from_row_slice(&vals[..])
    }

    fn get_elem(&self, idx: usize) -> f64 {
        self[idx]
    }

    fn get_fixed_rows<const R: usize>(&self, start: usize) -> VecF64<R> {
        self.fixed_rows::<R>(start).into()
    }

    fn norm(&self) -> f64 {
        self.norm()
    }

    fn real_vector(&self) -> &Self {
        self
    }

    fn set_elem(&mut self, idx: usize, v: f64) {
        self[idx] = v;
    }

    fn set_real_elem(&mut self, idx: usize, v: f64) {
        self[idx] = v;
    }

    fn squared_norm(&self) -> f64 {
        self.norm_squared()
    }

    fn to_mat(self) -> MatF64<ROWS, 1> {
        self
    }

    fn scaled(&self, v: f64) -> Self {
        self * v
    }

    fn dot(self, rhs: Self) -> f64 {
        VecF64::dot(&self, &rhs)
    }

    fn normalized(&self) -> Self {
        self.normalize()
    }

    fn from_f64(val: f64) -> Self {
        VecF64::<ROWS>::from_element(val)
    }

    fn to_dual(self) -> <f64 as IsScalar<1>>::DualVector<ROWS> {
        DualVector::from_real_vector(self)
    }

    fn outer<const R2: usize>(self, rhs: VecF64<R2>) -> MatF64<ROWS, R2> {
        self * rhs.transpose()
    }

    fn select(self, mask: &bool, other: Self) -> Self {
        if *mask {
            self
        } else {
            other
        }
    }

    fn get_fixed_subvec<const R: usize>(&self, start_r: usize) -> VecF64<R> {
        self.fixed_rows::<R>(start_r).into()
    }
}

/// cross product
pub fn cross<S: IsScalar<BATCH>, const BATCH: usize>(
    lhs: S::Vector<3>,
    rhs: S::Vector<3>,
) -> S::Vector<3> {
    let l0 = lhs.get_elem(0);
    let l1 = lhs.get_elem(1);
    let l2 = lhs.get_elem(2);

    let r0 = rhs.get_elem(0);
    let r1 = rhs.get_elem(1);
    let r2 = rhs.get_elem(2);

    S::Vector::from_array([
        l1.clone() * r2.clone() - l2.clone() * r1.clone(),
        l2 * r0.clone() - l0.clone() * r2,
        l0 * r1 - l1 * r0,
    ])
}

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

    fn vector(self) -> Self {
        self
    }

    fn dot(self, rhs: Self) -> BatchScalarF64<BATCH> {
        (self.transpose() * &rhs)[0]
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

    fn get_fixed_rows<const R: usize>(&self, start: usize) -> BatchVecF64<R, BATCH> {
        self.fixed_rows::<R>(start).into()
    }

    fn norm(&self) -> BatchScalarF64<BATCH> {
        self.squared_norm().sqrt()
    }

    fn normalized(&self) -> Self {
        let norm = self.norm();
        if norm == BatchScalarF64::<BATCH>::zeros() {
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

    fn set_real_elem(&mut self, idx: usize, v: BatchScalarF64<BATCH>) {
        self[idx] = v;
    }

    fn squared_norm(&self) -> BatchScalarF64<BATCH> {
        let mut squared_norm = BatchScalarF64::<BATCH>::zeros();
        for i in 0..ROWS {
            let val = self.get_elem(i);
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
