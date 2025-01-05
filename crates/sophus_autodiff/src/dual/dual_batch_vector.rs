use super::vector::HasJacobian;
use super::vector::VectorValuedDerivative;
use crate::dual::dual_batch_scalar::DualBatchScalar;
use crate::dual::DualBatchMatrix;
use crate::linalg::batch_mask::BatchMask;
use crate::linalg::BatchMatF64;
use crate::linalg::BatchScalarF64;
use crate::linalg::BatchVecF64;
use crate::linalg::SMat;
use crate::linalg::SVec;
use crate::prelude::*;
use approx::AbsDiffEq;
use approx::RelativeEq;
use core::borrow::Borrow;
use core::fmt::Debug;
use core::ops::Add;
use core::ops::Neg;
use core::ops::Sub;
use core::simd::LaneCount;
use core::simd::SupportedLaneCount;

/// Dual vector (batch version)
#[derive(Clone, Copy, Debug)]
pub struct DualBatchVector<const ROWS: usize, const BATCH: usize, const DM: usize, const DN: usize>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    pub(crate) inner: SVec<DualBatchScalar<BATCH, DM, DN>, ROWS>,
}

impl<const ROWS: usize, const BATCH: usize, const DM: usize, const DN: usize>
    IsDualVector<DualBatchScalar<BATCH, DM, DN>, ROWS, BATCH, DM, DN>
    for DualBatchVector<ROWS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn var(val: BatchVecF64<ROWS, BATCH>) -> Self {
        let mut dij_val: SVec<DualBatchScalar<BATCH, DM, DN>, ROWS> = SVec::zeros();
        for i in 0..ROWS {
            dij_val[(i, 0)].real_part = val[i];
            let mut v = SMat::<BatchScalarF64<BATCH>, DM, DN>::zeros();
            v[(i, 0)] = BatchScalarF64::<BATCH>::from_f64(1.0);
            dij_val[(i, 0)].infinitesimal_part = Some(v);
        }

        Self { inner: dij_val }
    }

    fn derivative(&self) -> VectorValuedDerivative<BatchScalarF64<BATCH>, ROWS, BATCH, DM, DN> {
        let mut v = SVec::<SMat<BatchScalarF64<BATCH>, DM, DN>, ROWS>::zeros();
        for i in 0..ROWS {
            v[i] = self.inner[i].derivative();
        }
        VectorValuedDerivative { out_vec: v }
    }
}

impl<const ROWS: usize, const BATCH: usize, const DM: usize>
    HasJacobian<DualBatchScalar<BATCH, DM, 1>, ROWS, BATCH, DM>
    for DualBatchVector<ROWS, BATCH, DM, 1>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn jacobian(&self) -> BatchMatF64<ROWS, DM, BATCH> {
        let mut v = BatchMatF64::<ROWS, DM, BATCH>::zeros();
        for i in 0..ROWS {
            let d = self.inner[i].derivative();
            for j in 0..DM {
                v[(i, j)] = d[j];
            }
        }
        v
    }
}

impl<const ROWS: usize, const BATCH: usize, const DM: usize, const DN: usize> num_traits::Zero
    for DualBatchVector<ROWS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn zero() -> Self {
        DualBatchVector {
            inner: SVec::zeros(),
        }
    }

    fn is_zero(&self) -> bool {
        self.inner == Self::zero().inner
    }
}

impl<const ROWS: usize, const BATCH: usize, const DM: usize, const DN: usize> Neg
    for DualBatchVector<ROWS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchVector<ROWS, BATCH, DM, DN>;

    fn neg(self) -> Self::Output {
        DualBatchVector { inner: -self.inner }
    }
}

impl<const ROWS: usize, const BATCH: usize, const DM: usize, const DN: usize> Sub
    for DualBatchVector<ROWS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchVector<ROWS, BATCH, DM, DN>;

    fn sub(self, rhs: Self) -> Self::Output {
        DualBatchVector {
            inner: self.inner - rhs.inner,
        }
    }
}

impl<const ROWS: usize, const BATCH: usize, const DM: usize, const DN: usize> Add
    for DualBatchVector<ROWS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Output = DualBatchVector<ROWS, BATCH, DM, DN>;

    fn add(self, rhs: Self) -> Self::Output {
        DualBatchVector {
            inner: self.inner + rhs.inner,
        }
    }
}

impl<const ROWS: usize, const BATCH: usize, const DM: usize, const DN: usize> PartialEq
    for DualBatchVector<ROWS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<const ROWS: usize, const BATCH: usize, const DM: usize, const DN: usize> AbsDiffEq
    for DualBatchVector<ROWS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.inner.abs_diff_eq(&other.inner, epsilon)
    }
}

impl<const ROWS: usize, const BATCH: usize, const DM: usize, const DN: usize> RelativeEq
    for DualBatchVector<ROWS, BATCH, DM, DN>
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
        self.inner.relative_eq(&other.inner, epsilon, max_relative)
    }
}

impl<const ROWS: usize, const BATCH: usize, const DM: usize, const DN: usize>
    IsVector<DualBatchScalar<BATCH, DM, DN>, ROWS, BATCH, DM, DN>
    for DualBatchVector<ROWS, BATCH, DM, DN>
where
    BatchScalarF64<BATCH>: IsCoreScalar,
    LaneCount<BATCH>: SupportedLaneCount,
{
    fn from_f64(val: f64) -> Self {
        DualBatchVector {
            inner: SVec::<DualBatchScalar<BATCH, DM, DN>, ROWS>::from_element(
                DualBatchScalar::from_f64(val),
            ),
        }
    }

    fn norm(&self) -> DualBatchScalar<BATCH, DM, DN> {
        self.clone().dot(*self).sqrt()
    }

    fn squared_norm(&self) -> DualBatchScalar<BATCH, DM, DN> {
        self.clone().dot(*self)
    }

    fn get_elem(&self, idx: usize) -> DualBatchScalar<BATCH, DM, DN> {
        self.inner[idx]
    }

    fn from_array<A>(duals: A) -> Self
    where
        A: Borrow<[DualBatchScalar<BATCH, DM, DN>; ROWS]>,
    {
        DualBatchVector {
            inner: SVec::<DualBatchScalar<BATCH, DM, DN>, ROWS>::from_row_slice(
                &duals.borrow()[..],
            ),
        }
    }

    fn from_real_array<A>(vals: A) -> Self
    where
        A: Borrow<[BatchScalarF64<BATCH>; ROWS]>,
    {
        let vals = vals.borrow();

        let mut out = Self::zeros();
        for i in 0..ROWS {
            out.inner[i].real_part = vals[i];
        }
        out
    }

    fn from_real_vector<A>(val: A) -> Self
    where
        A: Borrow<BatchVecF64<ROWS, BATCH>>,
    {
        let vals = val.borrow();

        let mut out = Self::zeros();
        for i in 0..ROWS {
            out.inner[i].real_part = vals[i];
        }
        out
    }

    fn real_vector(&self) -> BatchVecF64<ROWS, BATCH> {
        let mut r = BatchVecF64::<ROWS, BATCH>::zeros();
        for i in 0..ROWS {
            r[i] = self.get_elem(i).real_part;
        }
        r
    }

    fn to_mat(&self) -> DualBatchMatrix<ROWS, 1, BATCH, DM, DN> {
        DualBatchMatrix { inner: self.inner }
    }

    fn block_vec2<const R0: usize, const R1: usize>(
        top_row: DualBatchVector<R0, BATCH, DM, DN>,
        bot_row: DualBatchVector<R1, BATCH, DM, DN>,
    ) -> Self {
        assert_eq!(R0 + R1, ROWS);

        assert_eq!(ROWS, R0 + R1);
        let mut m = Self::zeros();

        m.inner
            .fixed_view_mut::<R0, 1>(0, 0)
            .copy_from(&top_row.inner);
        m.inner
            .fixed_view_mut::<R1, 1>(R0, 0)
            .copy_from(&bot_row.inner);
        m
    }

    fn scaled<U>(&self, s: U) -> Self
    where
        U: Borrow<DualBatchScalar<BATCH, DM, DN>>,
    {
        Self {
            inner: self.inner * *s.borrow(),
        }
    }

    fn dot<V>(&self, rhs: V) -> DualBatchScalar<BATCH, DM, DN>
    where
        V: Borrow<Self>,
    {
        let mut sum = <DualBatchScalar<BATCH, DM, DN>>::from_f64(0.0);

        for i in 0..ROWS {
            sum += self.get_elem(i) * rhs.borrow().get_elem(i);
        }

        sum
    }

    fn normalized(&self) -> Self {
        self.clone()
            .scaled(<DualBatchScalar<BATCH, DM, DN>>::from_f64(1.0) / self.norm())
    }

    fn from_f64_array<A>(vals: A) -> Self
    where
        A: Borrow<[f64; ROWS]>,
    {
        let vals = vals.borrow();

        let mut out = Self::zeros();
        for i in 0..ROWS {
            out.inner[i].real_part = BatchScalarF64::from_f64(vals[i]);
        }
        out
    }

    fn from_scalar_array<A>(vals: A) -> Self
    where
        A: Borrow<[DualBatchScalar<BATCH, DM, DN>; ROWS]>,
    {
        DualBatchVector {
            inner: SVec::<DualBatchScalar<BATCH, DM, DN>, ROWS>::from_row_slice(&vals.borrow()[..]),
        }
    }

    fn set_elem(&mut self, idx: usize, v: DualBatchScalar<BATCH, DM, DN>) {
        self.inner[idx] = v;
    }

    fn to_dual_const<const M: usize, const N: usize>(
        &self,
    ) -> <DualBatchScalar<BATCH, DM, DN> as IsScalar<BATCH, DM, DN>>::DualVector<ROWS, M, N> {
        DualBatchVector::<ROWS, BATCH, M, N>::from_real_vector(self.real_vector())
    }

    fn outer<const R2: usize, V>(
        &self,
        rhs: V,
    ) -> <DualBatchScalar<BATCH, DM, DN> as IsScalar<BATCH, DM, DN>>::Matrix<ROWS, R2>
    where
        V: Borrow<DualBatchVector<R2, BATCH, DM, DN>>,
    {
        let mut out = DualBatchMatrix::<ROWS, R2, BATCH, DM, DN>::zeros();
        for i in 0..ROWS {
            for j in 0..R2 {
                out.set_elem([i, j], self.get_elem(i) * rhs.borrow().get_elem(j));
            }
        }
        out
    }

    fn select<Q>(&self, mask: &BatchMask<BATCH>, other: Q) -> Self
    where
        Q: Borrow<Self>,
    {
        let mut v = SVec::<DualBatchScalar<BATCH, DM, DN>, ROWS>::zeros();
        let other = other.borrow();
        for i in 0..ROWS {
            v[i] = self.get_elem(i).select(mask, other.get_elem(i));
        }

        Self { inner: v }
    }

    fn get_fixed_subvec<const R: usize>(
        &self,
        start_r: usize,
    ) -> DualBatchVector<R, BATCH, DM, DN> {
        DualBatchVector {
            inner: self.inner.fixed_rows::<R>(start_r).into(),
        }
    }

    fn ones() -> Self {
        Self::from_f64(1.0)
    }

    fn zeros() -> Self {
        Self::from_f64(0.0)
    }
}
