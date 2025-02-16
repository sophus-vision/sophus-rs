use core::{
    borrow::Borrow,
    fmt::Debug,
    ops::{
        Add,
        Neg,
        Sub,
    },
};

use approx::{
    AbsDiffEq,
    RelativeEq,
};

use super::{
    dual_matrix::DualMatrix,
    dual_scalar::DualScalar,
    vector::{
        HasJacobian,
        IsDualVectorFromCurve,
        VectorValuedDerivative,
    },
};
use crate::{
    linalg::{
        MatF64,
        SMat,
        SVec,
        VecF64,
    },
    prelude::*,
};

/// A dual vector, storing a set of dual scalars (with partial derivatives) for each row.
///
/// Conceptually, this is the forward-mode AD version of a vector in \(\mathbb{R}^\text{ROWS}\),
/// where each element is a [`DualScalar<DM, DN>`], i.e., each element carries its own infinitesimal
/// part.
///
/// # Fields
/// - `inner`: An [`SVec`] of [`DualScalar<DM, DN>`], dimension `ROWS`.
///
/// # Generic Parameters
/// - `ROWS`: Number of vector components.
/// - `DM`, `DN`: Dimensions for each component’s derivative (infinitesimal) matrix. For example,
///   `DM=3, DN=1` might store partial derivatives w.r.t. a 3D input for each element.
///
/// See [crate::dual::IsDualVector] for more details.
#[derive(Clone, Debug, Copy)]
pub struct DualVector<const ROWS: usize, const DM: usize, const DN: usize> {
    /// Internal storage of the vector elements, each a `DualScalar<DM, DN>`.
    pub(crate) inner: SVec<DualScalar<DM, DN>, ROWS>,
}

impl<const ROWS: usize, const DM: usize, const DN: usize>
    IsDualVector<DualScalar<DM, DN>, ROWS, 1, DM, DN> for DualVector<ROWS, DM, DN>
{
    fn var(val: VecF64<ROWS>) -> Self {
        let mut dij_val: SVec<DualScalar<DM, DN>, ROWS> = SVec::zeros();
        for i in 0..ROWS {
            dij_val[(i, 0)].real_part = val[i];
            let mut v = SMat::<f64, DM, DN>::zeros();
            // identity derivative for the i-th element w.r.t. that row
            v[(i, 0)] = 1.0;
            dij_val[(i, 0)].infinitesimal_part = Some(v);
        }
        Self { inner: dij_val }
    }

    fn derivative(&self) -> VectorValuedDerivative<f64, ROWS, 1, DM, DN> {
        let mut v = SVec::<SMat<f64, DM, DN>, ROWS>::zeros();
        for i in 0..ROWS {
            v[i] = self.inner[i].derivative();
        }
        VectorValuedDerivative { out_vec: v }
    }
}

impl<const ROWS: usize> IsDualVectorFromCurve<DualScalar<1, 1>, ROWS, 1>
    for DualVector<ROWS, 1, 1>
{
    fn curve_derivative(&self) -> VecF64<ROWS> {
        self.jacobian()
    }
}

impl<const ROWS: usize, const DM: usize> HasJacobian<DualScalar<DM, 1>, ROWS, 1, DM>
    for DualVector<ROWS, DM, 1>
{
    fn jacobian(&self) -> MatF64<ROWS, DM> {
        let mut v = SMat::<f64, ROWS, DM>::zeros();
        for i in 0..ROWS {
            let d = self.inner[i].derivative();
            // each element’s derivative is DM x 1 => we collect the column into row i
            for j in 0..DM {
                v[(i, j)] = d[j];
            }
        }
        v
    }
}

impl<const ROWS: usize, const DM: usize, const DN: usize> num_traits::Zero
    for DualVector<ROWS, DM, DN>
{
    fn zero() -> Self {
        DualVector {
            inner: SVec::zeros(),
        }
    }

    fn is_zero(&self) -> bool {
        self.inner == Self::zero().inner
    }
}

impl<const ROWS: usize, const DM: usize, const DN: usize>
    IsSingleVector<DualScalar<DM, DN>, ROWS, DM, DN> for DualVector<ROWS, DM, DN>
where
    DualVector<ROWS, DM, DN>: IsVector<DualScalar<DM, DN>, ROWS, 1, DM, DN>,
{
    fn set_real_scalar(&mut self, idx: usize, v: f64) {
        self.inner[idx].real_part = v;
    }
}

impl<const ROWS: usize, const DM: usize, const DN: usize> Neg for DualVector<ROWS, DM, DN> {
    type Output = DualVector<ROWS, DM, DN>;

    fn neg(self) -> Self::Output {
        DualVector { inner: -self.inner }
    }
}

impl<const ROWS: usize, const DM: usize, const DN: usize> Sub for DualVector<ROWS, DM, DN> {
    type Output = DualVector<ROWS, DM, DN>;

    fn sub(self, rhs: Self) -> Self::Output {
        DualVector {
            inner: self.inner - rhs.inner,
        }
    }
}

impl<const ROWS: usize, const DM: usize, const DN: usize> Add for DualVector<ROWS, DM, DN> {
    type Output = DualVector<ROWS, DM, DN>;

    fn add(self, rhs: Self) -> Self::Output {
        DualVector {
            inner: self.inner + rhs.inner,
        }
    }
}

impl<const ROWS: usize, const DM: usize, const DN: usize> PartialEq for DualVector<ROWS, DM, DN> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<const ROWS: usize, const DM: usize, const DN: usize> AbsDiffEq for DualVector<ROWS, DM, DN> {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.inner.abs_diff_eq(&other.inner, epsilon)
    }
}

impl<const ROWS: usize, const DM: usize, const DN: usize> RelativeEq for DualVector<ROWS, DM, DN> {
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

impl<const ROWS: usize, const DM: usize, const DN: usize>
    IsVector<DualScalar<DM, DN>, ROWS, 1, DM, DN> for DualVector<ROWS, DM, DN>
{
    fn from_f64(val: f64) -> Self {
        DualVector {
            inner: SVec::<DualScalar<DM, DN>, ROWS>::from_element(DualScalar::from_f64(val)),
        }
    }

    fn norm(&self) -> DualScalar<DM, DN> {
        // accumulate dot = x^T x, then sqrt
        self.dot(*self).sqrt()
    }

    fn squared_norm(&self) -> DualScalar<DM, DN> {
        self.dot(*self)
    }

    fn elem(&self, idx: usize) -> DualScalar<DM, DN> {
        self.inner[idx]
    }

    fn from_array<A>(duals: A) -> Self
    where
        A: Borrow<[DualScalar<DM, DN>; ROWS]>,
    {
        DualVector {
            inner: SVec::<DualScalar<DM, DN>, ROWS>::from_row_slice(&duals.borrow()[..]),
        }
    }

    fn from_real_array<A>(vals: A) -> Self
    where
        A: Borrow<[f64; ROWS]>,
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
        A: Borrow<VecF64<ROWS>>,
    {
        let vals = val.borrow();
        let mut out = Self::zeros();
        for i in 0..ROWS {
            out.inner[i].real_part = vals[i];
        }
        out
    }

    fn real_vector(&self) -> VecF64<ROWS> {
        let mut r = VecF64::<ROWS>::zeros();
        for i in 0..ROWS {
            r[i] = self.elem(i).real_part;
        }
        r
    }

    fn to_mat(&self) -> DualMatrix<ROWS, 1, DM, DN> {
        DualMatrix { inner: self.inner }
    }

    fn block_vec2<const R0: usize, const R1: usize>(
        top_row: DualVector<R0, DM, DN>,
        bot_row: DualVector<R1, DM, DN>,
    ) -> Self {
        assert_eq!(R0 + R1, ROWS);
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
        U: Borrow<DualScalar<DM, DN>>,
    {
        // scale each element by s
        Self {
            inner: self.inner * *s.borrow(),
        }
    }

    fn dot<V>(&self, rhs: V) -> DualScalar<DM, DN>
    where
        V: Borrow<Self>,
    {
        // sum_{i} self[i] * rhs[i]
        let mut sum = DualScalar::<DM, DN>::from_f64(0.0);
        for i in 0..ROWS {
            sum += self.elem(i) * rhs.borrow().elem(i);
        }
        sum
    }

    fn normalized(&self) -> Self {
        self.scaled(DualScalar::<DM, DN>::from_f64(1.0) / self.norm())
    }

    fn from_f64_array<A>(vals: A) -> Self
    where
        A: Borrow<[f64; ROWS]>,
    {
        let vals = vals.borrow();
        let mut out = Self::zeros();
        for i in 0..ROWS {
            out.inner[i].real_part = vals[i];
        }
        out
    }

    fn from_scalar_array<A>(vals: A) -> Self
    where
        A: Borrow<[DualScalar<DM, DN>; ROWS]>,
    {
        DualVector {
            inner: SVec::<DualScalar<DM, DN>, ROWS>::from_row_slice(&vals.borrow()[..]),
        }
    }

    fn elem_mut(&mut self, idx: usize) -> &mut DualScalar<DM, DN> {
        &mut self.inner[idx]
    }

    fn to_dual_const<const M: usize, const N: usize>(
        &self,
    ) -> <DualScalar<DM, DN> as IsScalar<1, DM, DN>>::DualVector<ROWS, M, N> {
        // Discard derivative part, just copy real parts.
        DualVector::<ROWS, M, N>::from_real_vector(self.real_vector())
    }

    fn outer<const R2: usize, V>(
        &self,
        rhs: V,
    ) -> <DualScalar<DM, DN> as IsScalar<1, DM, DN>>::Matrix<ROWS, R2>
    where
        V: Borrow<DualVector<R2, DM, DN>>,
    {
        // build a dual matrix with each entry = self[i] * rhs[j]
        let mut out = DualMatrix::<ROWS, R2, DM, DN>::zeros();
        for i in 0..ROWS {
            for j in 0..R2 {
                *out.elem_mut([i, j]) = self.elem(i) * rhs.borrow().elem(j);
            }
        }
        out
    }

    fn select<Q>(&self, mask: &bool, other: Q) -> Self
    where
        Q: Borrow<Self>,
    {
        // single-lane bool => entire vector is chosen or not
        if *mask {
            *self
        } else {
            *other.borrow()
        }
    }

    fn get_fixed_subvec<const R: usize>(&self, start_r: usize) -> DualVector<R, DM, DN> {
        // sub-slice from row start_r to row start_r+R
        DualVector {
            inner: self.inner.fixed_rows::<R>(start_r).into(),
        }
    }
}
