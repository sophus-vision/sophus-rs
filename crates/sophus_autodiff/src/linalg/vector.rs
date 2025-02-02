use core::{
    borrow::Borrow,
    fmt::Debug,
    ops::{
        Add,
        Index,
        IndexMut,
        Neg,
        Sub,
    },
};

use approx::{
    AbsDiffEq,
    RelativeEq,
};

use crate::{
    dual::dual_vector::DualVector,
    linalg::{
        MatF64,
        VecF64,
    },
    prelude::*,
};

/// Vector - either a real (f64) or a dual number vector
pub trait IsVector<
    S: IsScalar<BATCH, DM, DN>,
    const ROWS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>:
    Clone
    + Copy
    + Neg<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg
    + Debug
    + AbsDiffEq<Epsilon = f64>
    + RelativeEq<Epsilon = f64>
{
    /// creates vector from a block of two vectors
    fn block_vec2<const R0: usize, const R1: usize>(
        top_row: S::Vector<R0>,
        bot_row: S::Vector<R1>,
    ) -> Self;

    /// dot product
    fn dot<V>(&self, rhs: V) -> S
    where
        V: Borrow<Self>;

    /// create a vector from an array
    fn from_array<A>(vals: A) -> Self
    where
        A: Borrow<[S; ROWS]>;

    /// create a constant vector from an array
    fn from_real_array<A>(vals: A) -> Self
    where
        A: Borrow<[S::RealScalar; ROWS]>;

    /// create a constant vector
    fn from_real_vector<A>(val: A) -> Self
    where
        A: Borrow<S::RealVector<ROWS>>;

    /// create a constant scalar
    fn from_f64(val: f64) -> Self;

    /// create a constant vector from an array
    fn from_f64_array<A>(vals: A) -> Self
    where
        A: Borrow<[f64; ROWS]>;

    /// get ith element
    fn elem(&self, idx: usize) -> S;

    /// get ith element
    fn elem_mut(&mut self, idx: usize) -> &mut S;

    /// Returns a fixed-size subvector starting at the given row
    fn get_fixed_subvec<const R: usize>(&self, start_r: usize) -> S::Vector<R>;

    /// create a constant vector from an array
    fn from_scalar_array<A>(vals: A) -> Self
    where
        A: Borrow<[S; ROWS]>;

    /// norm
    fn norm(&self) -> S;

    /// return normalized vector
    fn normalized(&self) -> Self;

    /// outer product
    fn outer<const R2: usize, V>(&self, rhs: V) -> S::Matrix<ROWS, R2>
    where
        V: Borrow<S::Vector<R2>>;

    /// return the real part
    fn real_vector(&self) -> S::RealVector<ROWS>;

    /// Returns self if mask is true, otherwise returns other
    ///
    /// For batch vectors, this is a lane-wise operation
    fn select<O>(&self, mask: &S::Mask, other: O) -> Self
    where
        O: Borrow<Self>;

    /// return scaled vector
    fn scaled<U>(&self, v: U) -> Self
    where
        U: Borrow<S>;

    /// squared norm
    fn squared_norm(&self) -> S;

    /// Return dual vector
    ///
    /// If self is a real vector, this will return a dual vector with the infinitesimal part set to
    /// zero: (self, 0ϵ)
    fn to_dual_const<const M: usize, const N: usize>(
        &self,
    ) -> <<S as IsScalar<BATCH, DM, DN>>::DualScalar<M, N> as IsScalar<BATCH, M, N>>::Vector<ROWS>;

    /// return the matrix representation - in self as a column vector
    fn to_mat(&self) -> S::Matrix<ROWS, 1>;

    /// ones
    fn ones() -> Self {
        Self::from_f64(1.0)
    }

    /// zeros
    fn zeros() -> Self {
        Self::from_f64(0.0)
    }
}

/// is real vector like
pub trait IsRealVector<
    S: IsRealScalar<BATCH> + IsScalar<BATCH, 0, 0>,
    const ROWS: usize,
    const BATCH: usize,
>:
    IsVector<S, ROWS, BATCH, 0, 0> + Index<usize, Output = S> + IndexMut<usize, Output = S> + Copy
{
}

/// Batch scalar
pub trait IsBatchVector<const ROWS: usize, const BATCH: usize, const DM: usize, const DN: usize>:
    IsScalar<BATCH, DM, DN>
{
    /// get item
    fn extract_single(&self, i: usize) -> Self::SingleScalar;
}

/// is scalar vector
pub trait IsSingleVector<
    S: IsSingleScalar<DM, DN>,
    const ROWS: usize,
    const DM: usize,
    const DN: usize,
>: IsVector<S, ROWS, 1, DM, DN>
{
    /// set real scalar
    fn set_real_scalar(&mut self, idx: usize, v: f64);
}

impl<const BATCH: usize> IsSingleVector<f64, BATCH, 0, 0> for VecF64<BATCH> {
    fn set_real_scalar(&mut self, idx: usize, v: f64) {
        self[idx] = v;
    }
}

impl<const ROWS: usize> IsRealVector<f64, ROWS, 1> for VecF64<ROWS> {}

impl<const ROWS: usize> IsVector<f64, ROWS, 1, 0, 0> for VecF64<ROWS> {
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

    fn from_array<A>(vals: A) -> VecF64<ROWS>
    where
        A: Borrow<[f64; ROWS]>,
    {
        VecF64::<ROWS>::from_row_slice(&vals.borrow()[..])
    }

    fn from_real_array<A>(vals: A) -> Self
    where
        A: Borrow<[f64; ROWS]>,
    {
        VecF64::<ROWS>::from_row_slice(&vals.borrow()[..])
    }

    fn from_real_vector<A>(val: A) -> Self
    where
        A: Borrow<VecF64<ROWS>>,
    {
        *val.borrow()
    }

    fn from_f64_array<A>(vals: A) -> Self
    where
        A: Borrow<[f64; ROWS]>,
    {
        VecF64::<ROWS>::from_row_slice(&vals.borrow()[..])
    }

    fn from_scalar_array<A>(vals: A) -> Self
    where
        A: Borrow<[f64; ROWS]>,
    {
        VecF64::<ROWS>::from_row_slice(&vals.borrow()[..])
    }

    fn elem(&self, idx: usize) -> f64 {
        self[idx]
    }

    fn elem_mut(&mut self, idx: usize) -> &mut f64 {
        &mut self[idx]
    }

    fn norm(&self) -> f64 {
        self.norm()
    }

    fn real_vector(&self) -> Self {
        *self
    }

    fn squared_norm(&self) -> f64 {
        self.norm_squared()
    }

    fn to_mat(&self) -> MatF64<ROWS, 1> {
        *self
    }

    fn scaled<U>(&self, v: U) -> Self
    where
        U: Borrow<f64>,
    {
        self * *v.borrow()
    }

    fn dot<V>(&self, rhs: V) -> f64
    where
        V: Borrow<Self>,
    {
        VecF64::dot(self, rhs.borrow())
    }

    fn normalized(&self) -> Self {
        self.normalize()
    }

    fn from_f64(val: f64) -> Self {
        VecF64::<ROWS>::from_element(val)
    }

    fn to_dual_const<const M: usize, const N: usize>(
        &self,
    ) -> <f64 as IsScalar<1, 0, 0>>::DualVector<ROWS, M, N> {
        DualVector::from_real_vector(*self)
    }

    fn outer<const R2: usize, V>(&self, rhs: V) -> MatF64<ROWS, R2>
    where
        V: Borrow<VecF64<R2>>,
    {
        self * rhs.borrow().transpose()
    }

    fn select<Q>(&self, mask: &bool, other: Q) -> Self
    where
        Q: Borrow<Self>,
    {
        if *mask {
            *self
        } else {
            *other.borrow()
        }
    }

    fn get_fixed_subvec<const R: usize>(&self, start_r: usize) -> VecF64<R> {
        self.fixed_rows::<R>(start_r).into()
    }
}

/// cross product
pub fn cross<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>(
    lhs: &S::Vector<3>,
    rhs: &S::Vector<3>,
) -> S::Vector<3> {
    let l0 = lhs.elem(0);
    let l1 = lhs.elem(1);
    let l2 = lhs.elem(2);

    let r0 = rhs.elem(0);
    let r1 = rhs.elem(1);
    let r2 = rhs.elem(2);

    S::Vector::from_array([l1 * r2 - l2 * r1, l2 * r0 - l0 * r2, l0 * r1 - l1 * r0])
}
