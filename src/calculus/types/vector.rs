use std::fmt::Debug;
use std::ops::Add;
use std::ops::Neg;
use std::ops::Sub;

use super::scalar::IsScalar;

use super::M;
use super::V;

pub trait IsVectorLike:
    Debug + Clone + Sized + Neg<Output = Self> + Add<Output = Self> + Sub<Output = Self>
{
    fn zero() -> Self;
}

pub trait IsVector<S: IsScalar, const ROWS: usize>: IsVectorLike {
    fn c(val: V<ROWS>) -> Self;
    fn real(&self) -> &V<ROWS>;

    fn squared_norm(&self) -> S;
    fn norm(&self) -> S;
    fn get(&self, idx: usize) -> S;
    fn set_c(&mut self, idx: usize, v: f64);

    fn from_array(vals: [S; ROWS]) -> Self;
    fn from_c_array(vals: [f64; ROWS]) -> Self;

    fn scaled(&self, v: S) -> Self;

    fn get_fixed_rows<const R: usize>(&self, start: usize) -> S::Vector<R>;
    // fn from_fn<F: FnMut(usize) -> S>(f: F) -> Self;

    fn to_mat(self) -> S::Matrix<ROWS, 1>;

    fn block_vec2<const R0: usize, const R1: usize>(
        top_row: S::Vector<R0>,
        bot_row: S::Vector<R1>,
    ) -> Self;

    fn dot(self, rhs: Self) -> S;
}

impl<const ROWS: usize> IsVector<f64, ROWS> for V<ROWS> {
    fn from_array(vals: [f64; ROWS]) -> V<ROWS> {
        V::<ROWS>::from_row_slice(&vals[..])
    }

    fn from_c_array(vals: [f64; ROWS]) -> Self {
        V::<ROWS>::from_row_slice(&vals[..])
    }

    fn get(&self, idx: usize) -> f64 {
        self[idx]
    }

    fn norm(&self) -> f64 {
        self.norm()
    }

    fn squared_norm(&self) -> f64 {
        self.norm_squared()
    }

    fn c(val: V<ROWS>) -> Self {
        val
    }

    fn real(&self) -> &Self {
        self
    }

    fn get_fixed_rows<const R: usize>(&self, start: usize) -> V<R> {
        self.fixed_rows::<R>(start).into()
    }

    fn to_mat(self) -> M<ROWS, 1> {
        self
    }

    fn block_vec2<const R0: usize, const R1: usize>(top_row: V<R0>, bot_row: V<R1>) -> Self {
        assert_eq!(ROWS, R0 + R1);
        let mut m = Self::zero();

        m.fixed_view_mut::<R0, 1>(0, 0).copy_from(&top_row);
        m.fixed_view_mut::<R1, 1>(R0, 0).copy_from(&bot_row);
        m
    }

    fn set_c(&mut self, idx: usize, v: f64) {
        self[idx] = v;
    }

    fn scaled(&self, v: f64) -> Self {
        self * v
    }

    fn dot(self, rhs: Self) -> f64 {
        V::dot(&self, &rhs)
    }
}

pub fn cross<S: IsScalar>(lhs: S::Vector<3>, rhs: S::Vector<3>) -> S::Vector<3> {
    let l0 = lhs.get(0);
    let l1 = lhs.get(1);
    let l2 = lhs.get(2);

    let r0 = rhs.get(0);
    let r1 = rhs.get(1);
    let r2 = rhs.get(2);

    S::Vector::from_array([
        l1.clone() * r2.clone() - l2.clone() * r1.clone(),
        l2 * r0.clone() - l0.clone() * r2,
        l0 * r1 - l1 * r0,
    ])
}
