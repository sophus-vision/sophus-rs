use super::scalar::IsScalar;
use super::MatF64;
use super::VecF64;

use std::fmt::Debug;
use std::ops::Add;
use std::ops::Neg;
use std::ops::Sub;

/// is vector like
pub trait IsVectorLike:
    Debug + Clone + Sized + Neg<Output = Self> + Add<Output = Self> + Sub<Output = Self>
{
    /// create a zero vector
    fn zero() -> Self;
}

/// Vector - either a real (f64) or a dual number vector
pub trait IsVector<S: IsScalar<BATCH_SIZE>, const ROWS: usize, const BATCH_SIZE: usize>:
    IsVectorLike
{
    /// create a constant vector
    fn c(val: VecF64<ROWS>) -> Self;

    /// return the real part
    fn real(&self) -> &VecF64<ROWS>;

    /// squared norm
    fn squared_norm(&self) -> S;

    /// norm
    fn norm(&self) -> S;

    /// get ith element
    fn get(&self, idx: usize) -> S;

    /// set ith element as constant
    fn set_c(&mut self, idx: usize, v: f64);

    /// create a vector from an array
    fn from_array(vals: [S; ROWS]) -> Self;

    /// create a constant vector from an array
    fn from_c_array(vals: [f64; ROWS]) -> Self;

    /// return scaled vector
    fn scaled(&self, v: S) -> Self;

    /// get fixed rows
    fn get_fixed_rows<const R: usize>(&self, start: usize) -> S::Vector<R>;

    /// return the matrix representation
    fn to_mat(self) -> S::Matrix<ROWS, 1>;

    /// create a block vector
    fn block_vec2<const R0: usize, const R1: usize>(
        top_row: S::Vector<R0>,
        bot_row: S::Vector<R1>,
    ) -> Self;

    /// dot product
    fn dot(self, rhs: Self) -> S;

    /// return normalized vector
    fn normalized(&self) -> Self;
}

impl<const ROWS: usize> IsVector<f64, ROWS, 1> for VecF64<ROWS> {
    fn from_array(vals: [f64; ROWS]) -> VecF64<ROWS> {
        VecF64::<ROWS>::from_row_slice(&vals[..])
    }

    fn from_c_array(vals: [f64; ROWS]) -> Self {
        VecF64::<ROWS>::from_row_slice(&vals[..])
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

    fn c(val: VecF64<ROWS>) -> Self {
        val
    }

    fn real(&self) -> &Self {
        self
    }

    fn get_fixed_rows<const R: usize>(&self, start: usize) -> VecF64<R> {
        self.fixed_rows::<R>(start).into()
    }

    fn to_mat(self) -> MatF64<ROWS, 1> {
        self
    }

    fn block_vec2<const R0: usize, const R1: usize>(
        top_row: VecF64<R0>,
        bot_row: VecF64<R1>,
    ) -> Self {
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
        VecF64::dot(&self, &rhs)
    }

    fn normalized(&self) -> Self {
        self.normalize()
    }
}

/// cross product
pub fn cross<S: IsScalar<1>>(lhs: S::Vector<3>, rhs: S::Vector<3>) -> S::Vector<3> {
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
