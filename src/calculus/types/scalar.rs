use std::fmt::Debug;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;

use super::matrix::IsMatrix;
use super::vector::IsVector;
use super::M;
use super::V;

pub trait IsScalar:
    PartialOrd
    + PartialEq
    + Debug
    + Clone
    + Add<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Sub<Output = Self>
    + Sized
    + Neg<Output = Self>
    + num_traits::One
    + num_traits::Zero
    + From<f64>
{
    type Vector<const ROWS: usize>: IsVector<Self, ROWS>;
    type Matrix<const ROWS: usize, const COLS: usize>: IsMatrix<Self, ROWS, COLS>;

    fn c(val: f64) -> Self;
    fn real(&self) -> f64;

    fn abs(self) -> Self;
    fn cos(self) -> Self;
    fn sin(self) -> Self;
    fn tan(self) -> Self;
    fn acos(self) -> Self;
    fn asin(self) -> Self;
    fn atan(self) -> Self;
    fn sqrt(self) -> Self;

    fn atan2(self, x: Self) -> Self;

    fn value(self) -> f64;

    fn to_vec(self) -> Self::Vector<1>;
}

impl IsScalar for f64 {
    type Vector<const ROWS: usize> = V<ROWS>;
    type Matrix<const ROWS: usize, const COLS: usize> = M<ROWS, COLS>;

    fn abs(self) -> f64 {
        f64::abs(self)
    }

    fn cos(self) -> f64 {
        f64::cos(self)
    }

    fn sin(self) -> f64 {
        f64::sin(self)
    }

    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }

    fn c(val: f64) -> f64 {
        val
    }

    fn value(self) -> f64 {
        self
    }

    fn atan2(self, x: Self) -> Self {
        self.atan2(x)
    }

    fn real(&self) -> f64 {
        self.value()
    }

    fn to_vec(self) -> V<1> {
        V::<1>::new(self)
    }

    fn tan(self) -> Self {
        self.tan()
    }

    fn acos(self) -> Self {
        self.acos()
    }

    fn asin(self) -> Self {
        self.asin()
    }

    fn atan(self) -> Self {
        self.atan()
    }
}
