use core::f64;

use nalgebra::SVector;
use num_traits::Bounded;

use crate::prelude::*;
extern crate alloc;

/// Traits for points
pub trait IsPoint<const D: usize>: Copy + Bounded {
    /// Point type
    type Point: Bounded;

    /// smallest point
    fn smallest() -> Self::Point {
        Bounded::min_value()
    }

    /// largest point
    fn largest() -> Self::Point {
        Bounded::max_value()
    }

    /// clamp point
    fn clamp(&self, min: Self, max: Self) -> Self::Point;

    /// check if point is less or equal to another point
    fn is_less_equal(&self, rhs: Self) -> bool;
}

/// Traits for points - including +infinity / -infinity
pub trait IsUnboundedPoint<const D: usize>: IsPoint<D> {
    /// +infinity
    fn infinity() -> Self::Point;

    /// -infinity
    fn neg_infinity() -> Self::Point;
}

impl IsPoint<1> for f64 {
    type Point = f64;

    fn clamp(&self, min: f64, max: f64) -> f64 {
        f64::clamp(*self, min, max)
    }

    fn is_less_equal(&self, rhs: f64) -> bool {
        self <= &rhs
    }
}

impl IsUnboundedPoint<1> for f64 {
    fn infinity() -> Self::Point {
        f64::INFINITY
    }

    fn neg_infinity() -> Self::Point {
        f64::NEG_INFINITY
    }
}

impl IsPoint<1> for i64 {
    type Point = i64;

    fn clamp(&self, min: i64, max: i64) -> i64 {
        Ord::clamp(*self, min, max)
    }

    fn is_less_equal(&self, rhs: i64) -> bool {
        self <= &rhs
    }
}

impl<const D: usize> IsPoint<D> for SVector<f64, D> {
    type Point = Self;

    fn clamp(&self, min: Self, max: Self) -> Self::Point {
        let mut p: Self::Point = Self::Point::zeros();
        for i in 0..D {
            p[i] = self[i].clamp(min[i], max[i]);
        }
        p
    }

    fn is_less_equal(&self, rhs: Self::Point) -> bool {
        self.iter()
            .zip(rhs.iter())
            .all(|(a, b)| a.is_less_equal(*b))
    }
}

impl<const D: usize> IsUnboundedPoint<D> for SVector<f64, D> {
    fn infinity() -> Self::Point {
        SVector::<f64, D>::repeat(f64::INFINITY)
    }

    fn neg_infinity() -> Self::Point {
        SVector::<f64, D>::repeat(f64::NEG_INFINITY)
    }
}

impl<const D: usize> IsPoint<D> for SVector<i64, D> {
    type Point = Self;

    fn clamp(&self, min: Self::Point, max: Self::Point) -> Self::Point {
        let mut p: Self::Point = Self::Point::zeros();
        for i in 0..D {
            p[i] = self[i].clamp(min[i], max[i]);
        }
        p
    }

    fn is_less_equal(&self, _rhs: Self::Point) -> bool {
        self.iter()
            .zip(_rhs.iter())
            .all(|(a, b)| a.is_less_equal(*b))
    }
}

/// Example points
pub fn example_points<
    S: IsScalar<BATCH, DM, DN>,
    const POINT: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>() -> alloc::vec::Vec<S::Vector<POINT>> {
    let points4 = alloc::vec![
        S::Vector::<4>::from_f64_array([0.1, 0.0, 0.0, 0.0]),
        S::Vector::<4>::from_f64_array([1.0, 4.0, 1.0, 0.5]),
        S::Vector::<4>::from_f64_array([0.7, 5.0, 1.1, (-5.0)]),
        S::Vector::<4>::from_f64_array([1.0, 3.0, 1.0, 0.5]),
        S::Vector::<4>::from_f64_array([0.7, 5.0, 0.8, (-5.0)]),
        S::Vector::<4>::from_f64_array([1.0, 3.0, 1.0, 0.5]),
        S::Vector::<4>::from_f64_array([-0.7, 5.0, 0.1, (-5.0)]),
        S::Vector::<4>::from_f64_array([2.0, (-3.0), 1.0, 0.5]),
    ];

    let mut out: alloc::vec::Vec<S::Vector<POINT>> = alloc::vec![];
    for p4 in points4 {
        let mut v = S::Vector::<POINT>::zeros();
        for i in 0..POINT.min(4) {
            let val = p4.elem(i);
            *v.elem_mut(i) = val;
        }
        out.push(v)
    }
    out
}
