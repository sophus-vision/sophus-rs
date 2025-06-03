extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;

use sophus_autodiff::{
    linalg::{cross, EPS_F64},
    params::{HasParams, IsParamsImpl},
    manifold::IsManifold,
    points::example_points,
};
use crate::prelude::*;

/// Quaternion represented as `(w, x, y, z)`.
#[derive(Clone, Debug)]
pub struct Quaternion<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    params: S::Vector<4>,
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    Quaternion<S, BATCH, DM, DN>
{
    /// Creates a quaternion from its parameter vector `(w, x, y, z)`.
    pub fn from_params(params: S::Vector<4>) -> Self {
        Self {
            params,
        }
    }

    /// Returns the zero quaternion `(0,0,0,0)`.
    pub fn zero() -> Self {
        Self::from_params(S::Vector::<4>::zeros())
    }

    /// Returns the identity quaternion `(1,0,0,0)`.
    pub fn one() -> Self {
        Self::from_params(S::Vector::<4>::from_f64_array([1.0, 0.0, 0.0, 0.0]))
    }

    /// Access the underlying parameter vector.
    pub fn params(&self) -> &S::Vector<4> {
        &self.params
    }

    /// Mutable access to the parameter vector.
    pub fn params_mut(&mut self) -> &mut S::Vector<4> {
        &mut self.params
    }

    /// Returns the real component `w`.
    pub fn real(&self) -> S {
        self.params.elem(0)
    }

    /// Returns the imaginary component `(x,y,z)`.
    pub fn imag(&self) -> S::Vector<3> {
        self.params.get_fixed_subvec::<3>(1)
    }

    /// Quaternion multiplication.
    pub fn mul_q(&self, rhs: &Self) -> Self {
        Self::from_params(QuaternionImpl::<S, BATCH, DM, DN>::multiplication(
            &self.params, &rhs.params,
        ))
    }

    /// Quaternion addition.
    pub fn add_q(&self, rhs: &Self) -> Self {
        Self::from_params(self.params + rhs.params)
    }

    /// Conjugated quaternion.
    pub fn conjugate(&self) -> Self {
        Self::from_params(QuaternionImpl::<S, BATCH, DM, DN>::conjugate(&self.params))
    }

    /// Inverse quaternion.
    pub fn inverse(&self) -> Self {
        Self::from_params(QuaternionImpl::<S, BATCH, DM, DN>::inverse(&self.params))
    }

    /// Quaternion norm.
    pub fn norm(&self) -> S {
        QuaternionImpl::<S, BATCH, DM, DN>::norm(&self.params)
    }

    /// Quaternion squared norm.
    pub fn squared_norm(&self) -> S {
        QuaternionImpl::<S, BATCH, DM, DN>::squared_norm(&self.params)
    }

    /// Scale quaternion by scalar.
    pub fn scale(&self, s: S) -> Self {
        Self::from_params(self.params.scaled(s))
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    core::ops::Add for Quaternion<S, BATCH, DM, DN>
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_q(&rhs)
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    core::ops::Mul for Quaternion<S, BATCH, DM, DN>
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_q(&rhs)
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsParamsImpl<S, 4, BATCH, DM, DN> for Quaternion<S, BATCH, DM, DN>
{
    fn are_params_valid(_params: S::Vector<4>) -> S::Mask {
        S::Mask::all_true()
    }

    fn params_examples() -> Vec<S::Vector<4>> {
        example_points::<S, 4, BATCH, DM, DN>()
    }

    fn invalid_params_examples() -> Vec<S::Vector<4>> {
        Vec::new()
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    HasParams<S, 4, BATCH, DM, DN> for Quaternion<S, BATCH, DM, DN>
{
    fn from_params(params: S::Vector<4>) -> Self {
        Self::from_params(params)
    }

    fn set_params(&mut self, params: S::Vector<4>) {
        self.params = params;
    }

    fn params(&self) -> &S::Vector<4> {
        &self.params
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsManifold<S, 4, 4, BATCH, DM, DN> for Quaternion<S, BATCH, DM, DN>
{
    fn oplus(&self, tangent: &S::Vector<4>) -> Self {
        Self::from_params(*self.params() + *tangent)
    }

    fn ominus(&self, rhs: &Self) -> S::Vector<4> {
        *self.params() - *rhs.params()
    }
}

/// Quaternion with `f64` scalar type.
pub type QuaternionF64 = Quaternion<f64, 1, 0, 0>;

/// Implementation utilities for [`Quaternion`].
#[derive(Clone, Copy, Debug)]
pub struct QuaternionImpl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    QuaternionImpl<S, BATCH, DM, DN>
{
    /// Returns the zero quaternion.
    pub fn zero() -> S::Vector<4> {
        S::Vector::<4>::zeros()
    }

    /// Returns the identity quaternion.
    pub fn one() -> S::Vector<4> {
        S::Vector::<4>::from_f64_array([1.0, 0.0, 0.0, 0.0])
    }

    /// Multiplies two quaternions.
    pub fn multiplication(lhs: &S::Vector<4>, rhs: &S::Vector<4>) -> S::Vector<4> {
        let lhs_re = lhs.elem(0);
        let rhs_re = rhs.elem(0);

        let lhs_ivec = lhs.get_fixed_subvec::<3>(1);
        let rhs_ivec = rhs.get_fixed_subvec::<3>(1);

        let re = lhs_re * rhs_re - lhs_ivec.dot(rhs_ivec);
        let ivec = rhs_ivec.scaled(lhs_re)
            + lhs_ivec.scaled(rhs_re)
            + cross::<S, BATCH, DM, DN>(lhs_ivec, rhs_ivec);

        let mut params = S::Vector::block_vec2(re.to_vec(), ivec);

        if ((params.norm() - S::from_f64(1.0))
            .abs()
            .greater_equal(&S::from_f64(EPS_F64)))
            .any()
        {
            params = params.normalized();
        }
        params
    }

    /// Adds two quaternions component-wise.
    pub fn addition(a: &S::Vector<4>, b: &S::Vector<4>) -> S::Vector<4> {
        *a + *b
    }

    /// Conjugates a quaternion.
    pub fn conjugate(a: &S::Vector<4>) -> S::Vector<4> {
        S::Vector::from_array([a.elem(0), -a.elem(1), -a.elem(2), -a.elem(3)])
    }

    /// Computes the inverse quaternion.
    pub fn inverse(q: &S::Vector<4>) -> S::Vector<4> {
        Self::conjugate(q).scaled(S::from_f64(1.0) / q.squared_norm())
    }

    /// Returns the quaternion norm.
    pub fn norm(q: &S::Vector<4>) -> S {
        q.norm()
    }

    /// Returns the squared quaternion norm.
    pub fn squared_norm(q: &S::Vector<4>) -> S {
        q.squared_norm()
    }
}
