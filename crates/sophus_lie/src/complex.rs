extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;

use sophus_autodiff::{
    manifold::IsManifold,
    params::{
        HasParams,
        IsParamsImpl,
    },
    points::example_points,
};

use crate::prelude::*;

/// Complex number represented as `(re, im)`.
#[derive(Clone, Debug, Copy)]
pub struct Complex<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
{
    params: S::Vector<2>,
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    Complex<S, BATCH, DM, DN>
{
    /// Creates a complex number from real and imaginary scalar.
    #[inline]
    #[must_use]
    pub fn from_real_imag(real: S, imag: S) -> Self {
        Self::from_params(S::Vector::<2>::from_array([real, imag]))
    }

    /// Creates a complex number from its parameter vector `(re, im)`.
    #[inline]
    #[must_use]
    pub fn from_params(params: S::Vector<2>) -> Self {
        Self { params }
    }

    /// Returns zero `(0,0)`.
    pub fn zero() -> Self {
        Self::from_params(S::Vector::<2>::zeros())
    }

    /// Returns one `(1,0)`.
    pub fn one() -> Self {
        Self::from_params(S::Vector::<2>::from_f64_array([1.0, 0.0]))
    }

    /// Access the underlying parameter vector.
    pub fn params(&self) -> &S::Vector<2> {
        &self.params
    }

    /// Mutable access to the parameter vector.
    pub fn params_mut(&mut self) -> &mut S::Vector<2> {
        &mut self.params
    }

    /// Returns the real component.
    pub fn real(&self) -> S {
        self.params.elem(0)
    }

    /// Returns the imaginary component.
    pub fn imag(&self) -> S {
        self.params.elem(1)
    }

    /// Complex multiplication.
    pub fn mult(&self, rhs: Self) -> Self {
        Self::from_params(ComplexImpl::<S, BATCH, DM, DN>::mult(
            &self.params,
            rhs.params,
        ))
    }

    /// Complex addition.
    pub fn add(&self, rhs: Self) -> Self {
        Self::from_params(self.params + rhs.params)
    }

    /// Conjugated complex number.
    pub fn conjugate(&self) -> Self {
        Self::from_params(ComplexImpl::<S, BATCH, DM, DN>::conjugate(&self.params))
    }

    /// Inverse complex number.
    pub fn inverse(&self) -> Self {
        Self::from_params(ComplexImpl::<S, BATCH, DM, DN>::inverse(&self.params))
    }

    /// Complex norm.
    pub fn norm(&self) -> S {
        ComplexImpl::<S, BATCH, DM, DN>::norm(&self.params)
    }

    /// Complex squared norm.
    pub fn squared_norm(&self) -> S {
        ComplexImpl::<S, BATCH, DM, DN>::squared_norm(&self.params)
    }

    /// Scale complex number by scalar.
    pub fn scale(&self, s: S) -> Self {
        Self::from_params(self.params.scaled(s))
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    core::ops::Add for Complex<S, BATCH, DM, DN>
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Complex::add(&self, rhs)
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    core::ops::Mul for Complex<S, BATCH, DM, DN>
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mult(rhs)
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsParamsImpl<S, 2, BATCH, DM, DN> for Complex<S, BATCH, DM, DN>
{
    fn are_params_valid(_params: S::Vector<2>) -> S::Mask {
        S::Mask::all_true()
    }

    fn params_examples() -> Vec<S::Vector<2>> {
        example_points::<S, 2, BATCH, DM, DN>()
    }

    fn invalid_params_examples() -> Vec<S::Vector<2>> {
        Vec::new()
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    HasParams<S, 2, BATCH, DM, DN> for Complex<S, BATCH, DM, DN>
{
    fn from_params(params: S::Vector<2>) -> Self {
        Self::from_params(params)
    }

    fn set_params(&mut self, params: S::Vector<2>) {
        self.params = params;
    }

    fn params(&self) -> &S::Vector<2> {
        &self.params
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsManifold<S, 2, 2, BATCH, DM, DN> for Complex<S, BATCH, DM, DN>
{
    fn oplus(&self, tangent: &S::Vector<2>) -> Self {
        Self::from_params(*self.params() + *tangent)
    }

    fn ominus(&self, rhs: &Self) -> S::Vector<2> {
        *self.params() - *rhs.params()
    }
}

/// Complex number with `f64` scalar type.
pub type ComplexF64 = Complex<f64, 1, 0, 0>;

/// Implementation utilities for [`Complex`].
#[derive(Clone, Copy, Debug)]
pub struct ComplexImpl<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    ComplexImpl<S, BATCH, DM, DN>
{
    /// Returns the zero complex number.
    pub fn zero() -> S::Vector<2> {
        S::Vector::<2>::zeros()
    }

    /// Returns the identity complex number.
    pub fn one() -> S::Vector<2> {
        S::Vector::<2>::from_f64_array([1.0, 0.0])
    }

    /// Multiplies two complex numbers.
    pub fn mult(lhs: &S::Vector<2>, rhs: S::Vector<2>) -> S::Vector<2> {
        let lhs_re = lhs.elem(0);
        let rhs_re = rhs.elem(0);

        let lhs_im = lhs.elem(1);
        let rhs_im = rhs.elem(1);

        let re = lhs_re * rhs_re - lhs_im * rhs_im;
        let im = lhs_re * rhs_im + lhs_im * rhs_re;

        S::Vector::<2>::from_array([re, im])
    }

    /// Adds two complex numbers component-wise.
    pub fn add(a: &S::Vector<2>, b: S::Vector<2>) -> S::Vector<2> {
        *a + b
    }

    /// Conjugates a complex number.
    pub fn conjugate(a: &S::Vector<2>) -> S::Vector<2> {
        S::Vector::from_array([a.elem(0), -a.elem(1)])
    }

    /// Computes the inverse complex number.
    pub fn inverse(z: &S::Vector<2>) -> S::Vector<2> {
        Self::conjugate(z).scaled(S::from_f64(1.0) / z.squared_norm())
    }

    /// Returns the complex norm.
    pub fn norm(z: &S::Vector<2>) -> S {
        z.norm()
    }

    /// Returns the squared complex norm.
    pub fn squared_norm(z: &S::Vector<2>) -> S {
        z.squared_norm()
    }
}
