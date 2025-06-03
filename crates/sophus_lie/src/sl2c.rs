//! Utilities for 2x2 complex matrices.
//!
//! This struct mirrors [`Quaternion`] and [`Complex`] and simply stores
//! the eight real components of a 2×2 complex matrix.

extern crate alloc;

use alloc::vec::Vec;
use core::marker::PhantomData;

use sophus_autodiff::{
    manifold::IsManifold,
    params::{HasParams, IsParamsImpl},
    points::example_points,
};

use crate::prelude::*;

/// 2×2 complex matrix represented as
/// `(re00, im00, re01, im01, re10, im10, re11, im11)`.
#[derive(Clone, Debug)]
pub struct Sl2c<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize> {
    params: S::Vector<8>,
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    Sl2c<S, BATCH, DM, DN>
{
    /// Creates a matrix from its parameter vector.
    pub fn from_params(params: S::Vector<8>) -> Self {
        Self { params }
    }

    /// Returns the zero matrix.
    pub fn zero() -> Self {
        Self::from_params(S::Vector::<8>::zeros())
    }

    /// Returns the identity matrix.
    pub fn one() -> Self {
        Self::from_params(S::Vector::<8>::from_f64_array([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ]))
    }

    /// Access the underlying parameter vector.
    pub fn params(&self) -> &S::Vector<8> {
        &self.params
    }

    /// Mutable access to the parameter vector.
    pub fn params_mut(&mut self) -> &mut S::Vector<8> {
        &mut self.params
    }

    /// Matrix multiplication.
    pub fn mult(&self, rhs: Self) -> Self {
        Self::from_params(Sl2cImpl::<S, BATCH, DM, DN>::mult(&self.params, rhs.params))
    }

    /// Component-wise addition.
    pub fn add(&self, rhs: Self) -> Self {
        Self::from_params(self.params + rhs.params)
    }

    /// Conjugated matrix.
    pub fn conjugate(&self) -> Self {
        Self::from_params(Sl2cImpl::<S, BATCH, DM, DN>::conjugate(&self.params))
    }

    /// Transposed matrix.
    pub fn transpose(&self) -> Self {
        Self::from_params(Sl2cImpl::<S, BATCH, DM, DN>::transpose(&self.params))
    }

    /// Conjugate transpose.
    pub fn conjugate_transpose(&self) -> Self {
        Self::from_params(Sl2cImpl::<S, BATCH, DM, DN>::conjugate_transpose(&self.params))
    }

    /// Inverse matrix.
    pub fn inverse(&self) -> Self {
        Self::from_params(Sl2cImpl::<S, BATCH, DM, DN>::inverse(&self.params))
    }

    /// Frobenius norm of the matrix.
    pub fn norm(&self) -> S {
        Sl2cImpl::<S, BATCH, DM, DN>::norm(&self.params)
    }

    /// Squared Frobenius norm of the matrix.
    pub fn squared_norm(&self) -> S {
        Sl2cImpl::<S, BATCH, DM, DN>::squared_norm(&self.params)
    }

    /// Scale the matrix by a scalar.
    pub fn scale(&self, s: S) -> Self {
        Self::from_params(self.params.scaled(s))
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    core::ops::Add for Sl2c<S, BATCH, DM, DN>
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Sl2c::add(&self, rhs)
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    core::ops::Mul for Sl2c<S, BATCH, DM, DN>
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mult(rhs)
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsParamsImpl<S, 8, BATCH, DM, DN> for Sl2c<S, BATCH, DM, DN>
{
    fn are_params_valid(_params: S::Vector<8>) -> S::Mask {
        S::Mask::all_true()
    }

    fn params_examples() -> Vec<S::Vector<8>> {
        example_points::<S, 8, BATCH, DM, DN>()
    }

    fn invalid_params_examples() -> Vec<S::Vector<8>> {
        Vec::new()
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    HasParams<S, 8, BATCH, DM, DN> for Sl2c<S, BATCH, DM, DN>
{
    fn from_params(params: S::Vector<8>) -> Self {
        Self::from_params(params)
    }

    fn set_params(&mut self, params: S::Vector<8>) {
        self.params = params;
    }

    fn params(&self) -> &S::Vector<8> {
        &self.params
    }
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    IsManifold<S, 8, 8, BATCH, DM, DN> for Sl2c<S, BATCH, DM, DN>
{
    fn oplus(&self, tangent: &S::Vector<8>) -> Self {
        Self::from_params(*self.params() + *tangent)
    }

    fn ominus(&self, rhs: &Self) -> S::Vector<8> {
        *self.params() - *rhs.params()
    }
}

/// Matrix with `f64` scalar type.
pub type Sl2cF64 = Sl2c<f64, 1, 0, 0>;

/// Implementation utilities for [`Sl2c`].
#[derive(Clone, Copy, Debug)]
pub struct Sl2cImpl<
    S: IsScalar<BATCH, DM, DN>,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<BATCH, DM, DN>, const BATCH: usize, const DM: usize, const DN: usize>
    Sl2cImpl<S, BATCH, DM, DN>
{
    /// Returns the zero matrix.
    pub fn zero() -> S::Vector<8> {
        S::Vector::<8>::zeros()
    }

    /// Returns the identity matrix.
    pub fn one() -> S::Vector<8> {
        S::Vector::<8>::from_f64_array([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ])
    }

    /// Multiplies two matrices.
    pub fn mult(lhs: &S::Vector<8>, rhs: S::Vector<8>) -> S::Vector<8> {
        let lhs_re = S::Matrix::<2, 2>::from_array2([
            [lhs.elem(0), lhs.elem(2)],
            [lhs.elem(4), lhs.elem(6)],
        ]);
        let lhs_im = S::Matrix::<2, 2>::from_array2([
            [lhs.elem(1), lhs.elem(3)],
            [lhs.elem(5), lhs.elem(7)],
        ]);

        let rhs_re = S::Matrix::<2, 2>::from_array2([
            [rhs.elem(0), rhs.elem(2)],
            [rhs.elem(4), rhs.elem(6)],
        ]);
        let rhs_im = S::Matrix::<2, 2>::from_array2([
            [rhs.elem(1), rhs.elem(3)],
            [rhs.elem(5), rhs.elem(7)],
        ]);

        let re = lhs_re.mat_mul(&rhs_re) - lhs_im.mat_mul(&rhs_im);
        let im = lhs_re.mat_mul(&rhs_im) + lhs_im.mat_mul(&rhs_re);

        S::Vector::<8>::from_array([
            re.elem([0, 0]),
            im.elem([0, 0]),
            re.elem([0, 1]),
            im.elem([0, 1]),
            re.elem([1, 0]),
            im.elem([1, 0]),
            re.elem([1, 1]),
            im.elem([1, 1]),
        ])
    }

    /// Adds two matrices component-wise.
    pub fn add(a: &S::Vector<8>, b: S::Vector<8>) -> S::Vector<8> {
        *a + b
    }

    /// Conjugates a matrix.
    pub fn conjugate(m: &S::Vector<8>) -> S::Vector<8> {
        S::Vector::<8>::from_array([
            m.elem(0),
            -m.elem(1),
            m.elem(2),
            -m.elem(3),
            m.elem(4),
            -m.elem(5),
            m.elem(6),
            -m.elem(7),
        ])
    }

    /// Transposes a matrix.
    pub fn transpose(m: &S::Vector<8>) -> S::Vector<8> {
        S::Vector::<8>::from_array([
            m.elem(0),
            m.elem(1),
            m.elem(4),
            m.elem(5),
            m.elem(2),
            m.elem(3),
            m.elem(6),
            m.elem(7),
        ])
    }

    /// Conjugate transpose of a matrix.
    pub fn conjugate_transpose(m: &S::Vector<8>) -> S::Vector<8> {
        Self::conjugate(&Self::transpose(m))
    }

    fn complex_mult(a_re: S, a_im: S, b_re: S, b_im: S) -> (S, S) {
        (
            a_re * b_re - a_im * b_im,
            a_re * b_im + a_im * b_re,
        )
    }

    fn determinant(m: &S::Vector<8>) -> (S, S) {
        let a_re = m.elem(0);
        let a_im = m.elem(1);
        let b_re = m.elem(2);
        let b_im = m.elem(3);
        let c_re = m.elem(4);
        let c_im = m.elem(5);
        let d_re = m.elem(6);
        let d_im = m.elem(7);

        let (ad_re, ad_im) = Self::complex_mult(a_re, a_im, d_re, d_im);
        let (bc_re, bc_im) = Self::complex_mult(b_re, b_im, c_re, c_im);
        (ad_re - bc_re, ad_im - bc_im)
    }

    /// Inverse of a matrix.
    pub fn inverse(m: &S::Vector<8>) -> S::Vector<8> {
        let (det_re, det_im) = Self::determinant(m);
        let norm_sq = det_re * det_re + det_im * det_im;
        let inv_det_re = det_re / norm_sq;
        let inv_det_im = -det_im / norm_sq;

        let a_re = m.elem(0);
        let a_im = m.elem(1);
        let b_re = m.elem(2);
        let b_im = m.elem(3);
        let c_re = m.elem(4);
        let c_im = m.elem(5);
        let d_re = m.elem(6);
        let d_im = m.elem(7);

        let (e0_re, e0_im) = Self::complex_mult(d_re, d_im, inv_det_re, inv_det_im);
        let (e1_re, e1_im) = Self::complex_mult(-b_re, -b_im, inv_det_re, inv_det_im);
        let (e2_re, e2_im) = Self::complex_mult(-c_re, -c_im, inv_det_re, inv_det_im);
        let (e3_re, e3_im) = Self::complex_mult(a_re, a_im, inv_det_re, inv_det_im);

        S::Vector::<8>::from_array([
            e0_re, e0_im, e1_re, e1_im, e2_re, e2_im, e3_re, e3_im,
        ])
    }

    /// Frobenius norm of the matrix.
    pub fn norm(m: &S::Vector<8>) -> S {
        m.norm()
    }

    /// Squared Frobenius norm of the matrix.
    pub fn squared_norm(m: &S::Vector<8>) -> S {
        m.squared_norm()
    }
}
