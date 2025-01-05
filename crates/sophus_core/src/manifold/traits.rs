use crate::linalg::VecF64;
use crate::params::ParamsImpl;
use crate::prelude::*;
extern crate alloc;

/// A tangent implementation.
pub trait TangentImpl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>
{
    /// Examples of tangent vectors.
    fn tangent_examples() -> alloc::vec::Vec<S::Vector<DOF>>;
}

/// A manifold implementation.
pub trait ManifoldImpl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>: ParamsImpl<S, PARAMS, BATCH, DM, DN> + TangentImpl<S, DOF, BATCH, DM, DN>
{
    /// o-plus operation.
    fn oplus(params: &S::Vector<PARAMS>, tangent: &S::Vector<DOF>) -> S::Vector<PARAMS>;
    /// o-minus operation.
    fn ominus(params1: &S::Vector<PARAMS>, params2: &S::Vector<PARAMS>) -> S::Vector<DOF>;
}

/// A manifold.
pub trait IsManifold<
    S: IsScalar<BATCH, DM, DN>,
    const PARAMS: usize,
    const DOF: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>: HasParams<S, PARAMS, BATCH, DM, DN> + core::fmt::Debug + Clone
{
    /// o-plus operation
    fn oplus(&self, tangent: &S::Vector<DOF>) -> Self;
    /// o-minus operation
    fn ominus(&self, rhs: &Self) -> S::Vector<DOF>;
}

impl<const N: usize> IsManifold<f64, N, N, 1, 0, 0> for VecF64<N> {
    fn oplus(&self, tangent: &VecF64<N>) -> Self {
        self + tangent
    }

    fn ominus(&self, rhs: &Self) -> VecF64<N> {
        self - rhs
    }
}
