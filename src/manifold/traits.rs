use crate::calculus::points::example_points;
use crate::calculus::types::params::HasParams;
use crate::calculus::types::params::ParamsImpl;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::V;

pub trait TangentImpl<S: IsScalar, const DOF: usize> {
    fn tangent_examples() -> Vec<S::Vector<DOF>>;
}

pub trait ManifoldImpl<
    S: IsScalar,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT_DIM: usize,
>: ParamsImpl<S, PARAMS> + TangentImpl<S, DOF>
{
    fn oplus(params: &S::Vector<PARAMS>, tangent: &S::Vector<DOF>) -> S::Vector<PARAMS>;
    fn ominus(params1: &S::Vector<PARAMS>, params2: &S::Vector<PARAMS>) -> S::Vector<DOF>;
}

pub trait IsManifold<S: IsScalar, const PARAMS: usize, const DOF: usize>:
    HasParams<S, PARAMS> + std::fmt::Debug + Clone
{
    fn params(&self) -> &S::Vector<PARAMS>;
    fn oplus(&self, tangent: &S::Vector<DOF>) -> Self;
    fn ominus(&self, rhs: &Self) -> S::Vector<DOF>;
}

impl<const N: usize> ParamsImpl<f64, N> for V<N> {
    fn are_params_valid(_params: &<f64 as IsScalar>::Vector<N>) -> bool {
        true
    }

    fn params_examples() -> Vec<<f64 as IsScalar>::Vector<N>> {
        example_points::<f64, N>()
    }

    fn invalid_params_examples() -> Vec<<f64 as IsScalar>::Vector<N>> {
        vec![]
    }
}

impl<const N: usize> HasParams<f64, N> for V<N> {
    fn from_params(params: &<f64 as IsScalar>::Vector<N>) -> Self {
        params.clone()
    }

    fn set_params(&mut self, params: &<f64 as IsScalar>::Vector<N>) {
        *self = *params
    }

    fn params(&self) -> &<f64 as IsScalar>::Vector<N> {
        self
    }
}

impl<const N: usize> IsManifold<f64, N, N> for V<N> {
    fn oplus(&self, tangent: &<f64 as IsScalar>::Vector<N>) -> Self {
        self + tangent
    }

    fn ominus(&self, rhs: &Self) -> <f64 as IsScalar>::Vector<N> {
        self - rhs
    }

    fn params(&self) -> &<f64 as IsScalar>::Vector<N> {
        self
    }
}
