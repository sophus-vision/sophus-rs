use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::V;

pub trait TangentImpl<S: IsScalar, const DOF: usize> {
    fn tangent_examples() -> Vec<S::Vector<DOF>>;
}

pub trait ParamsImpl<S: IsScalar, const PARAMS: usize> {
    fn are_params_valid(params: &S::Vector<PARAMS>) -> bool;
    fn params_examples() -> Vec<S::Vector<PARAMS>>;
    fn invalid_params_examples() -> Vec<S::Vector<PARAMS>>;
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
    std::fmt::Debug + Clone
{
    fn params(&self) -> &S::Vector<PARAMS>;
    fn oplus(&self, tangent: &S::Vector<DOF>) -> Self;
    fn ominus(&self, rhs: &Self) -> S::Vector<DOF>;
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
