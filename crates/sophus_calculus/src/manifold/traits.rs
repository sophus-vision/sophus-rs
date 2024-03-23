use crate::points::example_points;
use crate::types::params::HasParams;
use crate::types::params::ParamsImpl;
use crate::types::scalar::IsScalar;
use crate::types::VecF64;

/// A tangent implementation.
pub trait TangentImpl<S: IsScalar<BATCH_SIZE>, const DOF: usize, const BATCH_SIZE: usize> {
    /// Examples of tangent vectors.
    fn tangent_examples() -> Vec<S::Vector<DOF>>;
}

/// A manifold implementation.
pub trait ManifoldImpl<
    S: IsScalar<BATCH_SIZE>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT_DIM: usize,
    const BATCH_SIZE: usize,
>: ParamsImpl<S, PARAMS, BATCH_SIZE> + TangentImpl<S, DOF, BATCH_SIZE>
{
    /// o-plus operation.
    fn oplus(params: &S::Vector<PARAMS>, tangent: &S::Vector<DOF>) -> S::Vector<PARAMS>;
    /// o-minus operation.
    fn ominus(params1: &S::Vector<PARAMS>, params2: &S::Vector<PARAMS>) -> S::Vector<DOF>;
}

/// A manifold.
pub trait IsManifold<
    S: IsScalar<BATCH_SIZE>,
    const PARAMS: usize,
    const DOF: usize,
    const BATCH_SIZE: usize,
>: HasParams<S, PARAMS, BATCH_SIZE> + std::fmt::Debug + Clone
{
    /// manifold parameters
    fn params(&self) -> &S::Vector<PARAMS>;
    /// o-plus operation
    fn oplus(&self, tangent: &S::Vector<DOF>) -> Self;
    /// o-minus operation
    fn ominus(&self, rhs: &Self) -> S::Vector<DOF>;
}

impl<const N: usize> ParamsImpl<f64, N, 1> for VecF64<N> {
    fn are_params_valid(_params: &<f64 as IsScalar<1>>::Vector<N>) -> bool {
        true
    }

    fn params_examples() -> Vec<<f64 as IsScalar<1>>::Vector<N>> {
        example_points::<f64, N>()
    }

    fn invalid_params_examples() -> Vec<<f64 as IsScalar<1>>::Vector<N>> {
        vec![]
    }
}

impl<const N: usize> HasParams<f64, N, 1> for VecF64<N> {
    fn from_params(params: &<f64 as IsScalar<1>>::Vector<N>) -> Self {
        *params
    }

    fn set_params(&mut self, params: &<f64 as IsScalar<1>>::Vector<N>) {
        *self = *params
    }

    fn params(&self) -> &<f64 as IsScalar<1>>::Vector<N> {
        self
    }
}

impl<const N: usize> IsManifold<f64, N, N, 1> for VecF64<N> {
    fn oplus(&self, tangent: &<f64 as IsScalar<1>>::Vector<N>) -> Self {
        self + tangent
    }

    fn ominus(&self, rhs: &Self) -> <f64 as IsScalar<1>>::Vector<N> {
        self - rhs
    }

    fn params(&self) -> &<f64 as IsScalar<1>>::Vector<N> {
        self
    }
}
