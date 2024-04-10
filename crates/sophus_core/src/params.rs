use crate::linalg::VecF64;
use crate::points::example_points;
use crate::prelude::*;

/// Parameter implementation.
pub trait ParamsImpl<S: IsScalar<BATCH_SIZE>, const PARAMS: usize, const BATCH_SIZE: usize> {
    /// Is the parameter vector valid?
    fn are_params_valid(params: &S::Vector<PARAMS>) -> S::Mask;
    /// Examples of valid parameter vectors.
    fn params_examples() -> Vec<S::Vector<PARAMS>>;
    /// Examples of invalid parameter vectors.
    fn invalid_params_examples() -> Vec<S::Vector<PARAMS>>;
}

/// A trait for linalg that have parameters.
pub trait HasParams<S: IsScalar<BATCH_SIZE>, const PARAMS: usize, const BATCH_SIZE: usize>:
    ParamsImpl<S, PARAMS, BATCH_SIZE>
{
    /// Create from parameters.
    fn from_params(params: &S::Vector<PARAMS>) -> Self;
    /// Set parameters.
    fn set_params(&mut self, params: &S::Vector<PARAMS>);
    /// Get parameters.
    fn params(&self) -> &S::Vector<PARAMS>;
}

impl<const N: usize> ParamsImpl<f64, N, 1> for VecF64<N> {
    fn are_params_valid(_params: &VecF64<N>) -> bool {
        true
    }

    fn params_examples() -> Vec<VecF64<N>> {
        example_points::<f64, N, 1>()
    }

    fn invalid_params_examples() -> Vec<VecF64<N>> {
        vec![]
    }
}

impl<const N: usize> HasParams<f64, N, 1> for VecF64<N> {
    fn from_params(params: &VecF64<N>) -> Self {
        *params
    }

    fn set_params(&mut self, params: &VecF64<N>) {
        *self = *params
    }

    fn params(&self) -> &VecF64<N> {
        self
    }
}
