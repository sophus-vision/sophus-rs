use crate::types::scalar::IsScalar;

/// Parameter implementation.
pub trait ParamsImpl<S: IsScalar<BATCH_SIZE>, const PARAMS: usize, const BATCH_SIZE: usize> {
    /// Is the parameter vector valid?
    fn are_params_valid(params: &S::Vector<PARAMS>) -> bool;
    /// Examples of valid parameter vectors.
    fn params_examples() -> Vec<S::Vector<PARAMS>>;
    /// Examples of invalid parameter vectors.
    fn invalid_params_examples() -> Vec<S::Vector<PARAMS>>;
}

/// A trait for types that have parameters.
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
