use super::scalar::IsScalar;

pub trait ParamsImpl<S: IsScalar, const PARAMS: usize> {
    fn are_params_valid(params: &S::Vector<PARAMS>) -> bool;
    fn params_examples() -> Vec<S::Vector<PARAMS>>;
    fn invalid_params_examples() -> Vec<S::Vector<PARAMS>>;
}

pub trait HasParams<S: IsScalar, const PARAMS: usize>: ParamsImpl<S, PARAMS> {
    fn from_params(params: &S::Vector<PARAMS>) -> Self;
    fn set_params(&mut self, params: &S::Vector<PARAMS>);
    fn params(&self) -> &S::Vector<PARAMS>;
}
