use nalgebra::SVector;

type V<const N: usize> = SVector<f64, N>;

pub trait ParamsImpl<const PARAMS: usize> {
    fn are_params_valid(params: &V<PARAMS>) -> bool;
    fn params_examples() -> Vec<V<PARAMS>>;
    fn invalid_params_examples() -> Vec<V<PARAMS>>;
}
