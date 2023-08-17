use super::batch_types::*;

pub trait ParamsTestUtils<const B: usize, const PARAMS: usize> {
    fn tutil_params_examples() -> Vec<V<B, PARAMS>>;
    fn tutil_invalid_params_examples() -> Vec<V<B, PARAMS>>;
}

pub trait ParamsImpl<const B: usize, const PARAMS: usize>: ParamsTestUtils<B, PARAMS> {
    fn are_params_valid(params: &V<B, PARAMS>) -> bool;
}

// pub trait Params<const B: usize, const PARAMS: usize, MaybeTape: SophusTape> {
//     fn from_params(params: GenV<B, PARAMS, MaybeTape>) -> Self;

//     fn into_params(self) -> GenV<B, PARAMS, MaybeTape> {
//         self.params
//     }
// }
