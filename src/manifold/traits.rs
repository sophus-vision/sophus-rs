use nalgebra::SVector;
type V<const N: usize> = SVector<f64, N>;
use crate::calculus;

pub trait TangentImpl<const DOF: usize> {
    fn tangent_examples() -> Vec<V<DOF>>;
}

pub trait ManifoldImpl<
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT_DIM: usize,
>: calculus::traits::ParamsImpl<PARAMS> + TangentImpl<DOF>
{
    fn oplus(params: &V<PARAMS>, tangent: &V<DOF>) -> V<PARAMS>;

    fn ominus(params1: &V<PARAMS>, params2: &V<PARAMS>) -> V<DOF>;

    fn test_suite() {
        let params_examples = Self::params_examples();
        let tangent_examples = Self::tangent_examples();

        for (_params_id, params) in params_examples.iter().enumerate() {
            for (_tangent_id, delta) in tangent_examples.iter().enumerate() {
                let b = Self::oplus(params, delta);
                assert!((Self::ominus(params, &b) - delta).norm() < 1e-6);
            }
        }
    }
}
