use dfdx_core::shapes::Unit;
use dfdx_core::tensor::NoneTape;

use crate::calculus::batch_types::*;
use crate::calculus::params::*;

pub trait TangentTestUtil<const B: usize, const DOF: usize> {
    fn tutil_tangent_examples() -> Vec<V<B, DOF>>;
}

pub trait ManifoldImpl<
    const B: usize,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT_DIM: usize,
    D: dfdx_core::tensor::Storage<f64>,
    Tape: dfdx_core::tensor::Tape<f64, D>,
>: ParamsImpl<B, PARAMS> + TangentTestUtil<B, DOF>
{
    fn oplus(params: &GenV<B, PARAMS, Tape>, tangent: &GenV<B, DOF, Tape>)
        -> GenV<B, PARAMS, Tape>;

    fn ominus(
        params1: &GenV<B, PARAMS, Tape>,
        params2: &GenV<B, PARAMS, Tape>,
    ) -> GenV<B, DOF, Tape>;

    fn test_suite() {
        let _params_examples = Self::tutil_params_examples();
        let _tangent_examples = Self::tutil_tangent_examples();

        // for (_params_id, params) in params_examples.iter().enumerate() {
        //     for (_tangent_id, delta) in tangent_examples.iter().enumerate() {
        //         let b = Self::oplus(params, delta);
        //         assert!((Self::ominus(params, &b) - delta).norm() < Into::<T>::into(1e-6));
        //     }
        // }
    }

    // fn sym_diff_quotient<TFn, const INPUT: usize>(
    //     vector_field: TFn,
    //     a: V<PARAMS>,
    //     eps: f64,
    // ) -> M<DOF, PARAMS>
    // where
    //     TFn: Fn(&V<PARAMS>) -> V<PARAMS>,
    // {
    //     let mut result = M::< DOF, PARAMS>::zeros();
    //     for i in 0..DOF {
    //         let mut h = V::zeros();
    //         h[i] = eps;
    //         let a_plus = Self::oplus(&a, &h);
    //         h[i] = -eps;
    //         let a_minus = Self::oplus(&a, &h);
    //         let diff =  Self::ominus(&vector_field(&a_plus), &vector_field(&a_minus));
    //         result.set_column(i, &(&diff / (Into::<T>::into(2.0) * eps)));
    //     }
    //     result
    // }
}

pub trait Manifold<const DOF: usize>: std::fmt::Debug + Sized + Clone {
    fn into_oplus(self, tangent: V<1, DOF>) -> Self;
}

impl Manifold<1> for V<1, 1> {
    fn into_oplus(self, tangent: V<1, 1>) -> Self {
        self + tangent
    }
}

impl Manifold<2> for V<1, 2> {
    fn into_oplus(self, tangent: V<1, 2>) -> Self {
        self + tangent
    }
}
impl Manifold<3> for V<1, 3> {
    fn into_oplus(self, tangent: V<1, 3>) -> Self {
        self + tangent
    }
}
