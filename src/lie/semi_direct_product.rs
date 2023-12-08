use std::vec;

use super::lie_group::LieGroup;
use super::traits::IsF64LieGroupImpl;
use super::traits::IsLieFactorGroupImpl;
use super::traits::IsLieGroup;
use super::traits::IsLieGroupImpl;
use super::traits::IsTranslationProductGroup;
use crate::calculus::dual::dual_scalar::Dual;
use crate::calculus::points::example_points;
use crate::calculus::types::matrix::IsMatrix;
use crate::calculus::types::params::HasParams;
use crate::calculus::types::params::ParamsImpl;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::IsVector;
use crate::calculus::types::vector::IsVectorLike;
use crate::calculus::types::M;
use crate::calculus::types::V;
use crate::lie;
use crate::manifold::{self};

#[derive(Debug, Copy, Clone)]
pub struct TranslationProductGroupImpl<
    S: IsScalar,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const SDOF: usize,
    const SPARAMS: usize,
    F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, POINT>,
> {
    phantom: std::marker::PhantomData<(S, F)>,
}

impl<
        S: IsScalar,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, POINT>,
    > TranslationProductGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, F>
{
    pub fn translation(params: &S::Vector<PARAMS>) -> S::Vector<POINT> {
        params.get_fixed_rows::<POINT>(0)
    }

    pub fn factor_params(params: &S::Vector<PARAMS>) -> S::Vector<SPARAMS> {
        params.get_fixed_rows::<SPARAMS>(POINT)
    }

    pub fn params_from(
        translation: &S::Vector<POINT>,
        factor_params: &S::Vector<SPARAMS>,
    ) -> S::Vector<PARAMS> {
        S::Vector::block_vec2(translation.clone(), factor_params.clone())
    }

    fn translation_tangent(tangent: &S::Vector<DOF>) -> S::Vector<POINT> {
        tangent.get_fixed_rows::<POINT>(0)
    }

    fn factor_tangent(tangent: &S::Vector<DOF>) -> S::Vector<SDOF> {
        tangent.get_fixed_rows::<SDOF>(POINT)
    }

    fn tangent_from(
        translation: &S::Vector<POINT>,
        factor_tangent: &S::Vector<SDOF>,
    ) -> S::Vector<DOF> {
        S::Vector::block_vec2(translation.clone(), factor_tangent.clone())
    }

    fn translation_examples() -> Vec<S::Vector<POINT>> {
        example_points::<S, POINT>()
    }
}

impl<
        S: IsScalar,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, POINT>,
    > ParamsImpl<S, PARAMS>
    for TranslationProductGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, F>
{
    fn are_params_valid(params: &S::Vector<PARAMS>) -> bool {
        F::are_params_valid(&Self::factor_params(params))
    }

    fn params_examples() -> Vec<S::Vector<PARAMS>> {
        let mut examples = vec![];
        for factor_params in F::params_examples() {
            for translation in Self::translation_examples() {
                examples.push(Self::params_from(&translation, &factor_params));
            }
        }
        examples
    }

    fn invalid_params_examples() -> Vec<S::Vector<PARAMS>> {
        vec![Self::params_from(
            &S::Vector::zero(),
            &F::invalid_params_examples()[0],
        )]
    }
}

impl<
        S: IsScalar,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, POINT>,
    > manifold::traits::TangentImpl<S, DOF>
    for TranslationProductGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, F>
{
    fn tangent_examples() -> Vec<S::Vector<DOF>> {
        let mut examples = vec![];
        for group_tangent in F::tangent_examples() {
            for translation_tangent in Self::translation_examples() {
                examples.push(Self::tangent_from(&translation_tangent, &group_tangent));
            }
        }
        examples
    }
}

// TODO : Port to Rust

impl<
        S: IsScalar,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        Factor: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, POINT>,
    > IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT>
    for TranslationProductGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, Factor>
{
    const IS_ORIGIN_PRESERVING: bool = false;
    const IS_AXIS_DIRECTION_PRESERVING: bool = Factor::IS_AXIS_DIRECTION_PRESERVING;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = Factor::IS_DIRECTION_VECTOR_PRESERVING;
    const IS_SHAPE_PRESERVING: bool = Factor::IS_SHAPE_PRESERVING;
    const IS_DISTANCE_PRESERVING: bool = Factor::IS_DISTANCE_PRESERVING;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    fn identity_params() -> S::Vector<PARAMS> {
        Self::params_from(&S::Vector::zero(), &Factor::identity_params())
    }

    //    Manifold / Lie Group concepts

    fn adj(params: &S::Vector<PARAMS>) -> S::Matrix<DOF, DOF> {
        let factor_params = Self::factor_params(params);
        let translation = Self::translation(params);

        S::Matrix::block_mat2x2::<POINT, SDOF, POINT, SDOF>(
            (
                Factor::matrix(&factor_params),
                Factor::adj_of_translation(&factor_params, &translation),
            ),
            (S::Matrix::zero(), Factor::adj(&factor_params)),
        )
    }

    fn exp(omega: &S::Vector<DOF>) -> S::Vector<PARAMS> {
        let translation = Self::translation_tangent(omega);
        let factor_params = Factor::exp(&Self::factor_tangent(omega));

        let mat_v = Factor::mat_v(&factor_params, &Self::factor_tangent(omega));
        Self::params_from(&(mat_v * translation), &factor_params)
    }

    fn log(params: &S::Vector<PARAMS>) -> S::Vector<DOF> {
        let translation = Self::translation(params);

        let factor_params = Self::factor_params(params);

        let factor_tangent = Factor::log(&factor_params);
        let mat_v_inv = Factor::mat_v_inverse(&factor_params, &factor_tangent);
        let translation_tangent = mat_v_inv * translation;

        Self::tangent_from(&translation_tangent, &factor_tangent)
    }

    fn hat(omega: &S::Vector<DOF>) -> S::Matrix<AMBIENT, AMBIENT> {
        S::Matrix::block_mat2x2::<POINT, 1, POINT, 1>(
            (
                Factor::hat(&Self::factor_tangent(omega)),
                Self::translation_tangent(omega).to_mat(),
            ),
            (S::Matrix::zero(), S::Matrix::zero()),
        )
    }

    fn vee(hat: &S::Matrix<AMBIENT, AMBIENT>) -> S::Vector<DOF> {
        let factor_tangent = Factor::vee(&hat.get_fixed_submat::<POINT, POINT>(0, 0));
        let translation_tangent = hat.get_fixed_submat::<POINT, 1>(0, POINT);
        Self::tangent_from(&translation_tangent.get_col_vec(0), &factor_tangent)
    }

    // group operations

    fn group_mul(params1: &S::Vector<PARAMS>, params2: &S::Vector<PARAMS>) -> S::Vector<PARAMS> {
        let factor_params1 = Self::factor_params(params1);
        let factor_params2 = Self::factor_params(params2);
        let translation1 = Self::translation(params1);
        let translation2 = Self::translation(params2);
        let factor_params = Factor::group_mul(&factor_params1, &factor_params2);
        let f = Factor::transform(&factor_params1, &translation2);
        let translation = f + translation1;
        Self::params_from(&translation, &factor_params)
    }

    fn inverse(params: &S::Vector<PARAMS>) -> S::Vector<PARAMS> {
        let factor_params = Self::factor_params(params);
        let translation = Self::translation(params);
        let factor_params = Factor::inverse(&factor_params);
        let translation = -Factor::transform(&factor_params, &translation);
        Self::params_from(&translation, &factor_params)
    }

    fn transform(params: &S::Vector<PARAMS>, point: &S::Vector<POINT>) -> S::Vector<POINT> {
        let factor_params = Self::factor_params(params);
        let translation = Self::translation(params);
        Factor::transform(&factor_params, point) + translation
    }

    fn to_ambient(params: &S::Vector<POINT>) -> S::Vector<AMBIENT> {
        S::Vector::block_vec2(params.clone(), S::Vector::<1>::zero())
    }

    fn compact(params: &S::Vector<PARAMS>) -> S::Matrix<POINT, AMBIENT> {
        S::Matrix::block_mat1x2::<POINT, 1>(
            Factor::matrix(&Self::factor_params(params)),
            Self::translation(params).to_mat(),
        )
    }

    fn matrix(params: &S::Vector<PARAMS>) -> S::Matrix<AMBIENT, AMBIENT> {
        S::Matrix::block_mat2x2::<POINT, 1, POINT, 1>(
            (
                Factor::matrix(&Self::factor_params(params)),
                Self::translation(params).to_mat(),
            ),
            (S::Matrix::<1, POINT>::zero(), S::Matrix::<1, 1>::identity()),
        )
    }

    fn ad(tangent: &S::Vector<DOF>) -> S::Matrix<DOF, DOF> {
        let o = S::Matrix::<SDOF, POINT>::zero();
        S::Matrix::block_mat2x2::<POINT, SDOF, POINT, SDOF>(
            (
                Factor::hat(&Self::factor_tangent(tangent)),
                Factor::ad_of_translation(&Self::translation_tangent(tangent)),
            ),
            (o, Factor::ad(&Self::factor_tangent(tangent))),
        )
    }

    type GenG<S2: IsScalar> = TranslationProductGroupImpl<
        S2,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        Factor::GenFactorG<S2>,
    >;

    type RealG = TranslationProductGroupImpl<
        f64,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        Factor::GenFactorG<f64>,
    >;

    type DualG = TranslationProductGroupImpl<
        Dual,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        Factor::GenFactorG<Dual>,
    >;

    fn has_shortest_path_ambiguity(params: &<S as IsScalar>::Vector<PARAMS>) -> bool {
        Factor::has_shortest_path_ambiguity(&Self::factor_params(params))
    }
}

impl<
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        Factor: IsLieFactorGroupImpl<f64, SDOF, SPARAMS, POINT, POINT>
            + IsF64LieGroupImpl<SDOF, SPARAMS, POINT, POINT>,
    > IsF64LieGroupImpl<DOF, PARAMS, POINT, AMBIENT>
    for TranslationProductGroupImpl<f64, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, Factor>
{
    fn dx_exp_x_at_0() -> M<PARAMS, DOF> {
        M::block_mat2x2::<POINT, SPARAMS, POINT, SDOF>(
            (M::<POINT, POINT>::identity(), M::<POINT, SDOF>::zero()),
            (M::<SPARAMS, POINT>::zero(), Factor::dx_exp_x_at_0()),
        )
    }

    fn dx_exp_x_times_point_at_0(point: V<POINT>) -> M<POINT, DOF> {
        M::block_mat1x2(
            M::<POINT, POINT>::identity(),
            Factor::dx_exp_x_times_point_at_0(point),
        )
    }
}

impl<
        S: IsScalar,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        FactorImpl: lie::traits::IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, POINT>,
    >
    IsTranslationProductGroup<
        S,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        LieGroup<S, SDOF, SPARAMS, POINT, POINT, FactorImpl>,
    >
    for lie::lie_group::LieGroup<
        S,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        TranslationProductGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, FactorImpl>,
    >
{
    type Impl =
        TranslationProductGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, FactorImpl>;

    fn from_translation_and_factor(
        translation: &<S as IsScalar>::Vector<POINT>,
        factor: &LieGroup<S, SDOF, SPARAMS, POINT, POINT, FactorImpl>,
    ) -> Self {
        let params = Self::Impl::params_from(translation, factor.params());
        Self::from_params(&params)
    }

    fn set_translation(&mut self, translation: &<S as IsScalar>::Vector<POINT>) {
        self.set_params(&Self::G::params_from(translation, self.factor().params()))
    }

    fn translation(&self) -> <S as IsScalar>::Vector<POINT> {
        Self::Impl::translation(self.params())
    }

    fn set_factor(&mut self, factor: &LieGroup<S, SDOF, SPARAMS, POINT, POINT, FactorImpl>) {
        self.set_params(&Self::G::params_from(&self.translation(), factor.params()))
    }

    fn factor(&self) -> LieGroup<S, SDOF, SPARAMS, POINT, POINT, FactorImpl> {
        LieGroup::from_params(&Self::Impl::factor_params(self.params()))
    }
}
