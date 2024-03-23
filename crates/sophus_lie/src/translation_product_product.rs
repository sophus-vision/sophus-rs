use std::vec;

use super::lie_group::LieGroup;
use super::traits::IsF64LieFactorGroupImpl;
use super::traits::IsF64LieGroupImpl;
use super::traits::IsLieFactorGroupImpl;

use super::traits::IsLieGroupImpl;
use super::traits::IsTranslationProductGroup;
use sophus_calculus::dual::dual_scalar::Dual;
use sophus_calculus::manifold;
use sophus_calculus::points::example_points;
use sophus_calculus::types::matrix::IsMatrix;
use sophus_calculus::types::params::HasParams;
use sophus_calculus::types::params::ParamsImpl;
use sophus_calculus::types::scalar::IsScalar;
use sophus_calculus::types::vector::IsVector;
use sophus_calculus::types::vector::IsVectorLike;
use sophus_calculus::types::MatF64;
use sophus_calculus::types::VecF64;

/// implementation of a translation product group
///
/// It is a semi-direct product of the commutative translation group (Euclidean vector space) and a factor group.
#[derive(Debug, Copy, Clone)]
pub struct TranslationProductGroupImpl<
    S: IsScalar<1>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const SDOF: usize,
    const SPARAMS: usize,
    F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, 1>,
> {
    phantom: std::marker::PhantomData<(S, F)>,
}

impl<
        S: IsScalar<1>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, 1>,
    > TranslationProductGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, F>
{
    /// translation part of the group parameters
    pub fn translation(params: &S::Vector<PARAMS>) -> S::Vector<POINT> {
        params.get_fixed_rows::<POINT>(0)
    }

    /// factor part of the group parameters
    pub fn factor_params(params: &S::Vector<PARAMS>) -> S::Vector<SPARAMS> {
        params.get_fixed_rows::<SPARAMS>(POINT)
    }

    /// create group parameters from translation and factor parameters
    pub fn params_from(
        translation: &S::Vector<POINT>,
        factor_params: &S::Vector<SPARAMS>,
    ) -> S::Vector<PARAMS> {
        S::Vector::block_vec2(translation.clone(), factor_params.clone())
    }

    /// translation part of the tangent vector
    fn translation_tangent(tangent: &S::Vector<DOF>) -> S::Vector<POINT> {
        tangent.get_fixed_rows::<POINT>(0)
    }

    /// factor part of the tangent vector
    fn factor_tangent(tangent: &S::Vector<DOF>) -> S::Vector<SDOF> {
        tangent.get_fixed_rows::<SDOF>(POINT)
    }

    /// create tangent vector from translation and factor tangent
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
        S: IsScalar<1>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, 1>,
    > ParamsImpl<S, PARAMS, 1>
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
        S: IsScalar<1>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, 1>,
    > manifold::traits::TangentImpl<S, DOF, 1>
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
        S: IsScalar<1>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        Factor: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, 1>,
    > IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, 1>
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

        let mat_v = Factor::mat_v(&Self::factor_tangent(omega));
        Self::params_from(&(mat_v * translation), &factor_params)
    }

    fn log(params: &S::Vector<PARAMS>) -> S::Vector<DOF> {
        let translation = Self::translation(params);

        let factor_params = Self::factor_params(params);

        let factor_tangent = Factor::log(&factor_params);
        let mat_v_inv = Factor::mat_v_inverse(&factor_tangent);
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

    type GenG<S2: IsScalar<1>> = TranslationProductGroupImpl<
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

    fn has_shortest_path_ambiguity(params: &<S as IsScalar<1>>::Vector<PARAMS>) -> bool {
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
        Factor: IsF64LieFactorGroupImpl<SDOF, SPARAMS, POINT>,
    > IsF64LieGroupImpl<DOF, PARAMS, POINT, AMBIENT>
    for TranslationProductGroupImpl<f64, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, Factor>
{
    fn dx_exp_x_at_0() -> MatF64<PARAMS, DOF> {
        MatF64::block_mat2x2::<POINT, SPARAMS, POINT, SDOF>(
            (
                MatF64::<POINT, POINT>::identity(),
                MatF64::<POINT, SDOF>::zero(),
            ),
            (MatF64::<SPARAMS, POINT>::zero(), Factor::dx_exp_x_at_0()),
        )
    }

    fn dx_exp_x_times_point_at_0(point: VecF64<POINT>) -> MatF64<POINT, DOF> {
        MatF64::block_mat1x2(
            MatF64::<POINT, POINT>::identity(),
            Factor::dx_exp_x_times_point_at_0(point),
        )
    }

    fn dx_exp(tangent: &VecF64<DOF>) -> MatF64<PARAMS, DOF> {
        let factor_tangent = &Self::factor_tangent(tangent);
        let trans_tangent = &Self::translation_tangent(tangent);

        let dx_mat_v = Factor::dx_mat_v(factor_tangent);
        let mut dx_mat_v_tangent = MatF64::<POINT, SDOF>::zero();

        for i in 0..SDOF {
            dx_mat_v_tangent
                .fixed_columns_mut::<1>(i)
                .copy_from(&(dx_mat_v[i] * trans_tangent));
        }

        MatF64::block_mat2x2::<POINT, SPARAMS, POINT, SDOF>(
            (Factor::mat_v(factor_tangent), dx_mat_v_tangent),
            (
                MatF64::<SPARAMS, POINT>::zero(),
                Factor::dx_exp(factor_tangent),
            ),
        )
    }

    fn dx_log_x(params: &VecF64<PARAMS>) -> MatF64<DOF, PARAMS> {
        let factor_params = &Self::factor_params(params);
        let trans = &Self::translation(params);
        let factor_tangent = Factor::log(factor_params);

        let dx_log_x = Factor::dx_log_x(factor_params);
        let dx_mat_v_inverse = Factor::dx_mat_v_inverse(&factor_tangent);

        let mut dx_mat_v_inv_tangent = MatF64::<POINT, SPARAMS>::zero();

        for i in 0..SDOF {
            let v = dx_mat_v_inverse[i] * trans;
            let r = dx_log_x.row(i);
            dx_mat_v_inv_tangent += v * r;
        }

        MatF64::block_mat2x2::<POINT, SDOF, POINT, SPARAMS>(
            (Factor::mat_v_inverse(&factor_tangent), dx_mat_v_inv_tangent),
            (MatF64::<SDOF, POINT>::zero(), dx_log_x),
        )
    }

    fn da_a_mul_b(a: &VecF64<PARAMS>, b: &VecF64<PARAMS>) -> MatF64<PARAMS, PARAMS> {
        let a_factor_params = &Self::factor_params(a);
        let b_factor_params = &Self::factor_params(b);

        let b_trans = &Self::translation(b);

        MatF64::block_mat2x2::<POINT, SPARAMS, POINT, SPARAMS>(
            (
                MatF64::<POINT, POINT>::identity(),
                Factor::dparams_matrix_times_point(a_factor_params, b_trans),
            ),
            (
                MatF64::<SPARAMS, POINT>::zero(),
                Factor::da_a_mul_b(a_factor_params, b_factor_params),
            ),
        )
    }

    fn db_a_mul_b(a: &VecF64<PARAMS>, b: &VecF64<PARAMS>) -> MatF64<PARAMS, PARAMS> {
        let a_factor_params = &Self::factor_params(a);
        let b_factor_params = &Self::factor_params(b);

        MatF64::block_mat2x2::<POINT, SPARAMS, POINT, SPARAMS>(
            (
                Factor::matrix(a_factor_params),
                MatF64::<POINT, SPARAMS>::zero(),
            ),
            (
                MatF64::<SPARAMS, POINT>::zero(),
                Factor::db_a_mul_b(a_factor_params, b_factor_params),
            ),
        )
    }
}

impl<
        S: IsScalar<1>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        FactorImpl: crate::traits::IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, 1>,
    >
    IsTranslationProductGroup<
        S,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        1,
        LieGroup<S, SDOF, SPARAMS, POINT, POINT, 1, FactorImpl>,
    >
    for crate::lie_group::LieGroup<
        S,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        1,
        TranslationProductGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, FactorImpl>,
    >
{
    type Impl =
        TranslationProductGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, FactorImpl>;

    fn from_translation_and_factor(
        translation: &<S as IsScalar<1>>::Vector<POINT>,
        factor: &LieGroup<S, SDOF, SPARAMS, POINT, POINT, 1, FactorImpl>,
    ) -> Self {
        let params = Self::Impl::params_from(translation, factor.params());
        Self::from_params(&params)
    }

    fn from_t(translation: &<S as IsScalar<1>>::Vector<POINT>) -> Self {
        Self::from_translation_and_factor(translation, &LieGroup::identity())
    }

    fn set_translation(&mut self, translation: &<S as IsScalar<1>>::Vector<POINT>) {
        self.set_params(&Self::G::params_from(translation, self.factor().params()))
    }

    fn translation(&self) -> <S as IsScalar<1>>::Vector<POINT> {
        Self::Impl::translation(self.params())
    }

    fn set_factor(&mut self, factor: &LieGroup<S, SDOF, SPARAMS, POINT, POINT, 1, FactorImpl>) {
        self.set_params(&Self::G::params_from(&self.translation(), factor.params()))
    }

    fn factor(&self) -> LieGroup<S, SDOF, SPARAMS, POINT, POINT, 1, FactorImpl> {
        LieGroup::from_params(&Self::Impl::factor_params(self.params()))
    }
}
