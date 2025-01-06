use core::borrow::Borrow;

use crate::lie_group::LieGroup;
use crate::prelude::*;
use crate::traits::HasDisambiguate;
use crate::traits::IsLieFactorGroupImpl;
use crate::traits::IsLieGroupImpl;
use crate::traits::IsRealLieFactorGroupImpl;
use crate::traits::IsRealLieGroupImpl;
use sophus_autodiff::manifold::IsTangent;
use sophus_autodiff::params::IsParamsImpl;
use sophus_autodiff::points::example_points;

extern crate alloc;

/// implementation of a translation product group
///
/// It is a semi-direct product of the commutative translation group (Euclidean vector space) and a factor group.
#[derive(Debug, Copy, Clone, Default)]
pub struct TranslationProductGroupImpl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const SDOF: usize,
    const SPARAMS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
    F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, BATCH, DM, DN>,
> {
    phantom: core::marker::PhantomData<(S, F)>,
}

impl<
        S: IsScalar<BATCH, DM, DN>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
        F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, BATCH, DM, DN>,
    > TranslationProductGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, BATCH, DM, DN, F>
{
    /// translation part of the group parameters
    pub fn translation(params: &S::Vector<PARAMS>) -> S::Vector<POINT> {
        params.get_fixed_subvec::<POINT>(0)
    }

    /// factor part of the group parameters
    pub fn factor_params(params: &S::Vector<PARAMS>) -> S::Vector<SPARAMS> {
        params.get_fixed_subvec::<SPARAMS>(POINT)
    }

    /// create group parameters from translation and factor parameters
    pub fn params_from(
        translation: &S::Vector<POINT>,
        factor_params: &S::Vector<SPARAMS>,
    ) -> S::Vector<PARAMS> {
        S::Vector::block_vec2(*translation, *factor_params)
    }

    /// translation part of the tangent vector
    fn translation_tangent(tangent: &S::Vector<DOF>) -> S::Vector<POINT> {
        tangent.get_fixed_subvec::<POINT>(0)
    }

    /// factor part of the tangent vector
    fn factor_tangent(tangent: &S::Vector<DOF>) -> S::Vector<SDOF> {
        tangent.get_fixed_subvec::<SDOF>(POINT)
    }

    /// create tangent vector from translation and factor tangent
    fn tangent_from(
        translation: &S::Vector<POINT>,
        factor_tangent: &S::Vector<SDOF>,
    ) -> S::Vector<DOF> {
        S::Vector::block_vec2(*translation, *factor_tangent)
    }

    fn translation_examples() -> alloc::vec::Vec<S::Vector<POINT>> {
        example_points::<S, POINT, BATCH, DM, DN>()
    }
}

impl<
        S: IsScalar<BATCH, DM, DN>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
        F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, BATCH, DM, DN>,
    > HasDisambiguate<S, PARAMS, BATCH, DM, DN>
    for TranslationProductGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, BATCH, DM, DN, F>
{
    fn disambiguate(params: S::Vector<PARAMS>) -> S::Vector<PARAMS> {
        Self::params_from(
            &Self::translation(&params),
            &F::disambiguate(Self::factor_params(&params)),
        )
    }
}

impl<
        S: IsScalar<BATCH, DM, DN>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
        F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, BATCH, DM, DN>,
    > IsParamsImpl<S, PARAMS, BATCH, DM, DN>
    for TranslationProductGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, BATCH, DM, DN, F>
{
    fn are_params_valid<P>(params: P) -> S::Mask
    where
        P: Borrow<S::Vector<PARAMS>>,
    {
        F::are_params_valid(Self::factor_params(params.borrow()))
    }

    fn params_examples() -> alloc::vec::Vec<S::Vector<PARAMS>> {
        let mut examples = alloc::vec![];

        let factor_examples = F::params_examples();
        let translation_examples = Self::translation_examples();

        // Determine the maximum length of factor and translation examples
        let max_len = core::cmp::max(factor_examples.len(), translation_examples.len());

        for i in 0..max_len {
            // Wrap around indices if one vector is shorter than the other
            let factor_params = &factor_examples[i % factor_examples.len()];
            let translation = &translation_examples[i % translation_examples.len()];

            examples.push(Self::params_from(translation, factor_params));
        }
        examples
    }

    fn invalid_params_examples() -> alloc::vec::Vec<S::Vector<PARAMS>> {
        alloc::vec![Self::params_from(
            &S::Vector::zeros(),
            &F::invalid_params_examples()[0],
        )]
    }
}

impl<
        S: IsScalar<BATCH, DM, DN>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
        F: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, BATCH, DM, DN>,
    > IsTangent<S, DOF, BATCH, DM, DN>
    for TranslationProductGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, BATCH, DM, DN, F>
{
    fn tangent_examples() -> alloc::vec::Vec<S::Vector<DOF>> {
        let mut examples = alloc::vec![];

        let factor_examples = F::tangent_examples();
        let translation_examples = Self::translation_examples();

        // Determine the maximum length of factor and translation examples
        let max_len = core::cmp::max(factor_examples.len(), translation_examples.len());

        for i in 0..max_len {
            // Wrap around indices if one vector is shorter than the other
            let factor_params = &factor_examples[i % factor_examples.len()];
            let translation = &translation_examples[i % translation_examples.len()];
            examples.push(Self::tangent_from(translation, factor_params));
        }
        examples
    }
}

impl<
        S: IsScalar<BATCH, DM, DN>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
        Factor: IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, BATCH, DM, DN>,
    > IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN>
    for TranslationProductGroupImpl<
        S,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        BATCH,
        DM,
        DN,
        Factor,
    >
{
    const IS_ORIGIN_PRESERVING: bool = false;
    const IS_AXIS_DIRECTION_PRESERVING: bool = Factor::IS_AXIS_DIRECTION_PRESERVING;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = Factor::IS_DIRECTION_VECTOR_PRESERVING;
    const IS_SHAPE_PRESERVING: bool = Factor::IS_SHAPE_PRESERVING;
    const IS_DISTANCE_PRESERVING: bool = Factor::IS_DISTANCE_PRESERVING;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    fn identity_params() -> S::Vector<PARAMS> {
        Self::params_from(&S::Vector::zeros(), &Factor::identity_params())
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
            (S::Matrix::zeros(), Factor::adj(&factor_params)),
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
            (S::Matrix::zeros(), S::Matrix::zeros()),
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
        S::Vector::block_vec2(*params, S::Vector::<1>::zeros())
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
            (
                S::Matrix::<1, POINT>::zeros(),
                S::Matrix::<1, 1>::identity(),
            ),
        )
    }

    fn ad(tangent: &S::Vector<DOF>) -> S::Matrix<DOF, DOF> {
        let o = S::Matrix::<SDOF, POINT>::zeros();
        S::Matrix::block_mat2x2::<POINT, SDOF, POINT, SDOF>(
            (
                Factor::hat(&Self::factor_tangent(tangent)),
                Factor::ad_of_translation(&Self::translation_tangent(tangent)),
            ),
            (o, Factor::ad(&Self::factor_tangent(tangent))),
        )
    }

    type GenG<S2: IsScalar<BATCH, DM, DN>> = TranslationProductGroupImpl<
        S2,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        BATCH,
        DM,
        DN,
        Factor::GenFactorG<S2, DM, DN>,
    >;

    type RealG = TranslationProductGroupImpl<
        S::RealScalar,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        BATCH,
        0,
        0,
        Factor::GenFactorG<S::RealScalar, 0, 0>,
    >;

    type DualG<const M: usize, const N: usize> = TranslationProductGroupImpl<
        S::DualScalar<M, N>,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        BATCH,
        M,
        N,
        Factor::GenFactorG<S::DualScalar<M, N>, M, N>,
    >;
}

impl<
        S: IsRealScalar<BATCH>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        const BATCH: usize,
        Factor: IsRealLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, BATCH>,
    > IsRealLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH>
    for TranslationProductGroupImpl<
        S,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        BATCH,
        0,
        0,
        Factor,
    >
{
    fn dx_exp_x_at_0() -> S::Matrix<PARAMS, DOF> {
        S::Matrix::block_mat2x2::<POINT, SPARAMS, POINT, SDOF>(
            (
                S::Matrix::<POINT, POINT>::identity(),
                S::Matrix::<POINT, SDOF>::zeros(),
            ),
            (
                S::Matrix::<SPARAMS, POINT>::zeros(),
                Factor::dx_exp_x_at_0(),
            ),
        )
    }

    fn dx_exp_x_times_point_at_0(point: &S::Vector<POINT>) -> S::Matrix<POINT, DOF> {
        S::Matrix::block_mat1x2(
            S::Matrix::<POINT, POINT>::identity(),
            Factor::dx_exp_x_times_point_at_0(point),
        )
    }

    fn dx_exp(tangent: &S::Vector<DOF>) -> S::Matrix<PARAMS, DOF> {
        let factor_tangent = &Self::factor_tangent(tangent);
        let trans_tangent = &Self::translation_tangent(tangent);

        let dx_mat_v = Factor::dx_mat_v(factor_tangent);
        let mut dx_mat_v_tangent = S::Matrix::<POINT, SDOF>::zeros();

        for i in 0..SDOF {
            dx_mat_v_tangent.set_col_vec(i, dx_mat_v[i] * *trans_tangent);
        }

        S::Matrix::block_mat2x2::<POINT, SPARAMS, POINT, SDOF>(
            (Factor::mat_v(factor_tangent), dx_mat_v_tangent),
            (
                S::Matrix::<SPARAMS, POINT>::zeros(),
                Factor::dx_exp(factor_tangent),
            ),
        )
    }

    fn dx_log_x(params: &S::Vector<PARAMS>) -> S::Matrix<DOF, PARAMS> {
        let factor_params = &Self::factor_params(params);
        let trans = &Self::translation(params);
        let factor_tangent = Factor::log(factor_params);

        let dx_log_x = Factor::dx_log_x(factor_params);
        let dx_mat_v_inverse = Factor::dx_mat_v_inverse(&factor_tangent);

        let mut dx_mat_v_inv_tangent = S::Matrix::<POINT, SPARAMS>::zeros();

        for i in 0..SDOF {
            let v: S::Vector<POINT> = dx_mat_v_inverse[i] * *trans;
            let r: S::Vector<SPARAMS> = dx_log_x.get_row_vec(i);

            let m = v.outer(r);
            dx_mat_v_inv_tangent = dx_mat_v_inv_tangent + m;
        }

        S::Matrix::block_mat2x2::<POINT, SDOF, POINT, SPARAMS>(
            (Factor::mat_v_inverse(&factor_tangent), dx_mat_v_inv_tangent),
            (S::Matrix::<SDOF, POINT>::zeros(), dx_log_x),
        )
    }

    fn da_a_mul_b(a: &S::Vector<PARAMS>, b: &S::Vector<PARAMS>) -> S::Matrix<PARAMS, PARAMS> {
        let a_factor_params = &Self::factor_params(a);
        let b_factor_params = &Self::factor_params(b);

        let b_trans = &Self::translation(b);

        S::Matrix::block_mat2x2::<POINT, SPARAMS, POINT, SPARAMS>(
            (
                S::Matrix::<POINT, POINT>::identity(),
                Factor::dparams_matrix_times_point(a_factor_params, b_trans),
            ),
            (
                S::Matrix::<SPARAMS, POINT>::zeros(),
                Factor::da_a_mul_b(a_factor_params, b_factor_params),
            ),
        )
    }

    fn db_a_mul_b(a: &S::Vector<PARAMS>, b: &S::Vector<PARAMS>) -> S::Matrix<PARAMS, PARAMS> {
        let a_factor_params = &Self::factor_params(a);
        let b_factor_params = &Self::factor_params(b);

        S::Matrix::block_mat2x2::<POINT, SPARAMS, POINT, SPARAMS>(
            (
                Factor::matrix(a_factor_params),
                S::Matrix::<POINT, SPARAMS>::zeros(),
            ),
            (
                S::Matrix::<SPARAMS, POINT>::zeros(),
                Factor::db_a_mul_b(a_factor_params, b_factor_params),
            ),
        )
    }

    fn has_shortest_path_ambiguity(params: &<S>::Vector<PARAMS>) -> <S>::Mask {
        Factor::has_shortest_path_ambiguity(&Self::factor_params(params))
    }

    fn dparams_matrix(params: &<S>::Vector<PARAMS>, col_idx: usize) -> <S>::Matrix<POINT, PARAMS> {
        let factor_params = &Self::factor_params(params);

        if col_idx < POINT {
            S::Matrix::block_mat1x2::<POINT, SPARAMS>(
                S::Matrix::zeros(),
                Factor::dparams_matrix(factor_params, col_idx),
            )
        } else {
            S::Matrix::block_mat1x2::<POINT, SPARAMS>(S::Matrix::identity(), S::Matrix::zeros())
        }
    }
}

impl<
        S: IsScalar<BATCH, DM, DN>,
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        const BATCH: usize,
        const DM: usize,
        const DN: usize,
        FactorImpl: crate::traits::IsLieFactorGroupImpl<S, SDOF, SPARAMS, POINT, BATCH, DM, DN>,
    >
    IsTranslationProductGroup<
        S,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        BATCH,
        DM,
        DN,
        LieGroup<S, SDOF, SPARAMS, POINT, POINT, BATCH, DM, DN, FactorImpl>,
    >
    for crate::lie_group::LieGroup<
        S,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        BATCH,
        DM,
        DN,
        TranslationProductGroupImpl<
            S,
            DOF,
            PARAMS,
            POINT,
            AMBIENT,
            SDOF,
            SPARAMS,
            BATCH,
            DM,
            DN,
            FactorImpl,
        >,
    >
{
    type Impl = TranslationProductGroupImpl<
        S,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        BATCH,
        DM,
        DN,
        FactorImpl,
    >;

    fn from_translation_and_factor<P, F>(translation: P, factor: F) -> Self
    where
        P: Borrow<S::Vector<POINT>>,
        F: Borrow<LieGroup<S, SDOF, SPARAMS, POINT, POINT, BATCH, DM, DN, FactorImpl>>,
    {
        let params = Self::Impl::params_from(translation.borrow(), factor.borrow().params());
        Self::from_params(params)
    }

    fn set_translation<P>(&mut self, translation: P)
    where
        P: Borrow<<S as sophus_autodiff::prelude::IsScalar<BATCH, DM, DN>>::Vector<POINT>>,
    {
        self.set_params(Self::G::params_from(
            translation.borrow(),
            self.factor().params(),
        ))
    }

    fn translation(&self) -> <S as IsScalar<BATCH, DM, DN>>::Vector<POINT> {
        Self::Impl::translation(self.params())
    }

    fn set_factor<F>(&mut self, factor: F)
    where
        F: Borrow<LieGroup<S, SDOF, SPARAMS, POINT, POINT, BATCH, DM, DN, FactorImpl>>,
    {
        self.set_params(Self::G::params_from(
            &self.translation(),
            factor.borrow().params(),
        ))
    }

    fn factor(&self) -> LieGroup<S, SDOF, SPARAMS, POINT, POINT, BATCH, DM, DN, FactorImpl> {
        LieGroup::from_params(Self::Impl::factor_params(self.params()))
    }
}
