use std::vec;

use nalgebra::{SMatrix, SVector};

use crate::{
    calculus::{self, points::example_points},
    lie, manifold,
};

use super::traits::LieSubgroupImplTrait;

type V<const N: usize> = SVector<f64, N>;
type M<const N: usize, const O: usize> = SMatrix<f64, N, O>;

#[derive(Debug, Copy, Clone)]
pub struct SemiDirectProductImpl<
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const SDOF: usize,
    const SPARAMS: usize,
    S: LieSubgroupImplTrait<SDOF, SPARAMS, POINT, POINT>,
> {
    phantom: std::marker::PhantomData<S>,
}

impl<
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        S: LieSubgroupImplTrait<SDOF, SPARAMS, POINT, POINT>,
    > SemiDirectProductImpl<DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, S>
{
    fn translation(params: &V<PARAMS>) -> V<POINT> {
        params.fixed_rows::<POINT>(0).into_owned()
    }

    fn subgroup_params(params: &V<PARAMS>) -> V<SPARAMS> {
        params.fixed_rows::<SPARAMS>(POINT).into_owned()
    }

    pub fn params_from(translation: &V<POINT>, subgroup_params: &V<SPARAMS>) -> V<PARAMS> {
        let mut params = V::<PARAMS>::zeros();
        params
            .fixed_view_mut::<POINT, 1>(0, 0)
            .copy_from(&translation);
        params
            .fixed_view_mut::<SPARAMS, 1>(POINT, 0)
            .copy_from(subgroup_params);
        params
    }

    fn translation_tangent(tangent: &V<DOF>) -> V<POINT> {
        tangent.fixed_rows::<POINT>(0).into_owned()
    }

    fn subgroup_tangent(tangent: &V<DOF>) -> V<SDOF> {
        tangent.fixed_rows::<SDOF>(POINT).into_owned()
    }

    fn tangent_from(translation: &V<POINT>, subgroup_tangent: &V<SDOF>) -> V<DOF> {
        let mut tangent = V::<DOF>::zeros();
        tangent
            .fixed_view_mut::<POINT, 1>(0, 0)
            .copy_from(&translation);
        tangent
            .fixed_view_mut::<SDOF, 1>(POINT, 0)
            .copy_from(subgroup_tangent);
        tangent
    }

    fn translation_examples() -> Vec<V<POINT>> {
        example_points::<POINT>()
    }
}

impl<
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        S: LieSubgroupImplTrait<SDOF, SPARAMS, POINT, POINT>,
    > calculus::traits::ParamsImpl<PARAMS>
    for SemiDirectProductImpl<DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, S>
{
    fn are_params_valid(params: &V<PARAMS>) -> bool {
        S::are_params_valid(&Self::subgroup_params(params))
    }

    fn params_examples() -> Vec<V<PARAMS>> {
        let mut examples = vec![];
        for subgroup_params in S::params_examples() {
            for translation in Self::translation_examples() {
                examples.push(Self::params_from(&translation, &subgroup_params));
            }
        }
        examples
    }

    fn invalid_params_examples() -> Vec<V<PARAMS>> {
        vec![Self::params_from(
            &V::zeros(),
            &S::invalid_params_examples()[0],
        )]
    }
}

impl<
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        S: LieSubgroupImplTrait<SDOF, SPARAMS, POINT, POINT>,
    > manifold::traits::TangentImpl<DOF>
    for SemiDirectProductImpl<DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, S>
{
    fn tangent_examples() -> Vec<V<DOF>> {
        let mut examples = vec![];
        for group_tangent in S::tangent_examples() {
            for translation_tangent in Self::translation_examples() {
                examples.push(Self::tangent_from(&translation_tangent, &group_tangent));
            }
        }
        examples
    }
}

// TODO : Port to Rust

impl<
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        S: LieSubgroupImplTrait<SDOF, SPARAMS, POINT, POINT>,
    > lie::traits::LieGroupImpl<DOF, PARAMS, POINT, AMBIENT>
    for SemiDirectProductImpl<DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, S>
{
    const IS_ORIGIN_PRESERVING: bool = false;
    const IS_AXIS_DIRECTION_PRESERVING: bool = S::IS_AXIS_DIRECTION_PRESERVING;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = S::IS_DIRECTION_VECTOR_PRESERVING;
    const IS_SHAPE_PRESERVING: bool = S::IS_SHAPE_PRESERVING;
    const IS_DISTANCE_PRESERVING: bool = S::IS_DISTANCE_PRESERVING;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    fn identity_params() -> V<PARAMS> {
        Self::params_from(&V::zeros(), &S::identity_params())
    }

    //    Manifold / Lie Group concepts

    fn adj(params: &V<PARAMS>) -> M<DOF, DOF> {
        let mut mat_adjoint = M::<DOF, DOF>::zeros();
        let subgroup_params = Self::subgroup_params(params);
        let translation = Self::translation(params);

        // top row
        mat_adjoint
            .fixed_view_mut::<POINT, POINT>(0, 0)
            .copy_from(&S::matrix(&subgroup_params));
        mat_adjoint
            .fixed_view_mut::<POINT, SDOF>(0, POINT)
            .copy_from(&S::adj_of_translation(&subgroup_params, &translation));

        // bottom row
        mat_adjoint
            .fixed_view_mut::<SDOF, POINT>(POINT, 0)
            .copy_from(&M::zeros());
        mat_adjoint
            .fixed_view_mut::<SDOF, SDOF>(POINT, POINT)
            .copy_from(&S::adj(&subgroup_params));

        mat_adjoint
    }

    fn exp(omega: &V<DOF>) -> V<PARAMS> {
        let translation = Self::translation_tangent(omega);
        let subgroup_params = S::exp(&Self::subgroup_tangent(omega));

        let mat_v = S::mat_v(&subgroup_params, &Self::subgroup_tangent(omega));
        Self::params_from(&(&mat_v * &translation), &subgroup_params)
    }

    fn log(params: &V<PARAMS>) -> V<DOF> {
        let translation = Self::translation(params);
        let subgroup_params = Self::subgroup_params(params);

        let subgroup_tangent = S::log(&subgroup_params);
        let mat_v_inv = S::mat_v_inverse(&subgroup_params, &subgroup_tangent);
        let translation_tangent = &mat_v_inv * &translation;

        Self::tangent_from(&translation_tangent, &subgroup_tangent)
    }

    fn hat(omega: &V<DOF>) -> M<AMBIENT, AMBIENT> {
        let mut hat_mat = M::<AMBIENT, AMBIENT>::zeros();
        hat_mat
            .view_mut((0, 0), (POINT, POINT))
            .copy_from(&S::hat(&Self::subgroup_tangent(omega)));
        hat_mat
            .view_mut((0, POINT), (POINT, 1))
            .copy_from(&Self::translation_tangent(omega));
        hat_mat
    }

    fn vee(hat: &M<AMBIENT, AMBIENT>) -> V<DOF> {
        let subgroup_tangent = S::vee(&hat.fixed_view::<POINT, POINT>(0, 0).into_owned());
        let translation_tangent = hat.fixed_view::<POINT, 1>(0, POINT).into_owned();
        Self::tangent_from(&translation_tangent, &subgroup_tangent)
    }

    // group operations

    fn multiply(params1: &V<PARAMS>, params2: &V<PARAMS>) -> V<PARAMS> {
        let subgroup_params1 = Self::subgroup_params(params1);
        let subgroup_params2 = Self::subgroup_params(params2);
        let translation1 = Self::translation(params1);
        let translation2 = Self::translation(params2);
        let subgroup_params = S::multiply(&subgroup_params1, &subgroup_params2);
        let translation = &S::transform(&subgroup_params1, &translation2) + &translation1;
        Self::params_from(&translation, &subgroup_params)
    }

    fn inverse(params: &V<PARAMS>) -> V<PARAMS> {
        let subgroup_params = Self::subgroup_params(params);
        let translation = Self::translation(params);
        let subgroup_params = S::inverse(&subgroup_params);
        let translation = -&S::transform(&subgroup_params, &translation);
        Self::params_from(&translation, &subgroup_params)
    }

    fn transform(params: &V<PARAMS>, point: &V<POINT>) -> V<POINT> {
        let subgroup_params = Self::subgroup_params(params);
        let translation = Self::translation(params);
        &S::transform(&subgroup_params, point) + &translation
    }

    fn to_ambient(params: &V<POINT>) -> V<AMBIENT> {
        let mut ambient = V::<AMBIENT>::zeros();
        ambient.view_mut((0, 0), (AMBIENT, 1)).copy_from(&params);
        ambient
    }

    fn compact(params: &V<PARAMS>) -> M<POINT, AMBIENT> {
        let mut mat = M::<POINT, AMBIENT>::zeros();

        mat.view_mut((0, 0), (POINT, POINT))
            .copy_from(&S::matrix(&Self::subgroup_params(params)));

        mat.view_mut((0, POINT), (POINT, 1))
            .copy_from(&Self::translation(params));

        mat
    }

    fn matrix(params: &V<PARAMS>) -> M<AMBIENT, AMBIENT> {
        let mut mat = M::<AMBIENT, AMBIENT>::identity();

        mat.view_mut((0, 0), (POINT, POINT))
            .copy_from(&S::matrix(&Self::subgroup_params(params)));

        mat.view_mut((0, POINT), (POINT, 1))
            .copy_from(&Self::translation(params));
        mat
    }

    fn ad(tangent: &V<DOF>) -> M<DOF, DOF> {
        let mut ad = M::<DOF, DOF>::zeros();
        ad.fixed_view_mut::<POINT, POINT>(0, 0)
            .copy_from(&S::hat(&Self::subgroup_tangent(tangent)));
        ad.fixed_view_mut::<POINT, SDOF>(0, POINT)
            .copy_from(&S::ad_of_translation(&Self::translation_tangent(tangent)));
        ad.fixed_view_mut::<SDOF, SDOF>(POINT, POINT)
            .copy_from(&S::ad(&Self::subgroup_tangent(tangent)));
        ad
    }
}

impl<
        const DOF: usize,
        const PARAMS: usize,
        const POINT: usize,
        const AMBIENT: usize,
        const SDOF: usize,
        const SPARAMS: usize,
        S: LieSubgroupImplTrait<SDOF, SPARAMS, POINT, POINT>,
    >
    lie::lie_group::LieGroup<
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SemiDirectProductImpl<DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, S>,
    >
{
    pub fn from_t_and_subgroup(
        translation: &V<POINT>,
        subgroup: &lie::lie_group::LieGroup<SDOF, SPARAMS, POINT, POINT, S>,
    ) -> Self {
        let params =
            SemiDirectProductImpl::<DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, S>::params_from(
                translation,
                subgroup.params(),
            );
        Self::from_params(&params)
    }

    pub fn from_t(translation: &V<POINT>) -> Self {
        Self::from_t_and_subgroup(
            translation,
            &lie::lie_group::LieGroup::<SDOF, SPARAMS, POINT, POINT, S>::identity(),
        )
    }

    pub fn from_subgroup(
        subgroup: &lie::lie_group::LieGroup<SDOF, SPARAMS, POINT, POINT, S>,
    ) -> Self {
        Self::from_t_and_subgroup(&V::<POINT>::zeros(), subgroup)
    }
}
