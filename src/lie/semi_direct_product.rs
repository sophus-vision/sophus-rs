use std::vec;

use super::traits::LieFactorGroupImplTrait;
use super::traits::LieGroupImpl;
use crate::calculus::dual::dual_scalar::Dual;
use crate::calculus::points::example_points;
use crate::calculus::types::matrix::IsMatrix;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::IsVector;
use crate::calculus::types::vector::IsVectorLike;
use crate::lie;
use crate::manifold::traits::ParamsImpl;
use crate::manifold::{self};

#[derive(Debug, Copy, Clone)]
pub struct SemiDirectProductImpl<
    S: IsScalar,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const SDOF: usize,
    const SPARAMS: usize,
    F: LieFactorGroupImplTrait<S, SDOF, SPARAMS, POINT, POINT>,
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
        F: LieFactorGroupImplTrait<S, SDOF, SPARAMS, POINT, POINT>,
    > SemiDirectProductImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, F>
{
    fn translation(params: &S::Vector<PARAMS>) -> S::Vector<POINT> {
        params.get_fixed_rows::<POINT>(0)
    }

    fn subgroup_params(params: &S::Vector<PARAMS>) -> S::Vector<SPARAMS> {
        params.get_fixed_rows::<SPARAMS>(POINT)
    }

    pub fn params_from(
        translation: &S::Vector<POINT>,
        subgroup_params: &S::Vector<SPARAMS>,
    ) -> S::Vector<PARAMS> {
        S::Vector::block_vec2(translation.clone(), subgroup_params.clone())
    }

    fn translation_tangent(tangent: &S::Vector<DOF>) -> S::Vector<POINT> {
        tangent.get_fixed_rows::<POINT>(0)
    }

    fn subgroup_tangent(tangent: &S::Vector<DOF>) -> S::Vector<SDOF> {
        tangent.get_fixed_rows::<SDOF>(POINT)
    }

    fn tangent_from(
        translation: &S::Vector<POINT>,
        subgroup_tangent: &S::Vector<SDOF>,
    ) -> S::Vector<DOF> {
        S::Vector::block_vec2(translation.clone(), subgroup_tangent.clone())
    }

    fn translation_examples() -> Vec<S::Vector<POINT>> {
        example_points()
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
        F: LieFactorGroupImplTrait<S, SDOF, SPARAMS, POINT, POINT>,
    > ParamsImpl<S, PARAMS>
    for SemiDirectProductImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, F>
{
    fn are_params_valid(params: &S::Vector<PARAMS>) -> bool {
        F::are_params_valid(&Self::subgroup_params(params))
    }

    fn params_examples() -> Vec<S::Vector<PARAMS>> {
        let mut examples = vec![];
        for subgroup_params in F::params_examples() {
            for translation in Self::translation_examples() {
                examples.push(Self::params_from(&translation, &subgroup_params));
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
        F: LieFactorGroupImplTrait<S, SDOF, SPARAMS, POINT, POINT>,
    > manifold::traits::TangentImpl<S, DOF>
    for SemiDirectProductImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, F>
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
        F: LieFactorGroupImplTrait<S, SDOF, SPARAMS, POINT, POINT>,
    > LieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT>
    for SemiDirectProductImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, F>
{
    const IS_ORIGIN_PRESERVING: bool = false;
    const IS_AXIS_DIRECTION_PRESERVING: bool = F::IS_AXIS_DIRECTION_PRESERVING;
    const IS_DIRECTION_VECTOR_PRESERVING: bool = F::IS_DIRECTION_VECTOR_PRESERVING;
    const IS_SHAPE_PRESERVING: bool = F::IS_SHAPE_PRESERVING;
    const IS_DISTANCE_PRESERVING: bool = F::IS_DISTANCE_PRESERVING;
    const IS_PARALLEL_LINE_PRESERVING: bool = true;

    fn identity_params() -> S::Vector<PARAMS> {
        Self::params_from(&S::Vector::zero(), &F::identity_params())
    }

    //    Manifold / Lie Group concepts

    fn adj(params: &S::Vector<PARAMS>) -> S::Matrix<DOF, DOF> {
        let subgroup_params = Self::subgroup_params(params);
        let translation = Self::translation(params);

        
        S::Matrix::block_mat2x2::<POINT, SDOF, POINT, SDOF>(
            (
                F::matrix(&subgroup_params),
                F::adj_of_translation(&subgroup_params, &translation),
            ),
            (S::Matrix::zero(), F::adj(&subgroup_params)),
        )
    }

    fn exp(omega: &S::Vector<DOF>) -> S::Vector<PARAMS> {
        let translation = Self::translation_tangent(omega);
        let subgroup_params = F::exp(&Self::subgroup_tangent(omega));

        let mat_v = F::mat_v(&subgroup_params, &Self::subgroup_tangent(omega));
        Self::params_from(&(mat_v * translation), &subgroup_params)
    }

    fn log(params: &S::Vector<PARAMS>) -> S::Vector<DOF> {
        let translation = Self::translation(params);

        let subgroup_params = Self::subgroup_params(params);

        let subgroup_tangent = F::log(&subgroup_params);
        let mat_v_inv = F::mat_v_inverse(&subgroup_params, &subgroup_tangent);
        let translation_tangent = mat_v_inv * translation;

        Self::tangent_from(&translation_tangent, &subgroup_tangent)
    }

    fn hat(omega: &S::Vector<DOF>) -> S::Matrix<AMBIENT, AMBIENT> {
        S::Matrix::block_mat2x2::<POINT, 1, POINT, 1>(
            (
                F::hat(&Self::subgroup_tangent(omega)),
                Self::translation_tangent(omega).to_mat(),
            ),
            (S::Matrix::zero(), S::Matrix::zero()),
        )
    }

    fn vee(hat: &S::Matrix<AMBIENT, AMBIENT>) -> S::Vector<DOF> {
        let subgroup_tangent = F::vee(&hat.get_fixed_submat::<POINT, POINT>(0, 0));
        let translation_tangent = hat.get_fixed_submat::<POINT, 1>(0, POINT);
        Self::tangent_from(&translation_tangent.get_col_vec(0), &subgroup_tangent)
    }

    // group operations

    fn group_mul(params1: &S::Vector<PARAMS>, params2: &S::Vector<PARAMS>) -> S::Vector<PARAMS> {
        let subgroup_params1 = Self::subgroup_params(params1);
        let subgroup_params2 = Self::subgroup_params(params2);
        let translation1 = Self::translation(params1);
        let translation2 = Self::translation(params2);
        let subgroup_params = F::group_mul(&subgroup_params1, &subgroup_params2);
        let f = F::transform(&subgroup_params1, &translation2);
        let translation = f + translation1;
        Self::params_from(&translation, &subgroup_params)
    }

    fn inverse(params: &S::Vector<PARAMS>) -> S::Vector<PARAMS> {
        let subgroup_params = Self::subgroup_params(params);
        let translation = Self::translation(params);
        let subgroup_params = F::inverse(&subgroup_params);
        let translation = -F::transform(&subgroup_params, &translation);
        Self::params_from(&translation, &subgroup_params)
    }

    fn transform(params: &S::Vector<PARAMS>, point: &S::Vector<POINT>) -> S::Vector<POINT> {
        let subgroup_params = Self::subgroup_params(params);
        let translation = Self::translation(params);
        F::transform(&subgroup_params, point) + translation
    }

    fn to_ambient(params: &S::Vector<POINT>) -> S::Vector<AMBIENT> {
        S::Vector::block_vec2(params.clone(), S::Vector::<1>::zero())
    }

    fn compact(params: &S::Vector<PARAMS>) -> S::Matrix<POINT, AMBIENT> {
        S::Matrix::block_mat1x2::<POINT, 1>(
            F::matrix(&Self::subgroup_params(params)),
            Self::translation(params).to_mat(),
        )
    }

    fn matrix(params: &S::Vector<PARAMS>) -> S::Matrix<AMBIENT, AMBIENT> {
        S::Matrix::block_mat2x2::<POINT, 1, POINT, 1>(
            (
                F::matrix(&Self::subgroup_params(params)),
                Self::translation(params).to_mat(),
            ),
            (S::Matrix::<1, POINT>::zero(), S::Matrix::<1, 1>::identity()),
        )
    }

    fn ad(tangent: &S::Vector<DOF>) -> S::Matrix<DOF, DOF> {
        let o = S::Matrix::<SDOF, POINT>::zero();
        S::Matrix::block_mat2x2::<POINT, SDOF, POINT, SDOF>(
            (
                F::hat(&Self::subgroup_tangent(tangent)),
                F::ad_of_translation(&Self::translation_tangent(tangent)),
            ),
            (o, F::ad(&Self::subgroup_tangent(tangent))),
        )
    }

    type GenG<S2: IsScalar> =
        SemiDirectProductImpl<S2, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, F::GenFactorG<S2>>;

    type RealG =
        SemiDirectProductImpl<f64, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, F::GenFactorG<f64>>;

    type DualG = SemiDirectProductImpl<
        Dual,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SDOF,
        SPARAMS,
        F::GenFactorG<Dual>,
    >;

    fn has_shortest_path_ambiguity(params: &<S as IsScalar>::Vector<PARAMS>) -> bool {
        F::has_shortest_path_ambiguity(&Self::subgroup_params(params))
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
        F: LieFactorGroupImplTrait<S, SDOF, SPARAMS, POINT, POINT>,
    >
    lie::lie_group::LieGroup<
        S,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        SemiDirectProductImpl<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, F>,
    >
{
    pub fn from_t_and_subgroup(
        translation: &S::Vector<POINT>,
        subgroup: &lie::lie_group::LieGroup<S, SDOF, SPARAMS, POINT, POINT, F>,
    ) -> Self {
        let params =
            SemiDirectProductImpl::<S, DOF, PARAMS, POINT, AMBIENT, SDOF, SPARAMS, F>::params_from(
                translation,
                subgroup.params(),
            );
        Self::from_params(&params)
    }

    pub fn from_t(translation: &S::Vector<POINT>) -> Self {
        Self::from_t_and_subgroup(
            translation,
            &lie::lie_group::LieGroup::<S, SDOF, SPARAMS, POINT, POINT, F>::identity(),
        )
    }

    pub fn from_subgroup(
        subgroup: &lie::lie_group::LieGroup<S, SDOF, SPARAMS, POINT, POINT, F>,
    ) -> Self {
        Self::from_t_and_subgroup(&S::Vector::<POINT>::zero(), subgroup)
    }
}
