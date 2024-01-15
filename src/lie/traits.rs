use std::fmt::Debug;

use crate::calculus::dual::dual_scalar::Dual;
use crate::calculus::types::params::HasParams;
use crate::calculus::types::params::ParamsImpl;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::M;
use crate::calculus::types::V;
use crate::manifold::traits::TangentImpl;
use crate::manifold::{self};

/// DOF: Degrees of freedom
/// NUM_PARAMS:
/// POINT_DIM:
/// N: dimension ambient the dimension
pub trait IsLieGroupImpl<
    S: IsScalar,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
>: ParamsImpl<S, PARAMS> + manifold::traits::TangentImpl<S, DOF> + Clone + Debug
{
    type GenG<S2: IsScalar>: IsLieGroupImpl<S2, DOF, PARAMS, POINT, AMBIENT>;
    type RealG: IsLieGroupImpl<f64, DOF, PARAMS, POINT, AMBIENT>;
    type DualG: IsLieGroupImpl<Dual, DOF, PARAMS, POINT, AMBIENT>;

    const IS_ORIGIN_PRESERVING: bool;
    const IS_AXIS_DIRECTION_PRESERVING: bool;
    const IS_DIRECTION_VECTOR_PRESERVING: bool;
    const IS_SHAPE_PRESERVING: bool;
    const IS_DISTANCE_PRESERVING: bool;
    const IS_PARALLEL_LINE_PRESERVING: bool;

    fn identity_params() -> S::Vector<PARAMS>;

    // Manifold / Lie Group concepts

    fn has_shortest_path_ambiguity(params: &S::Vector<PARAMS>) -> bool;

    fn adj(params: &S::Vector<PARAMS>) -> S::Matrix<DOF, DOF>;

    fn ad(tangent: &S::Vector<DOF>) -> S::Matrix<DOF, DOF>;

    fn exp(omega: &S::Vector<DOF>) -> S::Vector<PARAMS>;

    fn log(params: &S::Vector<PARAMS>) -> S::Vector<DOF>;

    fn hat(omega: &S::Vector<DOF>) -> S::Matrix<AMBIENT, AMBIENT>;

    fn vee(hat: &S::Matrix<AMBIENT, AMBIENT>) -> S::Vector<DOF>;

    // group operations

    fn group_mul(params1: &S::Vector<PARAMS>, params2: &S::Vector<PARAMS>) -> S::Vector<PARAMS>;

    fn inverse(params: &S::Vector<PARAMS>) -> S::Vector<PARAMS>;

    // Group actions

    fn transform(params: &S::Vector<PARAMS>, point: &S::Vector<POINT>) -> S::Vector<POINT>;

    fn to_ambient(params: &S::Vector<POINT>) -> S::Vector<AMBIENT>;

    // Matrices
    fn compact(params: &S::Vector<PARAMS>) -> S::Matrix<POINT, AMBIENT>;

    fn matrix(params: &S::Vector<PARAMS>) -> S::Matrix<AMBIENT, AMBIENT>;
}

pub trait IsF64LieGroupImpl<
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
>: IsLieGroupImpl<f64, DOF, PARAMS, POINT, AMBIENT>
{
    fn da_a_mul_b(a: &V<PARAMS>, b: &V<PARAMS>) -> M<PARAMS, PARAMS>;

    fn db_a_mul_b(a: &V<PARAMS>, b: &V<PARAMS>) -> M<PARAMS, PARAMS>;

    fn dx_exp(tangent: &V<DOF>) -> M<PARAMS, DOF>;

    fn dx_exp_x_at_0() -> M<PARAMS, DOF>;

    fn dx_log_x(params: &V<PARAMS>) -> M<DOF, PARAMS>;

    fn dx_exp_x_times_point_at_0(point: V<POINT>) -> M<POINT, DOF>;
}

pub trait IsLieFactorGroupImpl<
    S: IsScalar,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
>: IsLieGroupImpl<S, DOF, PARAMS, POINT, POINT>
{
    type GenFactorG<S2: IsScalar>: IsLieFactorGroupImpl<S2, DOF, PARAMS, POINT>;
    type RealFactorG: IsLieFactorGroupImpl<f64, DOF, PARAMS, POINT>;
    type DualFactorG: IsLieFactorGroupImpl<Dual, DOF, PARAMS, POINT>;

    fn mat_v(tangent: &S::Vector<DOF>) -> S::Matrix<POINT, POINT>;

    fn mat_v_inverse(tangent: &S::Vector<DOF>) -> S::Matrix<POINT, POINT>;

    fn adj_of_translation(
        params: &S::Vector<PARAMS>,
        point: &S::Vector<POINT>,
    ) -> S::Matrix<POINT, DOF>;

    fn ad_of_translation(point: &S::Vector<POINT>) -> S::Matrix<POINT, DOF>;
}

pub trait IsF64LieFactorGroupImpl<const DOF: usize, const PARAMS: usize, const POINT: usize>:
    IsLieGroupImpl<f64, DOF, PARAMS, POINT, POINT>
    + IsLieFactorGroupImpl<f64, DOF, PARAMS, POINT>
    + IsF64LieGroupImpl<DOF, PARAMS, POINT, POINT>
{
    fn dx_mat_v(tangent: &V<DOF>) -> [M<POINT, POINT>; DOF];

    fn dx_mat_v_inverse(tangent: &V<DOF>) -> [M<POINT, POINT>; DOF];

    fn dparams_matrix_times_point(params: &V<PARAMS>, point: &V<POINT>) -> M<POINT, PARAMS>;
}

pub trait IsLieGroup<
    S: IsScalar,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
>: TangentImpl<S, DOF> + HasParams<S, PARAMS>
{
    type G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT>;
    type GenG<S2: IsScalar>: IsLieGroupImpl<S2, DOF, PARAMS, POINT, AMBIENT>;
    type RealG: IsLieGroupImpl<f64, DOF, PARAMS, POINT, AMBIENT>;
    type DualG: IsLieGroupImpl<Dual, DOF, PARAMS, POINT, AMBIENT>;

    type GenGroup<S2: IsScalar, G2: IsLieGroupImpl<S2, DOF, PARAMS, POINT, AMBIENT>>: IsLieGroup<
        S2,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
    >;
    type RealGroup: IsLieGroup<f64, DOF, PARAMS, POINT, AMBIENT>;
    type DualGroup: IsLieGroup<Dual, DOF, PARAMS, POINT, AMBIENT>;
}

pub trait IsTranslationProductGroup<
    S: IsScalar,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const SDOF: usize,
    const SPARAMS: usize,
    FactorGroup: IsLieGroup<S, SDOF, SPARAMS, POINT, POINT>,
>: IsLieGroup<S, DOF, PARAMS, POINT, AMBIENT>
{
    type Impl;

    fn from_translation_and_factor(translation: &S::Vector<POINT>, factor: &FactorGroup) -> Self;

    fn from_t(params: &S::Vector<POINT>) -> Self;

    fn set_translation(&mut self, translation: &S::Vector<POINT>);
    fn translation(&self) -> S::Vector<POINT>;

    fn set_factor(&mut self, factor: &FactorGroup);
    fn factor(&self) -> FactorGroup;
}
