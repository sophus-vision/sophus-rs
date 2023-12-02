use std::fmt::Debug;

use crate::calculus::dual::dual_scalar::Dual;
use crate::calculus::types::scalar::IsScalar;
use crate::manifold::traits::ParamsImpl;
use crate::manifold::{self};

/// DOF: Degrees of freedom
/// NUM_PARAMS:
/// POINT_DIM:
/// N: dimension ambient the dimension
pub trait LieGroupImpl<
    S: IsScalar,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
>: ParamsImpl<S, PARAMS> + manifold::traits::TangentImpl<S, DOF> + Clone + Debug
{
    type GenG<S2: IsScalar>: LieGroupImpl<S2, DOF, PARAMS, POINT, AMBIENT>;
    type RealG: LieGroupImpl<f64, DOF, PARAMS, POINT, AMBIENT>;
    type DualG: LieGroupImpl<Dual, DOF, PARAMS, POINT, AMBIENT>;

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

    // Derivatives

    fn ad(tangent: &S::Vector<DOF>) -> S::Matrix<DOF, DOF>;
}

pub trait IsLieGroup<
    S: IsScalar,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
>
{
    type GenGroup<S2: IsScalar, G2: LieGroupImpl<S2, DOF, PARAMS, POINT, AMBIENT>>: IsLieGroup<
        S2,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
    >;
    type RealGroup: IsLieGroup<f64, DOF, PARAMS, POINT, AMBIENT>;
    type DualGroup: IsLieGroup<Dual, DOF, PARAMS, POINT, AMBIENT>;
}

pub trait LieFactorGroupImplTrait<
    S: IsScalar,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
>: LieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT>
{
    type GenFactorG<S2: IsScalar>: LieFactorGroupImplTrait<S2, DOF, PARAMS, POINT, AMBIENT>;
    type RealFactorG: LieFactorGroupImplTrait<f64, DOF, PARAMS, POINT, AMBIENT>;
    type DualFactorG: LieFactorGroupImplTrait<Dual, DOF, PARAMS, POINT, AMBIENT>;

    fn mat_v(params: &S::Vector<PARAMS>, tangent: &S::Vector<DOF>) -> S::Matrix<POINT, POINT>;

    fn mat_v_inverse(
        params: &S::Vector<PARAMS>,
        tangent: &S::Vector<DOF>,
    ) -> S::Matrix<POINT, POINT>;

    fn adj_of_translation(
        params: &S::Vector<PARAMS>,
        point: &S::Vector<POINT>,
    ) -> S::Matrix<POINT, DOF>;

    fn ad_of_translation(point: &S::Vector<POINT>) -> S::Matrix<POINT, DOF>;
}
