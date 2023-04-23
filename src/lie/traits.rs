use crate::{calculus, manifold};
use nalgebra::{SMatrix, SVector};

type V<const N: usize> = SVector<f64, N>;
type M<const N: usize, const O: usize> = SMatrix<f64, N, O>;

/// DOF: Degrees of freedom
/// NUM_PARAMS:
/// POINT_DIM:
/// N: dimension ambient the dimension
pub trait LieGroupImpl<
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
>: calculus::traits::ParamsImpl<PARAMS> + manifold::traits::TangentImpl<DOF>
{
    const IS_ORIGIN_PRESERVING: bool;
    const IS_AXIS_DIRECTION_PRESERVING: bool;
    const IS_DIRECTION_VECTOR_PRESERVING: bool;
    const IS_SHAPE_PRESERVING: bool;
    const IS_DISTANCE_PRESERVING: bool;
    const IS_PARALLEL_LINE_PRESERVING: bool;

    fn identity_params() -> V<PARAMS>;

    // Manifold / Lie Group concepts

    fn adj(params: &V<PARAMS>) -> M<DOF, DOF>;

    fn exp(omega: &V<DOF>) -> V<PARAMS>;

    fn log(params: &V<PARAMS>) -> V<DOF>;

    fn hat(omega: &V<DOF>) -> M<AMBIENT, AMBIENT>;

    fn vee(hat: &M<AMBIENT, AMBIENT>) -> V<DOF>;

    // group operations

    fn multiply(params1: &V<PARAMS>, params2: &V<PARAMS>) -> V<PARAMS>;

    fn inverse(params: &V<PARAMS>) -> V<PARAMS>;

    // Group actions

    fn transform(params: &V<PARAMS>, point: &V<POINT>) -> V<POINT>;

    fn to_ambient(params: &V<POINT>) -> V<AMBIENT>;

    // Matrices
    fn compact(params: &V<PARAMS>) -> M<POINT, AMBIENT>;

    fn matrix(params: &V<PARAMS>) -> M<AMBIENT, AMBIENT>;

    // Derivatives

    fn ad(tangent: &V<DOF>) -> M<DOF, DOF>;
}

pub trait LieSubgroupImplTrait<
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT_DIM: usize,
>: LieGroupImpl<DOF, PARAMS, POINT, AMBIENT_DIM>
{
    fn mat_v(params: &V<PARAMS>, tangent: &V<DOF>) -> SMatrix<f64, POINT, POINT>;

    fn mat_v_inverse(params: &V<PARAMS>, tangent: &V<DOF>) -> SMatrix<f64, POINT, POINT>;

    fn adj_of_translation(params: &V<PARAMS>, point: &V<POINT>) -> SMatrix<f64, POINT, DOF>;

    fn ad_of_translation(point: &V<POINT>) -> SMatrix<f64, POINT, DOF>;
}
