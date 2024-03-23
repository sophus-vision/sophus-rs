use std::fmt::Debug;

use sophus_calculus::dual::dual_scalar::Dual;
use sophus_calculus::manifold::traits::TangentImpl;
use sophus_calculus::manifold::{self};
use sophus_calculus::types::params::HasParams;
use sophus_calculus::types::params::ParamsImpl;
use sophus_calculus::types::scalar::IsScalar;
use sophus_calculus::types::MatF64;
use sophus_calculus::types::VecF64;

/// Lie Group implementation trait
///
/// Here, the actual manifold is represented by as a N-dimensional parameter tuple.
/// See LieGroup struct for the concrete implementation.
///
/// DOF: Degrees of freedom
/// NUM_PARAMS:
/// POINT_DIM:
/// N: dimension ambient the dimension
pub trait IsLieGroupImpl<
    S: IsScalar<BATCH_SIZE>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const BATCH_SIZE: usize,
>:
    ParamsImpl<S, PARAMS, BATCH_SIZE>
    + manifold::traits::TangentImpl<S, DOF, BATCH_SIZE>
    + Clone
    + Debug
{
    /// Generic scalar, real scalar, and dual scalar
    type GenG<S2: IsScalar<BATCH_SIZE>>: IsLieGroupImpl<S2, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>;
    /// Real scalar
    type RealG: IsLieGroupImpl<f64, DOF, PARAMS, POINT, AMBIENT, 1>;
    /// Dual scalar - for automatic differentiation
    type DualG: IsLieGroupImpl<Dual, DOF, PARAMS, POINT, AMBIENT, 1>;

    /// is transformation origin preserving?
    const IS_ORIGIN_PRESERVING: bool;
    /// is transformation axis direction preserving?
    const IS_AXIS_DIRECTION_PRESERVING: bool;
    /// is transformation direction vector preserving?
    const IS_DIRECTION_VECTOR_PRESERVING: bool;
    /// is transformation shape preserving?
    const IS_SHAPE_PRESERVING: bool;
    /// is transformation distance preserving?
    const IS_DISTANCE_PRESERVING: bool;
    /// is transformation angle preserving?
    const IS_PARALLEL_LINE_PRESERVING: bool;

    /// identity parameters
    fn identity_params() -> S::Vector<PARAMS>;

    // Manifold / Lie Group concepts

    /// are there multiple shortest paths to the identity?
    fn has_shortest_path_ambiguity(params: &S::Vector<PARAMS>) -> bool;

    /// group adjoint
    fn adj(params: &S::Vector<PARAMS>) -> S::Matrix<DOF, DOF>;

    /// algebra adjoint
    fn ad(tangent: &S::Vector<DOF>) -> S::Matrix<DOF, DOF>;

    /// exponential map
    fn exp(omega: &S::Vector<DOF>) -> S::Vector<PARAMS>;

    /// logarithmic map
    fn log(params: &S::Vector<PARAMS>) -> S::Vector<DOF>;

    /// hat operator
    fn hat(omega: &S::Vector<DOF>) -> S::Matrix<AMBIENT, AMBIENT>;

    /// vee operator
    fn vee(hat: &S::Matrix<AMBIENT, AMBIENT>) -> S::Vector<DOF>;

    // group operations

    /// group multiplication
    fn group_mul(params1: &S::Vector<PARAMS>, params2: &S::Vector<PARAMS>) -> S::Vector<PARAMS>;

    /// group inverse
    fn inverse(params: &S::Vector<PARAMS>) -> S::Vector<PARAMS>;

    // Group actions

    /// group action on a point
    fn transform(params: &S::Vector<PARAMS>, point: &S::Vector<POINT>) -> S::Vector<POINT>;

    /// convert minimal manifold representation to ambient manifold representation
    fn to_ambient(params: &S::Vector<POINT>) -> S::Vector<AMBIENT>;

    /// return compact matrix representation
    fn compact(params: &S::Vector<PARAMS>) -> S::Matrix<POINT, AMBIENT>;

    /// return square matrix representation
    fn matrix(params: &S::Vector<PARAMS>) -> S::Matrix<AMBIENT, AMBIENT>;
}

/// Lie Group implementation trait for real scalar, f64
pub trait IsF64LieGroupImpl<
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
>: IsLieGroupImpl<f64, DOF, PARAMS, POINT, AMBIENT, 1>
{
    /// derivative of group multiplication with respect to the first argument
    fn da_a_mul_b(a: &VecF64<PARAMS>, b: &VecF64<PARAMS>) -> MatF64<PARAMS, PARAMS>;

    /// derivative of group multiplication with respect to the second argument
    fn db_a_mul_b(a: &VecF64<PARAMS>, b: &VecF64<PARAMS>) -> MatF64<PARAMS, PARAMS>;

    /// derivative of exponential map
    fn dx_exp(tangent: &VecF64<DOF>) -> MatF64<PARAMS, DOF>;

    /// derivative of exponential map at the identity
    fn dx_exp_x_at_0() -> MatF64<PARAMS, DOF>;

    /// derivative of logarithmic map
    fn dx_log_x(params: &VecF64<PARAMS>) -> MatF64<DOF, PARAMS>;

    /// derivative of exponential map times a point at the identity
    fn dx_exp_x_times_point_at_0(point: VecF64<POINT>) -> MatF64<POINT, DOF>;
}

/// Lie Factor Group
///
/// Can be a factor of a semi-direct product group
pub trait IsLieFactorGroupImpl<
    S: IsScalar<BATCH_SIZE>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const BATCH_SIZE: usize,
>: IsLieGroupImpl<S, DOF, PARAMS, POINT, POINT, BATCH_SIZE>
{
    /// Generic scalar, real scalar, and dual scalar
    type GenFactorG<S2: IsScalar<BATCH_SIZE>>: IsLieFactorGroupImpl<
        S2,
        DOF,
        PARAMS,
        POINT,
        BATCH_SIZE,
    >;
    /// Real scalar
    type RealFactorG: IsLieFactorGroupImpl<f64, DOF, PARAMS, POINT, 1>;
    /// Dual scalar - for automatic differentiation
    type DualFactorG: IsLieFactorGroupImpl<Dual, DOF, PARAMS, POINT, 1>;

    /// V matrix - used by semi-direct product exponential
    fn mat_v(tangent: &S::Vector<DOF>) -> S::Matrix<POINT, POINT>;

    /// V matrix inverse - used by semi-direct product logarithm
    fn mat_v_inverse(tangent: &S::Vector<DOF>) -> S::Matrix<POINT, POINT>;

    /// group adjoint of translation
    fn adj_of_translation(
        params: &S::Vector<PARAMS>,
        point: &S::Vector<POINT>,
    ) -> S::Matrix<POINT, DOF>;

    /// algebra adjoint of translation
    fn ad_of_translation(point: &S::Vector<POINT>) -> S::Matrix<POINT, DOF>;
}

/// Lie Factor Group implementation trait for real scalar, f64
pub trait IsF64LieFactorGroupImpl<const DOF: usize, const PARAMS: usize, const POINT: usize>:
    IsLieGroupImpl<f64, DOF, PARAMS, POINT, POINT, 1>
    + IsLieFactorGroupImpl<f64, DOF, PARAMS, POINT, 1>
    + IsF64LieGroupImpl<DOF, PARAMS, POINT, POINT>
{
    /// derivative of V matrix
    fn dx_mat_v(tangent: &VecF64<DOF>) -> [MatF64<POINT, POINT>; DOF];

    /// derivative of V matrix inverse
    fn dx_mat_v_inverse(tangent: &VecF64<DOF>) -> [MatF64<POINT, POINT>; DOF];

    /// derivative of group transformation times a point with respect to the group parameters
    fn dparams_matrix_times_point(
        params: &VecF64<PARAMS>,
        point: &VecF64<POINT>,
    ) -> MatF64<POINT, PARAMS>;
}

/// Lie Group trait
pub trait IsLieGroup<
    S: IsScalar<BATCH_SIZE>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const BATCH_SIZE: usize,
>: TangentImpl<S, DOF, BATCH_SIZE> + HasParams<S, PARAMS, BATCH_SIZE>
{
    /// This is the actual Lie Group implementation
    type G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>;
    /// Lie Group implementation with generic scalar, real scalar, and dual scalar
    type GenG<S2: IsScalar<BATCH_SIZE>>: IsLieGroupImpl<S2, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>;
    /// Lie Group implementation- with real scalar
    type RealG: IsLieGroupImpl<f64, DOF, PARAMS, POINT, AMBIENT, 1>;
    /// Lie Group implementation with dual scalar - for automatic differentiation
    type DualG: IsLieGroupImpl<Dual, DOF, PARAMS, POINT, AMBIENT, 1>;

    /// Get the Lie Group
    type GenGroup<S2: IsScalar<BATCH_SIZE>, G2: IsLieGroupImpl<S2, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>>: IsLieGroup<
        S2,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        BATCH_SIZE
    >;
    /// Lie Group with real scalar
    type RealGroup: IsLieGroup<f64, DOF, PARAMS, POINT, AMBIENT, 1>;
    /// Lie Group with dual scalar - for automatic differentiation
    type DualGroup: IsLieGroup<Dual, DOF, PARAMS, POINT, AMBIENT, 1>;
}

/// Lie Group trait for real scalar, f64
pub trait IsTranslationProductGroup<
    S: IsScalar<BATCH_SIZE>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const SDOF: usize,
    const SPARAMS: usize,
    const BATCH_SIZE: usize,
    FactorGroup: IsLieGroup<S, SDOF, SPARAMS, POINT, POINT, BATCH_SIZE>,
>: IsLieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>
{
    /// This is the actual Lie Factor Group implementation
    type Impl;

    /// Create from translation and factor group element
    fn from_translation_and_factor(translation: &S::Vector<POINT>, factor: &FactorGroup) -> Self;

    /// Create from translation
    fn from_t(params: &S::Vector<POINT>) -> Self;

    /// set translation
    fn set_translation(&mut self, translation: &S::Vector<POINT>);
    /// get translation
    fn translation(&self) -> S::Vector<POINT>;

    /// set factor group element
    fn set_factor(&mut self, factor: &FactorGroup);

    /// get factor group element
    fn factor(&self) -> FactorGroup;
}
