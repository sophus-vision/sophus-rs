use crate::prelude::*;
use sophus_core::manifold::traits::TangentImpl;
use sophus_core::params::ParamsImpl;
use std::fmt::Debug;

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
>: ParamsImpl<S, PARAMS, BATCH_SIZE> + TangentImpl<S, DOF, BATCH_SIZE> + Clone + Debug
{
    /// Generic scalar, real scalar, and dual scalar
    type GenG<S2: IsScalar<BATCH_SIZE>>: IsLieGroupImpl<S2, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>;
    /// Real scalar
    type RealG: IsLieGroupImpl<S::RealScalar, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>;
    /// DualScalar scalar - for automatic differentiation
    type DualG: IsLieGroupImpl<S::DualScalar, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>;

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
pub trait IsRealLieGroupImpl<
    S: IsRealScalar<BATCH_SIZE>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const BATCH_SIZE: usize,
>: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>
{
    /// derivative of group multiplication with respect to the first argument
    fn da_a_mul_b(a: &S::Vector<PARAMS>, b: &S::Vector<PARAMS>) -> S::Matrix<PARAMS, PARAMS>;

    /// derivative of group multiplication with respect to the second argument
    fn db_a_mul_b(a: &S::Vector<PARAMS>, b: &S::Vector<PARAMS>) -> S::Matrix<PARAMS, PARAMS>;

    /// derivative of exponential map
    fn dx_exp(tangent: &S::Vector<DOF>) -> S::Matrix<PARAMS, DOF>;

    /// derivative of exponential map at the identity
    fn dx_exp_x_at_0() -> S::Matrix<PARAMS, DOF>;

    /// derivative of logarithmic map
    fn dx_log_x(params: &S::Vector<PARAMS>) -> S::Matrix<DOF, PARAMS>;

    /// derivative of exponential map times a point at the identity
    fn dx_exp_x_times_point_at_0(point: S::Vector<POINT>) -> S::Matrix<POINT, DOF>;

    /// are there multiple shortest paths to the identity?
    fn has_shortest_path_ambiguity(params: &S::Vector<PARAMS>) -> S::Mask;
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
    type RealFactorG: IsLieFactorGroupImpl<S::RealScalar, DOF, PARAMS, POINT, BATCH_SIZE>;
    /// DualScalar scalar - for automatic differentiation
    type DualFactorG: IsLieFactorGroupImpl<S::DualScalar, DOF, PARAMS, POINT, BATCH_SIZE>;

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
pub trait IsRealLieFactorGroupImpl<
    S: IsRealScalar<BATCH_SIZE>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const BATCH_SIZE: usize,
>:
    IsLieGroupImpl<S, DOF, PARAMS, POINT, POINT, BATCH_SIZE>
    + IsLieFactorGroupImpl<S, DOF, PARAMS, POINT, BATCH_SIZE>
    + IsRealLieGroupImpl<S, DOF, PARAMS, POINT, POINT, BATCH_SIZE>
{
    /// derivative of V matrix
    fn dx_mat_v(tangent: &S::Vector<DOF>) -> [S::Matrix<POINT, POINT>; DOF];

    /// derivative of V matrix inverse
    fn dx_mat_v_inverse(tangent: &S::Vector<DOF>) -> [S::Matrix<POINT, POINT>; DOF];

    /// derivative of group transformation times a point with respect to the group parameters
    fn dparams_matrix_times_point(
        params: &S::Vector<PARAMS>,
        point: &S::Vector<POINT>,
    ) -> S::Matrix<POINT, PARAMS>;
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
    type RealG: IsLieGroupImpl<S::RealScalar, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>;
    /// Lie Group implementation with dual scalar - for automatic differentiation
    type DualG: IsLieGroupImpl<S::DualScalar, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>;

    /// degree of freedom
    const DOF: usize;
    /// number of parameters
    const PARAMS: usize;
    /// point dimension
    const POINT: usize;
    /// ambient dimension
    const AMBIENT: usize;

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
    type RealGroup: IsLieGroup<S::RealScalar, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>;
    /// Lie Group with dual scalar - for automatic differentiation
    type DualGroup: IsLieGroup<S::DualScalar, DOF, PARAMS, POINT, AMBIENT, BATCH_SIZE>;
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
