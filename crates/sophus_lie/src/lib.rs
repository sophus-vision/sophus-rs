#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![no_std]
#![doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]
#![cfg_attr(nightly, feature(doc_auto_cfg))]

#[doc = include_str!(concat!("../",  core::env!("CARGO_PKG_README")))]
#[cfg(doctest)]
pub struct ReadmeDoctests;

#[cfg(feature = "std")]
extern crate std;

mod groups;
mod lie_group;

/// sophus_lie prelude.
///
/// It is recommended to import this prelude when working with `sophus_lie` types:
///
/// ```
/// use sophus_lie::prelude::*;
/// ```
///
/// or
///
/// ```ignore
/// use sophus::prelude::*;
/// ```
///
/// to import all preludes when using the `sophus` umbrella crate.
pub mod prelude {
    pub use sophus_autodiff::prelude::*;

    pub use crate::{
        IsLieGroup,
        IsTranslationProductGroup,
    };
}

use core::{
    borrow::Borrow,
    fmt::Debug,
};

use sophus_autodiff::prelude::*;

pub use crate::{
    groups::{
        isometry2::*,
        isometry3::*,
        rotation2::*,
        rotation3::*,
        translation_product_product::*,
    },
    lie_group::{
        average::*,
        factor_lie_group::*,
        real_lie_group::*,
        LieGroup,
    },
};

/// Disambiguate the parameters.
pub trait HasDisambiguate<
    S: IsScalar<BATCH, DM, DN>,
    const PARAMS: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>: Clone + Debug
{
    /// Disambiguate the parameters.
    ///
    /// Note that that in most cases the default implementation is what you want, and a
    /// group parameterization SHAll be chosen that does not require disambiguation.
    ///
    /// For instance, do not represent 2d rotations as a single angle, since theta and theta + 2pi,
    /// and theta + 4pi, etc. are the same rotations, and which would require disambiguation.
    /// Use the following 2d unit vector (cos(theta), sin(theta)) - aka unit complex number -
    /// instead.
    ///
    /// Example where disambiguation is needed: A rotation in 3d represented by a unit quaternion.
    /// The quaternion (r, (x, y, z)) and (r, (-x, -y, -z)) represent the same rotation. In this
    /// case, the disambiguation function would return a disambiguated representation, e.g. a
    /// unit quaternion with the positive r component.
    fn disambiguate(params: S::Vector<PARAMS>) -> S::Vector<PARAMS> {
        params
    }
}

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
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>:
    IsParamsImpl<S, PARAMS, BATCH, DM, DN>
    + IsTangent<S, DOF, BATCH, DM, DN>
    + HasDisambiguate<S, PARAMS, BATCH, DM, DN>
    + Clone
    + Debug
{
    /// Generic scalar, real scalar, and dual scalar
    type GenG<S2: IsScalar<BATCH, DM, DN>>: IsLieGroupImpl<
        S2,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        BATCH,
        DM,
        DN,
    >;
    /// Real scalar
    type RealG: IsLieGroupImpl<S::RealScalar, DOF, PARAMS, POINT, AMBIENT, BATCH, 0, 0>;
    /// DualScalar scalar - for automatic differentiation
    type DualG<const M: usize, const N: usize>: IsLieGroupImpl<
        S::DualScalar<M, N>,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        BATCH,
        M,
        N,
    >;

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
    S: IsRealScalar<BATCH>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const BATCH: usize,
>: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH, 0, 0>
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
    fn dx_exp_x_times_point_at_0(point: &S::Vector<POINT>) -> S::Matrix<POINT, DOF>;

    /// derivative of matrix representation with respect to the internal parameters
    ///
    /// precondition: column index in [0, AMBIENT-1]
    fn dparams_matrix(params: &S::Vector<PARAMS>, col_idx: usize) -> S::Matrix<POINT, PARAMS>;

    /// are there multiple shortest paths to the identity?
    fn has_shortest_path_ambiguity(params: &S::Vector<PARAMS>) -> S::Mask;
}

/// Lie Factor Group
///
/// Can be a factor of a semi-direct product group
pub trait IsLieFactorGroupImpl<
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>: IsLieGroupImpl<S, DOF, PARAMS, POINT, POINT, BATCH, DM, DN>
{
    /// Generic scalar, real scalar, and dual scalar
    type GenFactorG<S2: IsScalar<BATCH, M, N>, const M: usize, const N: usize>: IsLieFactorGroupImpl<
        S2,
        DOF,
        PARAMS,
        POINT,
        BATCH,
        M,
        N,
    >;
    /// Real scalar
    type RealFactorG: IsLieFactorGroupImpl<S::RealScalar, DOF, PARAMS, POINT, BATCH, 0, 0>;
    /// DualScalar scalar - for automatic differentiation
    type DualFactorG<const M: usize, const N: usize>: IsLieFactorGroupImpl<
        S::DualScalar<M, N>,
        DOF,
        PARAMS,
        POINT,
        BATCH,
        M,
        N,
    >;

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
    S: IsRealScalar<BATCH>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const BATCH: usize,
>:
    IsLieGroupImpl<S, DOF, PARAMS, POINT, POINT, BATCH, 0, 0>
    + IsLieFactorGroupImpl<S, DOF, PARAMS, POINT, BATCH, 0, 0>
    + IsRealLieGroupImpl<S, DOF, PARAMS, POINT, POINT, BATCH>
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
    S: IsScalar<BATCH, DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const BATCH: usize,
    const DM: usize,
    const DN: usize,
>: IsTangent<S, DOF, BATCH, DM, DN> + HasParams<S, PARAMS, BATCH, DM, DN>
{
    /// This is the actual Lie Group implementation
    type G: IsLieGroupImpl<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN>;
    /// Lie Group implementation with generic scalar, real scalar, and dual scalar
    type GenG<S2: IsScalar<BATCH, DM, DN>>: IsLieGroupImpl<
        S2,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        BATCH,
        DM,
        DN,
    >;
    /// Lie Group implementation- with real scalar
    type RealG: IsLieGroupImpl<S::RealScalar, DOF, PARAMS, POINT, AMBIENT, BATCH, 0, 0>;
    /// Lie Group implementation with dual scalar - for automatic differentiation
    type DualG<const M: usize, const N: usize>: IsLieGroupImpl<
        S::DualScalar<M, N>,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        BATCH,
        M,
        N,
    >;

    /// degree of freedom
    const DOF: usize;
    /// number of parameters
    const PARAMS: usize;
    /// point dimension
    const POINT: usize;
    /// ambient dimension
    const AMBIENT: usize;

    /// Get the Lie Group
    type GenGroup<S2: IsScalar<BATCH, DM, DN>, G2: IsLieGroupImpl<S2, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN>>: IsLieGroup<
        S2,
        DOF,
        PARAMS,
        POINT,
        AMBIENT,
        BATCH
        ,DM,DN
    >;
}

/// Lie Group trait for real scalar, f64
pub trait IsTranslationProductGroup<
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
    FactorGroup: IsLieGroup<S, SDOF, SPARAMS, POINT, POINT, BATCH, DM, DN>,
>: IsLieGroup<S, DOF, PARAMS, POINT, AMBIENT, BATCH, DM, DN>
{
    /// This is the actual Lie Factor Group implementation
    type Impl;

    /// Create from translation and factor group element
    fn from_translation_and_factor<F>(translation: S::Vector<POINT>, factor: F) -> Self
    where
        F: Borrow<FactorGroup>;

    /// set translation
    fn set_translation(&mut self, translation: S::Vector<POINT>);
    /// get translation
    fn translation(&self) -> S::Vector<POINT>;

    /// set factor group element
    fn set_factor<F>(&mut self, factor: F)
    where
        F: Borrow<FactorGroup>;

    /// get factor group element
    fn factor(&self) -> FactorGroup;
}

/// slice is empty
#[derive(Debug)]
pub struct EmptySliceError;

/// Lie Group trait
pub trait HasAverage<
    S: IsSingleScalar<DM, DN>,
    const DOF: usize,
    const PARAMS: usize,
    const POINT: usize,
    const AMBIENT: usize,
    const DM: usize,
    const DN: usize,
>: IsLieGroup<S, DOF, PARAMS, POINT, AMBIENT, 1, DM, DN> + core::marker::Sized
{
    /// Lie group average
    fn average(parent_from_body_transforms: &[Self]) -> Result<Self, EmptySliceError>;
}
