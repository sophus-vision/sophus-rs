#![feature(portable_simd)]
#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]
//! Core math functionality including
//!  - linear algebra types
//!      * such as [linalg::VecF64], and [linalg::MatF64]
//!      * batch types such as [linalg::BatchScalarF64], [linalg::BatchVecF64],
//!        [linalg::BatchMatF64]
//!  - tensors
//!      * design: dynamic tensor (ndarray) of static tensors (nalgebra)
//!  - differentiation tools
//!      * dual numbers: [calculus::dual::DualScalar], [calculus::dual::DualVector],
//!        [calculus::dual::DualMatrix]
//!      * [calculus::maps::curves] f: ℝ -> ℝ,   f: ℝ -> ℝʳ,   f: ℝ -> ℝʳ x ℝᶜ
//!      * [calculus::maps::scalar_valued_maps]: f: ℝᵐ -> ℝ,   f: ℝᵐ x ℝⁿ -> ℝ
//!      * [calculus::maps::vector_valued_maps]: f: ℝᵐ -> ℝᵖ,   f: ℝᵐ x ℝⁿ -> ℝᵖ
//!      * [calculus::maps::matrix_valued_maps]: f: ℝᵐ -> ℝʳ x ℝᶜ,   f: ℝᵐ x ℝⁿ -> ℝʳ x ℝᶜ
//!  - splines
//!      * [calculus::spline::CubicBSpline]
//!  - intervals, regions
//!      * closed interval: [calculus::region::Interval]
//!      * closed region: [calculus::region::Interval]
//!  - manifolds: [manifold::traits]

/// calculus - differentiation, splines, and more
pub mod calculus;

/// linear algebra types
pub mod linalg;

/// manifolds
pub mod manifold;
pub use crate::manifold::*;

/// params
pub mod params;
pub use crate::params::*;

/// points
pub mod points;
pub use crate::points::*;

/// tensors
pub mod tensor;
pub use crate::tensor::arc_tensor::*;
pub use crate::tensor::mut_tensor::*;
pub use crate::tensor::mut_tensor_view::*;
pub use crate::tensor::tensor_view::*;

/// sophus_core prelude
pub mod prelude {
    pub use crate::calculus::dual::dual_matrix::IsDualMatrix;
    pub use crate::calculus::dual::dual_scalar::IsDual;
    pub use crate::calculus::dual::dual_scalar::IsDualScalar;
    pub use crate::calculus::dual::dual_vector::IsDualVector;
    pub use crate::calculus::region::IsRegion;
    pub use crate::linalg::bool_mask::IsBoolMask;
    pub use crate::linalg::matrix::IsMatrix;
    pub use crate::linalg::matrix::IsRealMatrix;
    pub use crate::linalg::matrix::IsSingleMatrix;
    pub use crate::linalg::scalar::IsCoreScalar;
    pub use crate::linalg::scalar::IsRealScalar;
    pub use crate::linalg::scalar::IsScalar;
    pub use crate::linalg::scalar::IsSingleScalar;
    pub use crate::linalg::vector::IsRealVector;
    pub use crate::linalg::vector::IsSingleVector;
    pub use crate::linalg::vector::IsVector;
    pub use crate::manifold::traits::IsManifold;
    pub use crate::params::HasParams;
    pub use crate::tensor::element::IsStaticTensor;
    pub use crate::tensor::mut_tensor_view::IsMutTensorLike;
    pub use crate::tensor::tensor_view::IsTensorLike;
    pub use crate::tensor::tensor_view::IsTensorView;
}
