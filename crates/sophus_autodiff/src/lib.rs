#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![no_std]
#![allow(clippy::needless_range_loop)]
//! Automatic differentiation module
//!  - linear algebra types
//!      * such as [linalg::VecF64], and [linalg::MatF64]
//!      * batch types such as [linalg::BatchScalarF64], [linalg::BatchVecF64],
//!        [linalg::BatchMatF64] - require the `simd` feature
//!  - dual numbers: [dual::DualScalar], [dual::DualVector], [dual::DualMatrix]
//!      * [maps::curves] f: ℝ -> ℝ,   f: ℝ -> ℝʳ,   f: ℝ -> ℝʳ x ℝᶜ
//!      * [maps::scalar_valued_maps]: f: ℝᵐ -> ℝ,   f: ℝᵐ x ℝⁿ -> ℝ
//!      * [maps::vector_valued_maps]: f: ℝᵐ -> ℝᵖ,   f: ℝᵐ x ℝⁿ -> ℝᵖ
//!      * [maps::matrix_valued_maps]: f: ℝᵐ -> ℝʳ x ℝᶜ,   f: ℝᵐ x ℝⁿ -> ℝʳ x ℝᶜ

#[cfg(feature = "std")]
extern crate std;

/// dual numbers - for automatic differentiation
pub mod dual;
/// floating point
pub mod floating_point;
/// core linear algebra types
pub mod linalg;
/// manifolds
pub mod manifold;
/// curves, scalar-valued, vector-valued, and matrix-valued maps
pub mod maps;
/// params
pub mod params;
/// points
pub mod points;

pub use nalgebra;
pub use ndarray;

pub use crate::points::*;

/// sophus_autodiff prelude
pub mod prelude {
    pub use crate::{
        dual::{
            matrix::IsDualMatrix,
            scalar::IsDualScalar,
            vector::IsDualVector,
        },
        linalg::{
            bool_mask::IsBoolMask,
            matrix::{
                IsMatrix,
                IsRealMatrix,
                IsSingleMatrix,
            },
            scalar::{
                IsCoreScalar,
                IsRealScalar,
                IsScalar,
                IsSingleScalar,
            },
            vector::{
                IsRealVector,
                IsSingleVector,
                IsVector,
            },
        },
        manifold::{
            IsManifold,
            IsTangent,
        },
        params::{
            HasParams,
            IsParamsImpl,
        },
    };
}
