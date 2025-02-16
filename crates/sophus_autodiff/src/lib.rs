#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![no_std]
#![allow(clippy::needless_range_loop)]
#![doc = include_str!(concat!("./", std::env!("CARGO_PKG_README")))]
#![cfg_attr(nightly, feature(doc_auto_cfg))]

#[cfg(feature = "std")]
extern crate std;

/// Dual numbers – for automatic differentiation. This module provides forward-mode AD through dual
/// number types, enabling derivative computations for scalars, vectors, and matrices
/// (including batch/`simd` forms).
pub mod dual;
/// Traits for core linear algebra types. Defines abstractions for scalars, vectors, and matrices,
/// along with optional batch/SIMD support.
pub mod linalg;
/// Traits for manifolds. A manifold generalizes vector spaces to curved settings. This module
/// offers interfaces for manifold-based computations, tangent spaces, and related logic.
pub mod manifold;
/// Numerical differentiation on curves, scalar-valued, vector-valued, and matrix-valued maps.
/// Provides finite-difference utilities for computing derivatives of user-defined functions.
pub mod maps;
/// Parameter traits. Provides a uniform interfaces for types which state is internally
/// represented by parameter vectors.
pub mod params;
/// Point traits. Defines interfaces for points in various dimensions, including bounds and
/// clamping.
pub mod points;

/// sophus_geo prelude.
///
/// It is recommended to import this prelude when working with `sophus_autodiff` types:
///
/// ```
/// use sophus_autodiff::prelude::*;
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
    pub use crate::{
        dual::{
            HasJacobian,
            IsDualMatrix,
            IsDualMatrixFromCurve,
            IsDualScalar,
            IsDualScalarFromCurve,
            IsDualVector,
            IsDualVectorFromCurve,
        },
        linalg::{
            IsBoolMask,
            IsCoreScalar,
            IsMatrix,
            IsRealMatrix,
            IsRealScalar,
            IsRealVector,
            IsScalar,
            IsSingleMatrix,
            IsSingleScalar,
            IsSingleVector,
            IsVector,
        },
        manifold::{
            IsManifold,
            IsTangent,
            IsVariable,
        },
        params::{
            HasParams,
            IsParamsImpl,
        },
        points::{
            IsPoint,
            IsUnboundedPoint,
        },
    };
}

/// nalgebra crate re-export.
pub use nalgebra;
