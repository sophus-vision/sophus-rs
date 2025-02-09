//! Geometry crate - part of the sophus-rs project
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![no_std]
#![allow(clippy::needless_range_loop)]

/// hyper-plane: line in 2d, plane in 3d, ...
pub mod hyperplane;
/// n-Sphere: circle, sphere, ...
pub mod hypersphere;
/// ray
pub mod ray;
/// region
pub mod region;
/// unit vector
pub mod unit_vector;

/// sophus_geo prelude
pub mod prelude {
    pub use sophus_autodiff::prelude::*;
    pub use sophus_lie::prelude::*;

    pub use crate::region::{
        IsNonEmptyRegion,
        IsRegion,
        IsRegionBase,
    };
}
