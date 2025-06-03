#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![no_std]
#![allow(clippy::needless_range_loop)]
#![doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]
#![cfg_attr(nightly, feature(doc_auto_cfg))]

#[cfg(feature = "std")]
extern crate std;

#[doc = include_str!(concat!("../",  core::env!("CARGO_PKG_README")))]
#[cfg(doctest)]
pub struct ReadmeDoctests;

mod hyperplane;
mod hypersphere;
mod ray;
/// Intervals and box regions.
pub mod region;
mod unit_vector;

pub use crate::{
    hyperplane::*,
    hypersphere::*,
    ray::*,
    unit_vector::*,
};

pub use sophus_lie::Quaternion;

/// sophus_geo prelude.
///
/// It is recommended to import this prelude when working with `sophus_geo types:
///
/// ```
/// use sophus_geo::prelude::*;
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
    pub use sophus_lie::prelude::*;

    pub use crate::region::{
        IsNonEmptyRegion,
        IsRegion,
        IsRegionBase,
    };
}
