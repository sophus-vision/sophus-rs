#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![no_std]
#![doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]
#![cfg_attr(any(docsrs, nightly), feature(doc_auto_cfg))]

#[doc = include_str!(concat!("../",  core::env!("CARGO_PKG_README")))]
#[cfg(doctest)]
pub struct ReadmeDoctests;

#[cfg(feature = "std")]
extern crate std;

/// Distortion models
pub mod distortions;
/// Projection models
pub mod projections;
/// Sensor traits
pub mod traits;
/// sophus_sensor prelude.
///
/// It is recommended to import this prelude when working with `sophus_sensor types:
///
/// ```
/// use sophus_sensor::prelude::*;
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
    pub use sophus_geo::prelude::*;
    pub use sophus_image::prelude::*;

    pub use crate::traits::{
        IsCamera,
        IsPerspectiveCamera,
        IsProjection,
    };
}

mod camera;
mod camera_enum;
mod distortion_table;
mod dyn_camera;

pub use crate::{
    camera::*,
    camera_enum::{
        BrownConradyCamera,
        KannalaBrandtCamera,
        PinholeCamera,
        *,
    },
    distortion_table::*,
    dyn_camera::*,
};
