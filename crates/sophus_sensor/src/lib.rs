#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![no_std]

//! Sensor (aka camera) crate - part of the sophus-rs project

/// Distortion lookup table
pub mod distortion_table;

/// A type-erased camera struct
pub mod dyn_camera;
pub use crate::dyn_camera::DynCamera;

/// A generic camera model
pub mod camera;
pub use crate::camera::Camera;

/// Projection models
pub mod camera_enum;
pub use crate::camera_enum::perspective_camera::{
    BrownConradyCamera,
    KannalaBrandtCamera,
    PinholeCamera,
};

/// Projection models
pub mod projections;

/// Distortion models
pub mod distortions;

/// Sensor traits
pub mod traits;

/// sophus sensor prelude
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
