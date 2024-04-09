#![feature(portable_simd)]
#![deny(missing_docs)]

//! # Sensor (aka camera) module

/// Distortion lookup table
pub mod distortion_table;

/// A type-erased camera struct
pub mod dyn_camera;

/// A generic camera model
pub mod camera;

/// Projection models
pub mod camera_enum;

/// Projection models
pub mod projections;

/// Distortion models
pub mod distortions;

/// Sensor traits
pub mod traits;
