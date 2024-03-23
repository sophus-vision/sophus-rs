#![deny(missing_docs)]

//! # Sensor (aka camera) module

/// Affine distortion - for pinhole cameras
pub mod affine;
/// Distortion lookup table
pub mod distortion_table;
/// A type-erased camera struct
pub mod dyn_camera;
/// A generalized camera enum
pub mod general_camera;
/// A generic camera struct
pub mod generic_camera;
/// Kannala-Brandt distortion - for fisheye cameras
pub mod kannala_brandt;
/// Orthographic camera
pub mod ortho_camera;
/// Perspective camera
pub mod perspective_camera;
/// Sensor traits
pub mod traits;
