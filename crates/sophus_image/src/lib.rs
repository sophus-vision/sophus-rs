#![deny(missing_docs)]
//! # image module

/// image with shared ownership
pub mod arc_image;
/// image view
pub mod image_view;
/// image of intensity (aka percentage) values
pub mod intensity_image;
/// bilinear interpolation
pub mod interpolation;
/// mutable image
pub mod mut_image;
/// mutable image view
pub mod mut_image_view;
/// png image io
pub mod png;
