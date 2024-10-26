#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![no_std]
//! image crate - part of the sophus-rs project

#[cfg(feature = "std")]
extern crate std;

/// image with shared ownership
pub mod arc_image;
/// color maps
pub mod color_map;
/// image view
pub mod image_view;
/// image of intensity (aka percentage) values
pub mod intensity_image;
/// bilinear interpolation
pub mod interpolation;
/// image io
pub mod io;
/// mutable image
pub mod mut_image;
/// mutable image view
pub mod mut_image_view;

pub use crate::arc_image::ArcImage;
pub use crate::image_view::ImageView;
pub use crate::interpolation::interpolate;
pub use crate::mut_image::MutImage;
pub use crate::mut_image_view::MutImageView;

/// Image size
#[derive(Debug, Copy, Clone, Default)]
pub struct ImageSize {
    /// Width of the image - number of columns
    pub width: usize,
    /// Height of the image - number of rows
    pub height: usize,
}

impl ImageSize {
    /// Create a new image size from width and height
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Get the area of the image - width * height
    pub fn area(&self) -> usize {
        self.width * self.height
    }

    /// Get the aspect ratio of the image - width / height
    pub fn aspect_ratio(&self) -> f32 {
        if self.area() == 0 {
            return 1.0;
        }
        self.width as f32 / self.height as f32
    }
}

impl From<[usize; 2]> for ImageSize {
    /// We are converting from Tensor (and matrix) convention (d0 = rows, d1 = cols)
    /// to Matrix convention (d0 = width = cols, d1 = height = rows)
    fn from(rows_cols: [usize; 2]) -> Self {
        ImageSize {
            width: rows_cols[1],
            height: rows_cols[0],
        }
    }
}

impl From<ImageSize> for [usize; 2] {
    /// We are converting from Image Indexing Convention (d0 = width = cols, d1 = height = rows)
    /// to tensor (and matrix) convention  (d0 = rows, d1 = cols).
    fn from(image_size: ImageSize) -> Self {
        [image_size.height, image_size.width]
    }
}

impl PartialEq for ImageSize {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height
    }
}

/// sophus_image prelude
pub mod prelude {
    pub use crate::image_view::IsImageView;
    pub use crate::intensity_image::dyn_intensity_image::DynIntensityMutImage;
    pub use crate::mut_image_view::IsMutImageView;
    pub use sophus_core::prelude::*;
}
