#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![no_std]
#![doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]
#![cfg_attr(nightly, feature(doc_auto_cfg))]

#[doc = include_str!(concat!("../",  core::env!("CARGO_PKG_README")))]
#[cfg(doctest)]
pub struct ReadmeDoctests;

#[cfg(feature = "std")]
extern crate std;

mod arc_image;
mod image_view;
mod intensity_image;
mod interpolation;
mod mut_image;
mod mut_image_view;

/// color maps
pub mod color_map;
/// image io
pub mod io;

/// sophus_image prelude.
///
/// It is recommended to import this prelude when working with `sophus_image types:
///
/// ```
/// use sophus_image::prelude::*;
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
    pub use sophus_tensor::prelude::*;

    pub use crate::{
        image_view::IsImageView,
        mut_image::IsMutImage,
        mut_image_view::IsMutImageView,
        IsIntensityArcImage,
    };
}

use sophus_autodiff::linalg::SVec;

pub use crate::{
    arc_image::GenArcImage,
    image_view::GenImageView,
    intensity_image::{
        dyn_intensity_image::*,
        intensity_arc_image::*,
        intensity_image_view::*,
        intensity_mut_image::*,
        intensity_pixel::*,
        intensity_scalar::*,
    },
    interpolation::{
        interpolate_f32,
        interpolate_xf32,
    },
    mut_image::GenMutImage,
    mut_image_view::GenMutImageView,
};

/// Image view of scalar values
pub type ImageView<'a, Scalar> = GenImageView<'a, 2, 0, Scalar, Scalar, 1, 1>;
/// Image view of vector values
///
/// Here, R indicates the number of rows in the vector
pub type ImageViewR<'a, Scalar, const ROWS: usize> =
    GenImageView<'a, 3, 1, Scalar, SVec<Scalar, ROWS>, ROWS, 1>;
/// Image view of u8 values
pub type ImageViewU8<'a> = ImageView<'a, u8>;
/// Image view of u16 values
pub type ImageViewU16<'a> = ImageView<'a, u16>;
/// Image view of f32 values
pub type ImageViewF32<'a> = ImageView<'a, f32>;
/// Image view of u8 2-vectors
pub type ImageView2U8<'a> = ImageViewR<'a, u8, 2>;
/// Image view of u16 2-vectors
pub type ImageView2U16<'a> = ImageViewR<'a, u16, 2>;
/// Image view of f32 2-vectors
pub type ImageView2F32<'a> = ImageViewR<'a, f32, 2>;
/// Image view of u8 3-vectors
pub type ImageView3U8<'a> = ImageViewR<'a, u8, 3>;
/// Image view of u16 3-vectors
pub type ImageView3U16<'a> = ImageViewR<'a, u16, 3>;
/// Image view of f32 3-vectors
pub type ImageView3F32<'a> = ImageViewR<'a, f32, 3>;
/// Image view of u8 4-vectors
pub type ImageView4U8<'a> = ImageViewR<'a, u8, 4>;
/// Image view of u16 4-vectors
pub type ImageView4U16<'a> = ImageViewR<'a, u16, 4>;
/// Image view of f32 4-vectors
pub type ImageView4F32<'a> = ImageViewR<'a, f32, 4>;

/// Mutable image view of scalar values
pub type MutImageView<'a, Scalar> = GenMutImageView<'a, 2, 0, Scalar, Scalar, 1, 1>;

/// Image of scalar values
pub type ArcImage<Scalar> = GenArcImage<2, 0, Scalar, Scalar, 1, 1>;
/// Image of vector values
///
/// Here, R indicates the number of rows in the vector
pub type ArcImageR<Scalar, const R: usize> = GenArcImage<3, 1, Scalar, SVec<Scalar, R>, R, 1>;
/// Image of u8 scalars
pub type ArcImageU8 = ArcImage<u8>;
/// Image of u16 scalars
pub type ArcImageU16 = ArcImage<u16>;
/// Image of f32 scalars
pub type ArcImageF32 = ArcImage<f32>;
/// Image of u8 2-vectors
pub type ArcImage2U8 = ArcImageR<u8, 2>;
/// Image of u16 2-vectors
pub type ArcImage2U16 = ArcImageR<u16, 2>;
/// Image of f32 2-vectors
pub type ArcImage2F32 = ArcImageR<f32, 2>;
/// Image of u8 3-vectors
pub type ArcImage3U8 = ArcImageR<u8, 3>;
/// Image of u16 3-vectors
pub type ArcImage3U16 = ArcImageR<u16, 3>;
/// Image of f32 3-vectors
pub type ArcImage3F32 = ArcImageR<f32, 3>;
/// Image of u8 4-vectors
pub type ArcImage4U8 = ArcImageR<u8, 4>;
/// Image of u16 4-vectors
pub type ArcImage4U16 = ArcImageR<u16, 4>;
/// Image of f32 4-vectors
pub type ArcImage4F32 = ArcImageR<f32, 4>;

/// Mutable image of scalar values
pub type MutImage<Scalar> = GenMutImage<2, 0, Scalar, Scalar, 1, 1>;
/// Mutable image of vector values
///
/// Here, R indicates the number of rows in the vector
pub type MutImageR<Scalar, const ROWS: usize> =
    GenMutImage<3, 1, Scalar, SVec<Scalar, ROWS>, ROWS, 1>;
/// Mutable image of u8 scalars
pub type MutImageU8 = MutImage<u8>;
/// Mutable image of u16 scalars
pub type MutImageU16 = MutImage<u16>;
/// Mutable image of f32 scalars
pub type MutImageF32 = MutImage<f32>;
/// Mutable image of u8 2-vectors
pub type MutImage2U8 = MutImageR<u8, 2>;
/// Mutable image of u16 2-vectors
pub type MutImage2U16 = MutImageR<u16, 2>;
/// Mutable image of f32 2-vectors
pub type MutImage2F32 = MutImageR<f32, 2>;
/// Mutable image of u8 3-vectors
pub type MutImage3U8 = MutImageR<u8, 3>;
/// Mutable image of u16 3-vectors
pub type MutImage3U16 = MutImageR<u16, 3>;
/// Mutable image of f32 3-vectors
pub type MutImage3F32 = MutImageR<f32, 3>;
/// Mutable image of u8 4-vectors
pub type MutImage4U8 = MutImageR<u8, 4>;
/// Mutable image of u16 4-vectors
pub type MutImage4U16 = MutImageR<u16, 4>;
/// Mutable image of f32 4-vectors
pub type MutImage4F32 = MutImageR<f32, 4>;

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
