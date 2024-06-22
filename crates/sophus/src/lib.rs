#![cfg_attr(feature = "simd", feature(portable_simd))]
#![allow(clippy::needless_range_loop)]

#![doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]


pub mod examples;

#[doc(inline)]
pub use sophus_core as core;
#[doc(inline)]
pub use sophus_image as image;
#[doc(inline)]
pub use sophus_lie as lie;
#[doc(inline)]
pub use sophus_opt as opt;
#[doc(inline)]
pub use sophus_sensor as sensor;
#[doc(inline)]
pub use sophus_viewer as viewer;

pub use hollywood;
pub use nalgebra;

pub mod prelude {
    pub use crate::core::prelude::*;
    pub use crate::image::prelude::*;
    pub use crate::lie::prelude::*;
    pub use crate::opt::prelude::*;
}
