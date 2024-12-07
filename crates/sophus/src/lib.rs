#![cfg_attr(feature = "simd", feature(portable_simd))]
#![allow(clippy::needless_range_loop)]
#![doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]

#[doc(inline)]
pub use sophus_core as core;
#[doc(inline)]
pub use sophus_image as image;
#[doc(inline)]
pub use sophus_lie as lie;
#[doc(inline)]
pub use sophus_opt as opt;
#[doc(inline)]
pub use sophus_renderer as renderer;
#[doc(inline)]
pub use sophus_sensor as sensor;
#[doc(inline)]
pub use sophus_sim as sim;
#[doc(inline)]
pub use sophus_viewer as viewer;

pub mod examples;
pub use eframe;
pub use thingbuf;

pub use nalgebra;
pub use ndarray;

pub mod prelude {
    pub use crate::core::prelude::*;
    pub use crate::image::prelude::*;
    pub use crate::lie::prelude::*;
    pub use crate::opt::prelude::*;
}
