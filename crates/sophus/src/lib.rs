#![feature(portable_simd)]
#![allow(clippy::needless_range_loop)]

pub use sophus_core as core;
pub use sophus_image as image;
pub use sophus_lie as lie;
pub use sophus_opt as opt;
pub use sophus_sensor as sensor;

pub mod viewer;

pub use hollywood;

pub mod prelude {
    pub use crate::core::prelude::*;
    pub use crate::image::prelude::*;
    pub use crate::lie::prelude::*;
    pub use crate::opt::prelude::*;
}
