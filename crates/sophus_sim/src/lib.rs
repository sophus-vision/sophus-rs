#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![no_std]

//! Simulator

/// camera simulator - camera image renderer
pub mod camera_simulator;

/// sophus sim prelude
pub mod prelude {
    pub use sophus_autodiff::prelude::*;
    pub use sophus_image::prelude::*;
    pub use sophus_lie::prelude::*;
    pub use sophus_opt::prelude::*;
}
