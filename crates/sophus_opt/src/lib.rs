#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]

//! # Non-linear least squares optimization crate - part of the sophus-rs project

/// Block vector and matrix operations
pub mod block;
/// Example problems
pub mod example_problems;
/// Entry point for the non-linear least squares optimization
pub mod nlls;
/// Cost functions, terms, residuals etc.
pub mod quadratic_cost;
/// Robust kernel functions
pub mod robust_kernel;
/// Linear solvers
pub mod solvers;
/// Decision variables
pub mod variables;

/// Sophus optimization prelude
pub mod prelude {
    pub use crate::robust_kernel::IsRobustKernel;
    pub use sophus_autodiff::prelude::*;
    pub use sophus_image::prelude::*;
    pub use sophus_lie::prelude::*;
    pub use sophus_sensor::prelude::*;
}
