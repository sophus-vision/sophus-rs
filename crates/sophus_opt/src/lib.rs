#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]
#![cfg_attr(any(docsrs, nightly), feature(doc_auto_cfg))]

#[doc = include_str!(concat!("../",  core::env!("CARGO_PKG_README")))]
#[cfg(doctest)]
pub struct ReadmeDoctests;

mod asserts;

/// Block vectors, block matrices and utilities
pub mod block;
/// Example problems
pub mod example_problems;
/// Non-linear least squares optimization
pub mod nlls;
/// Robust kernel functions
pub mod robust_kernel;
/// Decision variables
pub mod variables;
/// sophus_opt prelude.
///
/// It is recommended to import this prelude when working with `sophus_opt types:
///
/// ```
/// use sophus_opt::prelude::*;
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
    pub use sophus_image::prelude::*;
    pub use sophus_lie::prelude::*;
    pub use sophus_sensor::prelude::*;

    pub use crate::{
        nlls::{
            HasEqConstraintResidualFn,
            HasResidualFn,
            IsCostFn,
            IsEqConstraintsFn,
            IsEvaluatedCost,
            MakeEvaluatedCostTerm,
            MakeEvaluatedEqConstraint,
        },
        robust_kernel::IsRobustKernel,
    };
}
