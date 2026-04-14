#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]
#![cfg_attr(nightly, allow(unused_features))]
#![cfg_attr(nightly, feature(doc_cfg))]

#[doc = include_str!(concat!("../",  core::env!("CARGO_PKG_README")))]
#[cfg(doctest)]
pub struct ReadmeDoctests;

use sophus_solver::matrix::PartitionSpec;

/// Create a variable partition spec (default elimination order).
pub fn variable_partition(block_count: usize, block_dim: usize) -> PartitionSpec {
    PartitionSpec {
        block_count,
        block_dim,
        eliminate_last: false,
    }
}

/// Create a constraint partition spec (eliminated last in LDLᵀ ordering).
///
/// Use for equality-constraint multiplier partitions in KKT systems.
/// Constraint partitions have negative-definite diagonal (`-εI`) and must be
/// eliminated after variable partitions to avoid tiny pivots.
pub fn constraint_partition(block_count: usize, block_dim: usize) -> PartitionSpec {
    PartitionSpec {
        block_count,
        block_dim,
        eliminate_last: true,
    }
}

/// Block derivatives
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
    pub use sophus_solver::prelude::*;

    pub use crate::{
        nlls::{
            HasEqConstraintResidualFn,
            HasIneqConstraintFn,
            HasResidualFn,
            IsCostFn,
            IsEqConstraintsFn,
            IsEvaluatedCost,
            MakeEvaluatedCostTerm,
            MakeEvaluatedEqConstraint,
            MakeEvaluatedIneqConstraint,
        },
        robust_kernel::IsRobustKernel,
    };
}
