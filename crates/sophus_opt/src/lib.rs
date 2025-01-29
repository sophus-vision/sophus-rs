#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]

//! # Sophus optimization crate - part of the sophus-rs project

/// Some assert macros
pub mod asserts;
/// Block vector and matrix operations
pub mod block;
/// Example problems
pub mod example_problems;
/// Non-linear least squares optimization
pub mod nlls;
/// Robust kernel functions
pub mod robust_kernel;
/// Decision variables
pub mod variables;

/// Sophus optimization prelude
pub mod prelude {
    pub use crate::nlls::constraint::eq_constraint::IsEqConstraint;
    pub use crate::nlls::constraint::eq_constraint_fn::IsEqConstraintsFn;
    pub use crate::nlls::constraint::evaluated_constraint::MakeEvaluatedConstraint;
    pub use crate::nlls::quadratic_cost::cost_fn::IsCostFn;
    pub use crate::nlls::quadratic_cost::cost_term::IsCostTerm;
    pub use crate::nlls::quadratic_cost::evaluated_cost::IsEvaluatedCost;
    pub use crate::nlls::quadratic_cost::evaluated_term::MakeEvaluatedCostTerm;
    pub use crate::robust_kernel::IsRobustKernel;
    pub use sophus_autodiff::prelude::*;
    pub use sophus_image::prelude::*;
    pub use sophus_lie::prelude::*;
    pub use sophus_sensor::prelude::*;
}
