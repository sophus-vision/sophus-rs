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
    pub use sophus_autodiff::prelude::*;
    pub use sophus_image::prelude::*;
    pub use sophus_lie::prelude::*;
    pub use sophus_sensor::prelude::*;

    pub use crate::{
        nlls::{
            constraint::{
                eq_constraint::IsEqConstraint,
                eq_constraint_fn::IsEqConstraintsFn,
                evaluated_eq_constraint::MakeEvaluatedEqConstraint,
            },
            cost::{
                cost_fn::IsCostFn,
                cost_term::IsCostTerm,
                evaluated_cost::IsEvaluatedCost,
                evaluated_term::MakeEvaluatedCostTerm,
            },
        },
        robust_kernel::IsRobustKernel,
    };
}
