extern crate alloc;

mod constraint;
mod cost;
/// Hessian-plus-LM-damping wrapper.
pub(crate) mod damped_hessian;
/// Filter for the filter line-search IPM (Wächter & Biegler, 2006).
pub(crate) mod filter;
mod functor_library;
/// Inequality constraint strategies (IPM, SQP, Elastic).
pub mod ineq_strategy;
mod linear_system;
/// Optimizer struct, constructors, accessors, and the main step() method.
pub mod optimizer;
/// Helper free functions (merit, line search, gain ratio, pattern builders, etc.).
pub(crate) mod optimizer_helpers;
/// Inner step free functions (step_nlls_inner, step_ipm_inner, step_sqp_inner, etc.).
pub(crate) mod optimizer_step;
/// Types, enums, and structs for the optimizer (OptParams, StepInfo, NllsError, etc.).
pub mod optimizer_types;

pub use constraint::{
    eq_constraint::*,
    eq_constraint_fn::*,
    evaluated_eq_constraint::*,
    evaluated_eq_set::*,
    ineq_constraint::*,
};
pub use cost::{
    cost_fn::*,
    cost_term::*,
    evaluated_cost::*,
    evaluated_term::*,
    ineq_barrier_fn::{
        ConstraintLinearization,
        IneqBarrierCostFn,
        IsIneqBarrierFn,
    },
};
pub use ineq_strategy::{
    ElasticState,
    IneqStrategy,
    IpmState,
    SqpState,
};
pub use linear_system::{
    cost_system::*,
    eq_system::*,
    *,
};
pub use optimizer::Optimizer;
pub use optimizer_types::*;

pub use crate::nlls::functor_library::{
    costs,
    eq_constraints,
    ineq_constraints,
};
// Re-export crate types used by sub-modules via `use super::*`.
pub(crate) use crate::{
    nlls::ineq_strategy::FrozenBarrierEvaluatedCost,
    variables::{
        VarFamilies,
        VarKind,
    },
};

// ── Free-function entry points ──────────────────────────────────────────────

/// Run NLLS optimization to completion, with optional equality constraints.
///
/// Pass `vec![]` for `eq_fns` if no equality constraints are needed.
pub fn optimize_nlls(
    variables: VarFamilies,
    cost_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsCostFn>>,
    eq_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsEqConstraintsFn>>,
    params: OptParams,
) -> Result<OptimizationSolution, NllsError> {
    optimize_nlls_with_inequality(variables, cost_fns, eq_fns, vec![], params)
}

/// Run NLLS optimization with inequality (and optional equality) constraints.
///
/// The inequality method (IPM or SQP) is selected via `params.ineq_method`.
pub fn optimize_nlls_with_inequality(
    variables: VarFamilies,
    cost_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsCostFn>>,
    eq_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsEqConstraintsFn>>,
    ineq_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsIneqBarrierFn>>,
    params: OptParams,
) -> Result<OptimizationSolution, NllsError> {
    let problem = OptProblem {
        costs: cost_fns,
        eq_constraints: eq_fns,
        ineq_constraints: ineq_fns,
    };
    let mut opt = Optimizer::new(variables, problem, params)?;
    loop {
        if opt.step()?.termination.is_some() {
            break;
        }
    }
    opt.build_solution()
}
