extern crate alloc;

mod constraint;
mod cost;
mod functor_library;
mod linear_system;

use core::fmt::Debug;

pub use constraint::{
    eq_constraint::*,
    eq_constraint_fn::*,
    evaluated_eq_constraint::*,
    evaluated_eq_set::*,
};
pub use cost::{
    cost_fn::*,
    cost_term::*,
    evaluated_cost::*,
    evaluated_term::*,
};
pub(crate) use linear_system::solvers::*;
pub use linear_system::{
    cost_system::*,
    eq_system::*,
    *,
};
use log::{
    debug,
    info,
};
use snafu::Snafu;

use self::sparse_ldlt::SparseLdltParams;
pub use crate::nlls::functor_library::{
    costs,
    eq_constraints,
};
use crate::{
    block::{
        block_vector::BlockVector,
        symmetric_block_sparse_matrix_builder::SymmetricBlockSparseMatrixBuilder,
    },
    variables::VarFamilies,
};

/// Linear solver type
#[derive(Copy, Clone, Debug)]
pub enum LinearSolverType {
    /// Sparse LDLT solver (using faer crate)
    SparseLdlt(SparseLdltParams),
    /// Sparse partial pivoting LU solver (using faer crate)
    SparseLu,
    /// Sparse QR solver (using faer crate)
    SparseQr,
    /// Dense full-pivot. LU solver (using nalgebra crate)
    DenseLu,
}

impl LinearSolverType {
    /// Get all available solvers
    pub fn all_solvers() -> Vec<LinearSolverType> {
        vec![
            LinearSolverType::SparseLdlt(Default::default()),
            LinearSolverType::SparseLu,
            LinearSolverType::SparseQr,
            LinearSolverType::DenseLu,
        ]
    }

    /// Get all sparse solvers
    pub fn sparse_solvers() -> Vec<LinearSolverType> {
        vec![
            LinearSolverType::SparseLdlt(Default::default()),
            LinearSolverType::SparseLu,
            LinearSolverType::SparseQr,
        ]
    }

    /// Get solvers which can be used for indefinite systems
    pub fn indefinite_solvers() -> Vec<LinearSolverType> {
        vec![
            LinearSolverType::SparseLu,
            LinearSolverType::SparseQr,
            LinearSolverType::DenseLu,
        ]
    }
}

impl Default for LinearSolverType {
    fn default() -> Self {
        LinearSolverType::SparseLdlt(Default::default())
    }
}

/// Optimization parameters
#[derive(Copy, Clone, Debug)]
pub struct OptParams {
    /// number of iterations
    ///
    /// This is currently the only stopping criterion.
    /// TODO: Add additional stopping criteria.
    pub num_iterations: usize,
    /// Initial value of the Levenberg-Marquardt regularization parameter
    pub initial_lm_damping: f64,
    /// whether to use parallelization
    pub parallelize: bool,
    /// linear solver type
    pub solver: LinearSolverType,
}

impl Default for OptParams {
    fn default() -> Self {
        Self {
            num_iterations: 20,
            initial_lm_damping: 10.0,
            parallelize: true,
            solver: Default::default(),
        }
    }
}

fn evaluate_cost_and_build_linear_system(
    variables: &VarFamilies,
    cost_system: &mut CostSystem,
    eq_system: &mut EqSystem,
    params: OptParams,
) -> Result<LinearSystem, NllsError> {
    let eval_mode = EvalMode::CalculateDerivatives;
    cost_system
        .eval(variables, eval_mode, params)
        .map_err(|e| NllsError::NllsCostSystemError { details: e })?;
    eq_system
        .eval(variables, eval_mode, params)
        .map_err(|e| NllsError::NllsEqConstraintSystemError { details: e })?;

    Ok(LinearSystem::from_families_costs_and_constraints(
        variables,
        cost_system,
        eq_system,
        params.solver,
        params.parallelize,
    ))
}

/// Optimization solution
pub struct OptimizationSolution {
    /// The optimized variables
    pub variables: VarFamilies,
    /// The final cost
    pub final_cost: f64,
    /// the gradient vector
    pub final_neg_gradient: BlockVector,
    /// the hessian matrix
    pub final_hessian_plus_damping: SymmetricBlockSparseMatrixBuilder,
}

/// Linear solver error
#[derive(Snafu, Debug)]
pub enum NllsError {
    /// Sparse LDLt error
    #[snafu(display("sparse LDLt error {}", details))]
    SparseLdltError {
        /// details
        details: SparseSolverError,
    },
    /// Sparse LU error
    #[snafu(display("sparse LU error {}", details))]
    SparseLuError {
        /// details
        details: SparseSolverError,
    },
    /// Sparse QR error
    #[snafu(display("sparse QR error {}", details))]
    SparseQrError {
        /// details
        details: SparseSolverError,
    },
    /// Dense LU error
    #[snafu(display("dense LU solve failed"))]
    DenseLuError,

    /// Quadratic cost system error
    #[snafu(display("{}", details))]
    NllsCostSystemError {
        /// details
        details: CostError,
    },
    /// Eq constraint system error
    #[snafu(display("{}", details))]
    NllsEqConstraintSystemError {
        /// details
        details: EqConstraintError,
    },
}

/// Non-linear least squares optimization
pub fn optimize_nlls(
    variables: VarFamilies,
    cost_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsCostFn>>,
    params: OptParams,
) -> Result<OptimizationSolution, NllsError> {
    optimize_nlls_with_eq_constraints(variables, cost_fns, vec![], params)
}

// calculate the merit value - at this point just the cost
pub(crate) fn calc_merit(
    variables: &VarFamilies,
    cost_system: &mut CostSystem,
    _eq_system: &mut EqSystem,
    params: OptParams,
) -> Result<f64, NllsError> {
    cost_system
        .eval(variables, EvalMode::DontCalculateDerivatives, params)
        .map_err(|e| NllsError::NllsCostSystemError { details: e })?;
    let mut c = 0.0;
    for cost in cost_system.evaluated_costs.iter() {
        c += cost.calc_square_error();
    }
    Ok(c)
}

/// Non-linear least squares optimization with equality constraints
pub fn optimize_nlls_with_eq_constraints(
    mut variables: VarFamilies,
    cost_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsCostFn>>,
    eq_constraints_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsEqConstraintsFn>>,
    params: OptParams,
) -> Result<OptimizationSolution, NllsError> {
    let mut cost_system = CostSystem::new(&variables, cost_fns, params)
        .map_err(|e| NllsError::NllsCostSystemError { details: e })?;
    let mut eq_system = EqSystem::new(&variables, eq_constraints_fns, params)
        .map_err(|e| NllsError::NllsEqConstraintSystemError { details: e })?;

    let mut merit = calc_merit(&variables, &mut cost_system, &mut eq_system, params)?;
    let initial_merit = merit;

    for i in 0..params.num_iterations {
        debug!("lm-damping: {}", cost_system.lm_damping);
        let mut linear_system = evaluate_cost_and_build_linear_system(
            &variables,
            &mut cost_system,
            &mut eq_system,
            params,
        )?;

        let delta = linear_system.solve()?;
        let updated_families = variables.update(&delta);
        let updated_lambdas = eq_system.update_lambdas(&variables, &delta);

        let new_merit = calc_merit(&updated_families, &mut cost_system, &mut eq_system, params)?;

        if new_merit < merit {
            cost_system.lm_damping *= 0.0333;
            variables = updated_families;
            eq_system.lambda = updated_lambdas;
            merit = new_merit;
        } else {
            cost_system.lm_damping *= 2.0;
        }

        info!(
            "i: {:?}, lm-damping: {:?}, merit {:?}, (new_merit {:?})",
            i, cost_system.lm_damping, merit, new_merit
        );
    }
    info!("e^2: {initial_merit:?} -> {merit:?}");

    // Calculate the final gradient and hessian. This is not strictly necessary, but it is useful
    // for debugging, and not too expensive.
    // TODO: Consider making this optional, i.e. only return the final gradient and hessian if
    // requested.
    let final_linear_system = evaluate_cost_and_build_linear_system(
        &variables,
        &mut cost_system,
        &mut eq_system,
        params,
    )?;

    Ok(OptimizationSolution {
        variables,
        final_cost: merit,
        final_neg_gradient: final_linear_system.neg_gradient,
        final_hessian_plus_damping: final_linear_system.sparse_hessian_plus_damping,
    })
}
