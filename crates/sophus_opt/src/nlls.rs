/// equality/inequality constraints
pub mod constraint;
/// functor library
pub mod functor_library;
/// Linear system
pub mod linear_system;
/// Cost functions, terms, residuals etc.
pub mod quadratic_cost;

use core::fmt::Debug;

use linear_system::{
    eq_system::EqSystem,
    quadratic_cost_system::CostSystem,
    solvers::SolveError,
};
use log::{
    debug,
    info,
};

use crate::{
    block::{
        block_vector::BlockVector,
        symmetric_block_sparse_matrix_builder::SymmetricBlockSparseMatrixBuilder,
    },
    nlls::{
        constraint::eq_constraint_fn::IsEqConstraintsFn,
        linear_system::{
            solvers::sparse_ldlt::SparseLdltParams,
            EvalMode,
            LinearSystem,
        },
        quadratic_cost::cost_fn::IsCostFn,
    },
    variables::var_families::VarFamilies,
};

extern crate alloc;

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
) -> LinearSystem {
    let eval_mode = EvalMode::CalculateDerivatives;
    cost_system.eval(variables, eval_mode, params);
    eq_system.eval(variables, eval_mode, params);

    LinearSystem::from_families_costs_and_constraints(
        variables,
        cost_system,
        eq_system,
        params.solver,
    )
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

/// Non-linear least squares optimization
pub fn optimize(
    variables: VarFamilies,
    cost_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsCostFn>>,
    params: OptParams,
) -> Result<OptimizationSolution, SolveError> {
    optimize_with_eq_constraints(variables, cost_fns, vec![], params)
}

// calculate the merit value - at this point just the cost
pub(crate) fn calc_merit(
    variables: &VarFamilies,
    cost_system: &mut CostSystem,
    _eq_system: &mut EqSystem,
    params: OptParams,
) -> f64 {
    cost_system.eval(variables, EvalMode::DontCalculateDerivatives, params);
    let mut c = 0.0;
    for cost in cost_system.evaluated_costs.iter() {
        c += cost.calc_square_error();
    }
    c
}

/// Non-linear least squares optimization with equality constraints
pub fn optimize_with_eq_constraints(
    mut variables: VarFamilies,
    cost_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsCostFn>>,
    eq_constraints_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsEqConstraintsFn>>,
    params: OptParams,
) -> Result<OptimizationSolution, SolveError> {
    let mut cost_system = CostSystem::new(&variables, cost_fns, params);
    let mut eq_system = EqSystem::new(&variables, eq_constraints_fns, params);

    let mut merit = calc_merit(&variables, &mut cost_system, &mut eq_system, params);
    let initial_merit = merit;

    for i in 0..params.num_iterations {
        debug!("lm-damping: {}", cost_system.lm_damping);
        let mut linear_system = evaluate_cost_and_build_linear_system(
            &variables,
            &mut cost_system,
            &mut eq_system,
            params,
        );

        let delta = linear_system.solve()?;
        let updated_families = variables.update(&delta);
        let updated_lambdas = eq_system.update_lambdas(&variables, &delta);

        let new_merit = calc_merit(&updated_families, &mut cost_system, &mut eq_system, params);

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
    info!("e^2: {:?} -> {:?}", initial_merit, merit);

    // Calculate the final gradient and hessian. This is not strictly necessary, but it is useful
    // for debugging, and not too expensive.
    // TODO: Consider making this optional, i.e. only return the final gradient and hessian if
    // requested.
    let final_linear_system =
        evaluate_cost_and_build_linear_system(&variables, &mut cost_system, &mut eq_system, params);

    Ok(OptimizationSolution {
        variables,
        final_cost: merit,
        final_neg_gradient: final_linear_system.neg_gradient,
        final_hessian_plus_damping: final_linear_system.sparse_hessian_plus_damping,
    })
}
