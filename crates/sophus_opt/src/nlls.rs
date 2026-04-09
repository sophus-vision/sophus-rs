extern crate alloc;

mod constraint;
mod cost;
/// Hessian-plus-LM-damping wrapper.
pub(crate) mod damped_hessian;
mod functor_library;
/// Inequality constraint strategies (IPM, SQP).
pub mod ineq_strategy;
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
pub(crate) use damped_hessian::DampedHessian;
pub use ineq_strategy::IneqStrategy;
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
use sophus_solver::{
    LinearSolverEnum,
    LinearSolverError,
    matrix::{
        block::BlockVector,
        block_sparse::BlockSparseSymmetricMatrixPattern,
    },
};

pub use crate::nlls::functor_library::{
    costs,
    eq_constraints,
};
use crate::variables::{
    VarFamilies,
    VarKind,
};
/// Information returned by each `Optimizer::step()` call.
#[derive(Clone, Debug)]
pub struct StepInfo {
    /// Current merit value (smooth cost + barrier terms).
    pub merit: f64,
    /// Smooth cost only (without barrier terms).
    pub smooth_cost: f64,
    /// Termination reason. `None` means keep going. `Some(reason)` means done.
    pub termination: Option<TerminationReason>,
}

/// Why the optimizer stopped.
///
/// Each variant carries the actual measured value and the threshold it was
/// compared against, so the Display output shows e.g.
/// `"function tolerance: 3.2e-11 < 1e-10"`.
///
/// Based on convergence criteria from Ceres Solver, IPOPT, and Nocedal & Wright.
#[derive(Clone, Debug, PartialEq)]
pub enum TerminationReason {
    /// Relative cost decrease below threshold: `|Δf| / (|f| + ε) < function_tolerance`.
    FunctionTolerance {
        /// Measured relative improvement.
        rel_improvement: f64,
        /// Threshold from OptParams.
        threshold: f64,
    },
    /// Maximum gradient component below threshold: `‖∇f‖∞ < gradient_tolerance`.
    GradientTolerance {
        /// Measured max gradient component.
        gradient_max: f64,
        /// Threshold from OptParams.
        threshold: f64,
    },
    /// Max step component below threshold: `max(|Δx_i|) < parameter_tolerance`.
    ParameterTolerance {
        /// Measured max step component.
        max_step: f64,
        /// Threshold from OptParams.
        threshold: f64,
    },
    /// Maximum number of iterations reached.
    MaxIterations,
    /// Too many consecutive rejected steps (LM damping saturated).
    MaxConsecutiveRejects {
        /// Number of consecutive rejects.
        count: usize,
    },
    /// Linear solver failed (singular or numerically unstable system).
    LinearSolverFailed,
}

impl core::fmt::Display for TerminationReason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TerminationReason::FunctionTolerance {
                rel_improvement,
                threshold,
            } => {
                write!(f, "function tol: {rel_improvement:.2e} < {threshold:.0e}")
            }
            TerminationReason::GradientTolerance {
                gradient_max,
                threshold,
            } => {
                write!(f, "gradient tol: {gradient_max:.2e} < {threshold:.0e}")
            }
            TerminationReason::ParameterTolerance {
                max_step,
                threshold,
            } => {
                write!(f, "parameter tol: {max_step:.2e} < {threshold:.0e}")
            }
            TerminationReason::MaxIterations => write!(f, "max iterations"),
            TerminationReason::MaxConsecutiveRejects { count } => {
                write!(f, "max consecutive rejects ({count})")
            }
            TerminationReason::LinearSolverFailed => write!(f, "linear solver failed"),
        }
    }
}

/// Optimization parameters.
///
/// Convergence criteria follow Ceres Solver conventions:
/// - `function_tolerance`: relative cost change threshold.
/// - `gradient_tolerance`: max gradient component threshold.
/// - `parameter_tolerance`: relative step size threshold.
/// - `max_consecutive_rejects`: LM damping saturation.
#[derive(Copy, Clone, Debug)]
pub struct OptParams {
    /// Maximum number of iterations.
    pub num_iterations: usize,
    /// Initial value of the Levenberg-Marquardt regularization parameter.
    pub initial_lm_damping: f64,
    /// Whether to use parallelization.
    pub parallelize: bool,
    /// Linear solver type.
    pub solver: LinearSolverEnum,
    /// If true, skip the final hessian/gradient evaluation after the last iteration.
    pub skip_final_hessian: bool,
    /// Stop when relative cost improvement drops below this value.
    /// `|Δf| / (|f| + MIN_POSITIVE) < function_tolerance`.
    /// Set to `0.0` to disable.
    pub function_tolerance: f64,
    /// Stop when the max absolute gradient component drops below this value.
    /// `‖∇f‖∞ < gradient_tolerance`.
    /// Set to `0.0` to disable.
    pub gradient_tolerance: f64,
    /// Stop when the maximum tangent-space step component drops below this value.
    /// `max(|Δx_i|) < parameter_tolerance`.
    /// Set to `0.0` to disable.
    pub parameter_tolerance: f64,
    /// Stop after this many consecutive rejected steps.
    /// Set to `usize::MAX` to disable.
    pub max_consecutive_rejects: usize,
}

impl Default for OptParams {
    fn default() -> Self {
        Self {
            num_iterations: 100,
            initial_lm_damping: 10.0,
            parallelize: true,
            solver: Default::default(),
            skip_final_hessian: false,
            function_tolerance: 1e-10,
            gradient_tolerance: 1e-14,
            parameter_tolerance: 1e-12,
            max_consecutive_rejects: 20,
        }
    }
}
fn evaluate_cost_and_build_linear_system(
    variables: &VarFamilies,
    cost_system: &mut CostSystem,
    eq_system: &mut EqSystem,
    params: OptParams,
    pattern: Option<BlockSparseSymmetricMatrixPattern>,
    timing: bool,
) -> Result<(LinearSystem, Option<BlockSparseSymmetricMatrixPattern>), NllsError> {
    let eval_mode = EvalMode::CalculateDerivatives;

    let t0 = web_time::Instant::now();
    eq_system
        .eval(variables, eval_mode, params)
        .map_err(|e| NllsError::NllsEqConstraintSystemError { source: e })?;
    let t1 = web_time::Instant::now();
    cost_system
        .eval(variables, eval_mode, params)
        .map_err(|e| NllsError::NllsCostSystemError { source: e })?;
    let t2 = web_time::Instant::now();

    let (ls, next_pattern) = LinearSystem::from_families_costs_and_constraints(
        variables,
        &cost_system.evaluated_costs,
        cost_system.lm_damping,
        eq_system,
        params.solver,
        params.parallelize,
        pattern,
    )
    .map_err(|e| NllsError::NllsCostSystemError { source: e })?;
    let t3 = web_time::Instant::now();
    if timing {
        debug!(
            "  eq_eval={:.2}ms  cost_eval={:.2}ms  populate={:.2}ms",
            (t1 - t0).as_secs_f64() * 1000.0,
            (t2 - t1).as_secs_f64() * 1000.0,
            (t3 - t2).as_secs_f64() * 1000.0,
        );
    }
    Ok((ls, next_pattern))
}

/// Optimization solution
pub struct OptimizationSolution {
    /// The optimized variables
    pub variables: VarFamilies,
    /// The final cost
    pub final_cost: f64,
    /// The negated gradient vector.
    pub final_neg_gradient: BlockVector,
    /// the hessian matrix (use `precision_block` / `covariance_block` /
    /// `hessian_sparsity` for queries).
    final_damped_hessian: DampedHessian,
    /// Dense constraint Jacobian G (num_constraints × num_active_scalars).
    constraint_jacobian: Option<nalgebra::DMatrix<f64>>,
    /// Lazily computed covariance for block queries (lives in sophus_solver).
    covariance: Option<sophus_solver::covariance::Covariance>,
}

impl OptimizationSolution {
    /// Create a new solution (used internally by the optimizer).
    pub(crate) fn new(
        variables: VarFamilies,
        final_cost: f64,
        final_neg_gradient: BlockVector,
        final_damped_hessian: DampedHessian,
        constraint_jacobian: Option<nalgebra::DMatrix<f64>>,
    ) -> Self {
        Self {
            variables,
            final_cost,
            final_neg_gradient,
            final_damped_hessian,
            constraint_jacobian,
            covariance: None,
        }
    }

    /// Extract a precision (information) matrix block by family name and variable index.
    ///
    /// Returns the Hessian sub-matrix `H(i,j)` between variable `row_var_idx` in
    /// `row_family` and variable `col_var_idx` in `col_family`.
    ///
    /// **Note:** This returns the raw Hessian block and does *not* account for
    /// equality constraints. For the constraint-projected dual, use
    /// [`covariance_block`](Self::covariance_block).
    ///
    /// ```text
    /// solution.precision_block("poses", 1, "poses", 1)   // pose 1 self-information
    /// solution.precision_block("points", 0, "poses", 1)  // point 0 × pose 1 cross-block
    /// ```
    ///
    /// Returns `None` if a family doesn't exist or a variable is conditioned (fixed).
    pub fn precision_block(
        &self,
        row_family: &str,
        row_var_idx: usize,
        col_family: &str,
        col_var_idx: usize,
    ) -> Option<nalgebra::DMatrix<f64>> {
        use sophus_solver::matrix::PartitionBlockIndex;

        let row_partition = self.variables.partition_index(row_family)?;
        let row_block = self.variables.block_index(row_family, row_var_idx)?;
        let col_partition = self.variables.partition_index(col_family)?;
        let col_block = self.variables.block_index(col_family, col_var_idx)?;

        let row_idx = PartitionBlockIndex {
            partition: row_partition,
            block: row_block,
        };
        let col_idx = PartitionBlockIndex {
            partition: col_partition,
            block: col_block,
        };

        Some(self.final_damped_hessian.get_block(row_idx, col_idx))
    }

    /// Extract a covariance block by family name and variable index.
    ///
    /// Returns the covariance sub-matrix between variable `row_var_idx` in
    /// `row_family` and variable `col_var_idx` in `col_family`.
    ///
    /// Handles all cases uniformly:
    /// - **No constraints, full rank H**: returns `H⁻¹(i,j)`
    /// - **No constraints, rank-deficient H** (gauge): returns `H⁺(i,j)` (min-norm)
    /// - **With constraints**: returns `H⁺(i,j) - αᵢ · M⁺ · βⱼᵀ` (constraint projection)
    ///
    /// ```text
    /// solution.covariance_block("poses", 1, "poses", 1)   // pose 1 self-covariance
    /// solution.covariance_block("points", 0, "poses", 1)  // point 0 × pose 1 cross-covariance
    /// ```
    ///
    /// Returns `None` if a family doesn't exist or a variable is conditioned (fixed).
    pub fn covariance_block(
        &mut self,
        row_family: &str,
        row_var_idx: usize,
        col_family: &str,
        col_var_idx: usize,
    ) -> Option<nalgebra::DMatrix<f64>> {
        use sophus_solver::matrix::PartitionBlockIndex;

        let row_partition = self.variables.partition_index(row_family)?;
        let row_block = self.variables.block_index(row_family, row_var_idx)?;
        let col_partition = self.variables.partition_index(col_family)?;
        let col_block = self.variables.block_index(col_family, col_var_idx)?;

        let row_idx = PartitionBlockIndex {
            partition: row_partition,
            block: row_block,
        };
        let col_idx = PartitionBlockIndex {
            partition: col_partition,
            block: col_block,
        };

        // Lazily build the Covariance on first query.
        if self.covariance.is_none() {
            self.covariance = Some(
                self.final_damped_hessian
                    .clone()
                    .into_covariance(self.constraint_jacobian.as_ref()),
            );
        }

        Some(
            self.covariance
                .as_mut()
                .unwrap()
                .covariance_block(row_idx, col_idx),
        )
    }

    /// Return a view of the Hessian's block-sparsity pattern for visualization.
    pub fn hessian_sparsity(&self) -> HessianSparsity<'_> {
        HessianSparsity {
            matrix: &self.final_damped_hessian.matrix,
        }
    }
}

/// Block-sparsity pattern of the Hessian for visualization / analysis.
///
/// Provides partition layout and block-level sparsity queries without
/// exposing the underlying matrix values.
pub struct HessianSparsity<'a> {
    matrix: &'a sophus_solver::matrix::SymmetricMatrixEnum,
}

impl HessianSparsity<'_> {
    /// Partition specs (block_count, block_dim per partition).
    pub fn partition_specs(&self) -> &[sophus_solver::matrix::PartitionSpec] {
        use sophus_solver::matrix::IsSymmetricMatrix;
        self.matrix.partitions().specs()
    }

    /// Whether block `(row, col)` is structurally nonzero.
    pub fn has_block(
        &self,
        row_idx: sophus_solver::matrix::PartitionBlockIndex,
        col_idx: sophus_solver::matrix::PartitionBlockIndex,
    ) -> bool {
        use sophus_solver::matrix::IsSymmetricMatrix;
        self.matrix.has_block(row_idx, col_idx)
    }

    /// Schur complement fill-in pattern for the free-variable blocks.
    ///
    /// Returns global block index pairs `(gi, gj)` that are nonzero in the
    /// Schur complement `S_ff = H_ff - H_fm H_mm⁻¹ H_mf`.
    pub fn schur_fill_in(
        &self,
        num_free_partitions: usize,
    ) -> std::collections::HashSet<(usize, usize)> {
        damped_hessian::schur_fill_in(self.matrix, num_free_partitions)
    }
}

/// Linear solver error
#[derive(Snafu, Debug)]
pub enum NllsError {
    /// Linear solver error
    #[snafu(transparent)]
    LinearSolver {
        /// source
        source: LinearSolverError,
    },

    /// Quadratic cost system error
    #[snafu(transparent)]
    NllsCostSystemError {
        /// source
        source: CostError,
    },
    /// Eq constraint system error
    #[snafu(transparent)]
    NllsEqConstraintSystemError {
        /// source
        source: EqConstraintError,
    },
    /// Solver does not support equality constraints.
    #[snafu(display(
        "Solver {solver:?} does not support equality constraints. \
         Use FaerSparseLu, SchurBlockSparseLdlt, or SchurFaerSparseLdlt."
    ))]
    SolverDoesNotSupportEqConstraints {
        /// The solver that was requested.
        solver: LinearSolverEnum,
    },

    /// Schur solver with constraint on marginalized variable family.
    #[snafu(display(
        "Equality constraint touches marginalized family '{family_name}'. \
         Schur range-space KKT only supports constraints on free variables. \
         Use a non-Schur solver (e.g. FaerSparseLu) or make '{family_name}' free."
    ))]
    SchurConstraintOnMarginalizedFamily {
        /// The marginalized family name.
        family_name: String,
    },

    /// Schur solver requires at least one active free and one active marginalized variable.
    #[snafu(display(
        "Schur solver requires at least one active free and one active marginalized variable. \
         Got {num_free_families} free families ({num_free_scalars} active scalars), \
         {num_marg_families} marginalized families ({num_marg_scalars} active scalars)."
    ))]
    SchurMissingFreeOrMarg {
        /// Number of free families.
        num_free_families: usize,
        /// Number of active free scalars (excluding fixed members).
        num_free_scalars: usize,
        /// Number of marginalized families.
        num_marg_families: usize,
        /// Number of active marginalized scalars.
        num_marg_scalars: usize,
    },

    /// A cost term connects two marginalized families, breaking the block-diagonal
    /// H_mm assumption required by Schur complement.
    #[snafu(display(
        "Cost term connects marginalized families {family_a:?} and {family_b:?}. \
         Schur complement requires H_mm to be block-diagonal (each cost term may \
         touch at most one marginalized family). Make one of them free, or use a \
         non-Schur solver."
    ))]
    SchurCostConnectsTwoMargFamilies {
        /// First marginalized family.
        family_a: String,
        /// Second marginalized family.
        family_b: String,
    },

    /// Schur complement failed (H_mm is not positive definite)
    #[snafu(display("Schur complement failed: marginalized Hessian block is not PD"))]
    SchurComplementFailed,
    /// build_schur_covariance called on a system without marginalized variables
    /// or with equality constraints (Schur path not active for this system).
    #[snafu(display(
        "NotSchurSystem: build_schur_covariance requires nm>0 and no equality constraints"
    ))]
    NotSchurSystem,
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
    // Fast path: sum residuals directly without building EvaluatedCost structures.
    // This avoids the repeated rayon task-group overhead of the InnerLoop eval strategy.
    let mut c = 0.0;
    for cost_fn in cost_system.cost_fns.iter() {
        c += cost_fn
            .calc_total_cost(variables, params.parallelize)
            .map_err(|e| NllsError::NllsCostSystemError { source: e })?;
    }
    Ok(c)
}

/// Unified stateful optimizer.
///
/// This is the single optimizer struct that replaces the former `NllsOptimizer`.
///
/// # Construction
///
/// - [`Optimizer::new`] — no constraints (standard NLLS).
/// - [`Optimizer::new_with_eq`] — equality constraints only.
pub struct Optimizer {
    variables: VarFamilies,
    smooth_cost_system: CostSystem,
    eq_system: EqSystem,
    merit: f64,
    /// Smooth cost only (without barrier/dual terms).
    smooth_cost: f64,
    lm_damping: f64,
    consecutive_rejects: usize,
    params: OptParams,
    current_pattern: Option<BlockSparseSymmetricMatrixPattern>,
    /// Total steps taken.
    step_count: usize,
}
impl Optimizer {
    // ── Constructors ──────────────────────────────────────────────────────

    /// Create a new stateful LM optimizer (no constraints).
    pub fn new(
        variables: VarFamilies,
        cost_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsCostFn>>,
        params: OptParams,
    ) -> Result<Self, NllsError> {
        Self::new_with_eq(variables, cost_fns, vec![], params)
    }

    /// Create a new stateful LM optimizer with equality constraints.
    pub fn new_with_eq(
        variables: VarFamilies,
        cost_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsCostFn>>,
        eq_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsEqConstraintsFn>>,
        params: OptParams,
    ) -> Result<Self, NllsError> {
        validate_solver_config(&variables, &cost_fns, &eq_fns, params)?;

        let mut smooth_cost_system = CostSystem::new(&variables, cost_fns, params)
            .map_err(|e| NllsError::NllsCostSystemError { source: e })?;
        let mut eq_system = EqSystem::new(&variables, eq_fns, params)
            .map_err(|e| NllsError::NllsEqConstraintSystemError { source: e })?;
        let merit = calc_merit(&variables, &mut smooth_cost_system, &mut eq_system, params)?;

        // The pre-built pattern only records cost-function blocks, not G/G^T constraint
        // blocks, so it must not be used when equality constraints are present.
        let current_pattern = if eq_system.partitions.is_empty() {
            Some(LinearSystem::build_initial_pattern(
                &variables,
                &smooth_cost_system.cost_fns,
                &eq_system,
            ))
        } else {
            None
        };

        Ok(Self {
            variables,
            smooth_cost_system,
            eq_system,
            smooth_cost: merit,
            merit,
            lm_damping: params.initial_lm_damping,
            consecutive_rejects: 0,
            params,
            current_pattern,
            step_count: 0,
        })
    }

    // ── Step methods ──────────────────────────────────────────────────────

    /// Run one optimizer iteration.
    ///
    /// Returns [`StepInfo`] with current costs and termination status.
    pub fn step(&mut self) -> Result<StepInfo, NllsError> {
        let inner_result = self.step_nlls_inner()?;

        self.step_count += 1;

        let termination = if self.step_count >= self.params.num_iterations {
            Some(inner_result.unwrap_or(TerminationReason::MaxIterations))
        } else {
            inner_result
        };
        Ok(StepInfo {
            merit: self.merit,
            smooth_cost: self.smooth_cost,
            termination,
        })
    }

    // ── Shared step helpers ────────────────────────────────────────────────

    /// Handle a linear solver failure: increase damping, track rejects, possibly terminate.
    fn handle_solve_failure(&mut self) -> Option<TerminationReason> {
        self.lm_damping *= 2.0;
        self.consecutive_rejects += 1;
        if self.consecutive_rejects >= self.params.max_consecutive_rejects {
            Some(TerminationReason::MaxConsecutiveRejects {
                count: self.consecutive_rejects,
            })
        } else {
            None
        }
    }

    /// Check function tolerance: returns `Some(FunctionTolerance{..})` if converged.
    fn check_function_tolerance(&self, new_merit: f64) -> Option<TerminationReason> {
        let rel = (self.merit - new_merit) / (self.merit.abs() + f64::MIN_POSITIVE);
        if rel < self.params.function_tolerance {
            Some(TerminationReason::FunctionTolerance {
                rel_improvement: rel,
                threshold: self.params.function_tolerance,
            })
        } else {
            None
        }
    }

    /// Check gradient tolerance: returns `Some(GradientTolerance{..})` if the max absolute
    /// component of the gradient is below threshold.
    fn check_gradient_tolerance(&self, linear_system: &LinearSystem) -> Option<TerminationReason> {
        let max_grad = linear_system
            .neg_gradient
            .scalar_vector()
            .iter()
            .map(|g| g.abs())
            .fold(0.0_f64, f64::max);
        if max_grad < self.params.gradient_tolerance {
            Some(TerminationReason::GradientTolerance {
                gradient_max: max_grad,
                threshold: self.params.gradient_tolerance,
            })
        } else {
            None
        }
    }

    /// Solve the linear system, handling solver failures uniformly.
    fn try_solve(
        &mut self,
        linear_system: &mut LinearSystem,
    ) -> Result<Option<nalgebra::DVector<f64>>, NllsError> {
        match linear_system.solve() {
            Ok(d) => Ok(Some(d)),
            Err(NllsError::LinearSolver { .. }) | Err(NllsError::SchurComplementFailed) => Ok(None),
            Err(e) => Err(e),
        }
    }
    /// NLLS / equality-constrained inner step (no inequality constraints).
    fn step_nlls_inner(&mut self) -> Result<Option<TerminationReason>, NllsError> {
        debug!("lm-damping: {}", self.lm_damping);
        self.smooth_cost_system.lm_damping = self.lm_damping;
        let (mut linear_system, next_pattern) = evaluate_cost_and_build_linear_system(
            &self.variables,
            &mut self.smooth_cost_system,
            &mut self.eq_system,
            self.params,
            self.current_pattern.take(),
            false,
        )?;
        self.current_pattern = next_pattern;

        // Check gradient tolerance before solving.
        if let Some(reason) = self.check_gradient_tolerance(&linear_system) {
            return Ok(Some(reason));
        }

        let delta = match self.try_solve(&mut linear_system)? {
            Some(d) => d,
            None => return Ok(self.handle_solve_failure()),
        };

        let updated_families = self.variables.update(&delta);
        let updated_lambdas = self.eq_system.update_lambdas(&self.variables, &delta);
        let new_merit = calc_merit(
            &updated_families,
            &mut self.smooth_cost_system,
            &mut self.eq_system,
            self.params,
        )?;

        // Check parameter tolerance (max component of tangent-space step).
        let max_step = delta.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
        if max_step < self.params.parameter_tolerance {
            if new_merit < self.merit {
                self.variables = updated_families;
                self.eq_system.lambda = updated_lambdas;
                self.smooth_cost = new_merit;
                self.merit = new_merit;
            }
            return Ok(Some(TerminationReason::ParameterTolerance {
                max_step,
                threshold: self.params.parameter_tolerance,
            }));
        }

        // Gain ratio ρ = (actual decrease) / (predicted decrease by the linear model).
        //   predicted = δᵀ(ν·δ + g), where g = -neg_gradient
        // This measures how well the quadratic model approximates the true cost.
        let neg_grad = linear_system.neg_gradient.scalar_vector();
        let predicted_decrease = self.lm_damping * delta.dot(&delta) + delta.dot(neg_grad);

        let actual_decrease = self.merit - new_merit;
        let gain_ratio = if predicted_decrease.abs() > f64::MIN_POSITIVE {
            actual_decrease / predicted_decrease
        } else {
            if actual_decrease > 0.0 { 1.0 } else { 0.0 }
        };

        let reason;
        if gain_ratio > 0.0 && new_merit < self.merit {
            // Accepted step — decrease damping using Nielsen's formula.
            // ν *= max(1/3, 1 - (2ρ - 1)³)
            let rho_factor = 1.0 - (2.0 * gain_ratio - 1.0).powi(3);
            self.lm_damping *= rho_factor.max(1.0 / 3.0);

            reason = self.check_function_tolerance(new_merit);
            self.consecutive_rejects = 0;
            self.variables = updated_families;
            self.eq_system.lambda = updated_lambdas;
            self.smooth_cost = new_merit;
            self.merit = new_merit;
        } else {
            // Rejected step — increase damping.
            self.lm_damping *= 2.0;
            self.consecutive_rejects += 1;
            reason = if self.consecutive_rejects >= self.params.max_consecutive_rejects {
                Some(TerminationReason::MaxConsecutiveRejects {
                    count: self.consecutive_rejects,
                })
            } else {
                None
            };
        }

        info!(
            "lm-damping: {:?}, merit {:?}, (new_merit {:?})",
            self.lm_damping, self.merit, new_merit
        );
        Ok(reason)
    }

    // ── Common accessors ──────────────────────────────────────────────────

    /// Current variable families.
    pub fn variables(&self) -> &VarFamilies {
        &self.variables
    }

    /// Consume the optimizer and return the current variables.
    pub fn into_variables(self) -> VarFamilies {
        self.variables
    }

    /// Evaluate cost at arbitrary variables (does not modify optimizer state).
    pub fn eval_cost_at(&mut self, variables: &VarFamilies) -> Result<f64, NllsError> {
        calc_merit(
            variables,
            &mut self.smooth_cost_system,
            &mut self.eq_system,
            self.params,
        )
    }

    /// Current merit value.
    pub fn merit(&self) -> f64 {
        self.merit
    }

    /// Current smooth cost only.
    pub fn smooth_cost(&self) -> f64 {
        self.smooth_cost
    }

    /// Current LM damping parameter.
    pub fn lm_damping(&self) -> f64 {
        self.lm_damping
    }
}
/// Validate solver configuration before optimization.
///
/// Checks:
/// - Solver supports equality constraints (if present).
/// - Schur solver has at least one free and one marginalized family.
/// - Schur constraints only touch free families (not marginalized).
/// - No cost term connects two marginalized families (Schur block-diagonal assumption).
fn validate_solver_config(
    variables: &VarFamilies,
    cost_fns: &[alloc::boxed::Box<dyn IsCostFn>],
    eq_fns: &[alloc::boxed::Box<dyn IsEqConstraintsFn>],
    params: OptParams,
) -> Result<(), NllsError> {
    let has_constraints = eq_fns.iter().any(|f| f.residual_dim() > 0);
    if has_constraints {
        if !params.solver.supports_eq_constraints() {
            return Err(NllsError::SolverDoesNotSupportEqConstraints {
                solver: params.solver,
            });
        }

        // Schur solvers use range-space KKT which only works when constraints
        // touch free variables, not marginalized ones.
        if params.solver.is_schur() {
            for eq_fn in eq_fns {
                for name in eq_fn.constraint_family_names() {
                    if let Some(family) = variables.collection.get(&name)
                        && family.get_var_kind() == VarKind::Marginalized
                    {
                        return Err(NllsError::SchurConstraintOnMarginalizedFamily {
                            family_name: name,
                        });
                    }
                }
            }
        }
    }

    if params.solver.is_schur() {
        let num_free_families = variables.num_of_kind(VarKind::Free);
        let num_marg_families = variables.num_of_kind(VarKind::Marginalized);
        let num_free_scalars = variables.num_free_scalars();
        let num_marg_scalars = variables.num_marg_scalars();
        if num_free_scalars == 0 || num_marg_scalars == 0 {
            return Err(NllsError::SchurMissingFreeOrMarg {
                num_free_families,
                num_free_scalars,
                num_marg_families,
                num_marg_scalars,
            });
        }

        // Check no cost term connects two marginalized families.
        // Schur requires H_mm to be block-diagonal.
        for cost_fn in cost_fns {
            let names = cost_fn.cost_family_names();
            let marg_names: Vec<&String> = names
                .iter()
                .filter(|n| {
                    variables
                        .collection
                        .get(n.as_str())
                        .is_some_and(|f| f.get_var_kind() == VarKind::Marginalized)
                })
                .collect();
            if marg_names.len() > 1 {
                return Err(NllsError::SchurCostConnectsTwoMargFamilies {
                    family_a: marg_names[0].clone(),
                    family_b: marg_names[1].clone(),
                });
            }
        }
    }

    Ok(())
}

/// Non-linear least squares optimization with equality constraints
pub fn optimize_nlls_with_eq_constraints(
    mut variables: VarFamilies,
    cost_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsCostFn>>,
    eq_constraints_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsEqConstraintsFn>>,
    params: OptParams,
) -> Result<OptimizationSolution, NllsError> {
    validate_solver_config(&variables, &cost_fns, &eq_constraints_fns, params)?;

    let mut cost_system = CostSystem::new(&variables, cost_fns, params)
        .map_err(|e| NllsError::NllsCostSystemError { source: e })?;
    let mut eq_system = EqSystem::new(&variables, eq_constraints_fns, params)
        .map_err(|e| NllsError::NllsEqConstraintSystemError { source: e })?;

    let mut merit: f64 = calc_merit(&variables, &mut cost_system, &mut eq_system, params)?;
    let initial_merit = merit;

    let mut last_linear_system: Option<LinearSystem> = None;
    let has_eq_constraints = !eq_system.partitions.is_empty();
    let mut current_pattern: Option<BlockSparseSymmetricMatrixPattern> = if !has_eq_constraints {
        Some(LinearSystem::build_initial_pattern(
            &variables,
            &cost_system.cost_fns,
            &eq_system,
        ))
    } else {
        None
    };
    for i in 0..params.num_iterations {
        debug!("lm-damping: {}", cost_system.lm_damping);
        let do_timing = i == params.num_iterations - 1;
        let t_iter = web_time::Instant::now();
        let (mut linear_system, next_pattern) = evaluate_cost_and_build_linear_system(
            &variables,
            &mut cost_system,
            &mut eq_system,
            params,
            current_pattern.take(),
            do_timing,
        )?;
        current_pattern = next_pattern;

        let t_solve = web_time::Instant::now();
        let delta = match linear_system.solve() {
            Ok(d) => d,
            Err(NllsError::LinearSolver { .. }) | Err(NllsError::SchurComplementFailed) => {
                cost_system.lm_damping *= 2.0;
                continue;
            }
            Err(e) => return Err(e),
        };
        let t_after_solve = web_time::Instant::now();

        if do_timing {
            debug!(
                "  solve={:.2}ms  iter_total={:.2}ms",
                (t_after_solve - t_solve).as_secs_f64() * 1000.0,
                t_iter.elapsed().as_secs_f64() * 1000.0,
            );
        }
        let updated_families = variables.update(&delta);
        let updated_lambdas = eq_system.update_lambdas(&variables, &delta);

        let new_merit = calc_merit(&updated_families, &mut cost_system, &mut eq_system, params)?;

        // Gain ratio ρ = (actual decrease) / (predicted decrease).
        let neg_grad = linear_system.neg_gradient.scalar_vector();
        let predicted_decrease = cost_system.lm_damping * delta.dot(&delta) + delta.dot(neg_grad);
        let actual_decrease = merit - new_merit;
        let gain_ratio = if predicted_decrease.abs() > f64::MIN_POSITIVE {
            actual_decrease / predicted_decrease
        } else {
            if actual_decrease > 0.0 { 1.0 } else { 0.0 }
        };

        if gain_ratio > 0.0 && new_merit < merit {
            let rho_factor = 1.0 - (2.0 * gain_ratio - 1.0).powi(3);
            cost_system.lm_damping *= rho_factor.max(1.0 / 3.0);
            variables = updated_families;
            eq_system.lambda = updated_lambdas;
            merit = new_merit;
            last_linear_system = Some(linear_system);
        } else {
            cost_system.lm_damping *= 2.0;
        }

        info!(
            "i: {:?}, lm-damping: {:?}, merit {:?}, (new_merit {:?})",
            i, cost_system.lm_damping, merit, new_merit
        );
    }
    info!("e^2: {initial_merit:?} -> {merit:?}");

    let final_linear_system = if params.skip_final_hessian {
        if let Some(ls) = last_linear_system {
            ls
        } else {
            evaluate_cost_and_build_linear_system(
                &variables,
                &mut cost_system,
                &mut eq_system,
                params,
                current_pattern.take(),
                false,
            )?
            .0
        }
    } else {
        cost_system.lm_damping = 0.0;
        evaluate_cost_and_build_linear_system(
            &variables,
            &mut cost_system,
            &mut eq_system,
            params,
            current_pattern.take(),
            false,
        )?
        .0
    };

    let (final_neg_gradient, final_damped_hessian) =
        final_linear_system.into_gradient_and_hessian();
    let constraint_jacobian = eq_system.dense_constraint_jacobian(&variables);
    Ok(OptimizationSolution::new(
        variables,
        merit,
        final_neg_gradient,
        final_damped_hessian,
        constraint_jacobian,
    ))
}
