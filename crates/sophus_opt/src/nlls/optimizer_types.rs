extern crate alloc;

use snafu::Snafu;
use sophus_solver::{
    LinearSolverEnum,
    LinearSolverError,
    matrix::block::BlockVector,
};

use super::damped_hessian;
pub(crate) use super::damped_hessian::DampedHessian;
use crate::{
    nlls::{
        constraint::eq_constraint_fn::{
            EqConstraintError,
            IsEqConstraintsFn,
        },
        cost::{
            cost_fn::{
                CostError,
                IsCostFn,
            },
            ineq_barrier_fn::IsIneqBarrierFn,
        },
    },
    variables::VarFamilies,
};

/// Inequality constraint method selector.
///
/// Used by `Optimizer::new` when `ineq_constraints` is non-empty
/// to choose between primal-dual IPM and SQP strategies.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum IneqMethod {
    /// Primal-dual IPM (Nocedal & Wright Ch. 19).
    Ipm {
        /// Fraction-to-boundary safety factor.
        tau: f64,
        /// Inner steps per outer iteration.
        inner_iters: usize,
        /// Multiplicative decay applied to lambdas between outer iterations.
        lambda_decay: f64,
    },
    /// SQP with frozen Jacobians (Nocedal & Wright Ch. 18).
    Sqp {
        /// Fraction-to-boundary safety factor.
        tau: f64,
        /// Inner steps per outer iteration.
        inner_iters: usize,
        /// Multiplicative decay applied to mu between outer iterations.
        mu_decay: f64,
    },
}

impl Default for IneqMethod {
    fn default() -> Self {
        IneqMethod::Ipm {
            tau: 0.99,
            inner_iters: 50,
            lambda_decay: 0.5,
        }
    }
}

/// Information returned by each `Optimizer::step()` call.
#[derive(Clone, Debug)]
pub struct StepInfo {
    /// Current merit value (smooth cost + barrier terms).
    pub merit: f64,
    /// Smooth cost only (without barrier terms).
    pub smooth_cost: f64,
    /// Whether an outer transition happened on this step
    /// (lambda/mu decay, damping reset, Jacobian refreeze for SQP).
    pub did_outer_step: bool,
    /// Current inner step index within the current outer iteration (0-based).
    pub inner_step: usize,
    /// Current LM damping parameter ν.
    pub lm_damping: f64,
    /// Whether the optimizer is in phase-1 (feasibility search).
    pub in_phase1: bool,
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
    /// Inequality constraint method. Only used when `ineq_constraints` is non-empty.
    pub ineq_method: IneqMethod,
}

impl OptParams {
    /// Defaults for IPM (interior point method) with inequality constraints.
    ///
    /// Use with the spread operator: `OptParams { num_iterations: 200, ..OptParams::ipm() }`.
    pub fn ipm() -> Self {
        Self {
            num_iterations: 5000,
            initial_lm_damping: 1.0,
            ineq_method: IneqMethod::Ipm {
                tau: 0.99,
                inner_iters: 50,
                lambda_decay: 0.5,
            },
            ..Default::default()
        }
    }

    /// Defaults for SQP (sequential quadratic programming) with inequality constraints.
    ///
    /// Use with the spread operator: `OptParams { num_iterations: 200, ..OptParams::sqp() }`.
    pub fn sqp() -> Self {
        Self {
            num_iterations: 5000,
            initial_lm_damping: 1.0,
            ineq_method: IneqMethod::Sqp {
                tau: 0.99,
                inner_iters: 50,
                mu_decay: 0.5,
            },
            ..Default::default()
        }
    }
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
            ineq_method: IneqMethod::default(),
        }
    }
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
    pub(crate) final_damped_hessian: DampedHessian,
    /// Dense constraint Jacobian G (num_constraints × num_active_scalars).
    pub(crate) constraint_jacobian: Option<nalgebra::DMatrix<f64>>,
    /// Lazily computed covariance for block queries (lives in sophus_solver).
    pub(crate) covariance: Option<sophus_solver::covariance::Covariance>,
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
    pub(crate) matrix: &'a sophus_solver::matrix::SymmetricMatrixEnum,
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

/// Problem definition for the optimizer: costs, equality constraints, and inequality barriers.
pub struct OptProblem {
    /// Cost functions (least-squares residuals).
    pub costs: alloc::vec::Vec<alloc::boxed::Box<dyn IsCostFn>>,
    /// Equality constraint functions (c(x) = 0).
    pub eq_constraints: alloc::vec::Vec<alloc::boxed::Box<dyn IsEqConstraintsFn>>,
    /// Inequality constraint barriers (h(x) >= 0, as log-barrier or dual).
    pub ineq_constraints: alloc::vec::Vec<alloc::boxed::Box<dyn IsIneqBarrierFn>>,
}

impl OptProblem {
    /// Create a problem with only costs (no constraints).
    pub fn costs_only(costs: alloc::vec::Vec<alloc::boxed::Box<dyn IsCostFn>>) -> Self {
        Self {
            costs,
            eq_constraints: vec![],
            ineq_constraints: vec![],
        }
    }
}
