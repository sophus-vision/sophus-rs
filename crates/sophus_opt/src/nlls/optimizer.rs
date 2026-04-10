extern crate alloc;

use log::info;
use sophus_solver::matrix::block_sparse::BlockSparseSymmetricMatrixPattern;

use super::{
    optimizer_helpers as helpers,
    optimizer_step as step_fns,
    *,
};

/// Result of a successful fraction-to-boundary line search.
pub(crate) struct LineSearchResult {
    /// Proposed variables after the line search step.
    pub(crate) proposed: VarFamilies,
    /// Constraint values h_j(x_proposed) for each barrier, per constraint.
    pub(crate) h_values: alloc::vec::Vec<alloc::vec::Vec<f64>>,
    /// Smooth cost at the proposed point (without barrier terms).
    pub(crate) smooth_cost: f64,
    /// Full merit (smooth + barrier) at the proposed point.
    pub(crate) merit: f64,
}

/// Stateful optimizer supporting NLLS, IPM, and SQP strategies.
///
/// The inequality constraint handling is determined by the [`IneqStrategy`]
/// stored at construction time.
///
/// # Construction
///
/// Use [`Optimizer::new`] for all cases, passing an [`OptProblem`] that bundles the costs
/// and constraints:
/// - No constraints: `Optimizer::new(vars, OptProblem::costs_only(costs), params)`
/// - Equality constraints only: `Optimizer::new(vars, OptProblem { costs, eq_constraints: eq_fns,
///   ineq_constraints: vec![] }, params)`
/// - Inequality constraints only: `Optimizer::new(vars, OptProblem { costs, eq_constraints: vec![],
///   ineq_constraints: barriers }, params)`
/// - Both: `Optimizer::new(vars, OptProblem { costs, eq_constraints: eq_fns, ineq_constraints:
///   barriers }, params)`
pub struct Optimizer {
    pub(crate) variables: VarFamilies,
    pub(crate) smooth_cost_system: CostSystem,
    pub(crate) eq_system: EqSystem,
    pub(crate) merit: f64,
    /// Smooth cost only (without barrier/dual terms).
    pub(crate) smooth_cost: f64,
    pub(crate) lm_damping: f64,
    pub(crate) consecutive_rejects: usize,
    pub(crate) params: OptParams,
    pub(crate) current_pattern: Option<BlockSparseSymmetricMatrixPattern>,
    pub(crate) strategy: IneqStrategy,
    /// Steps since last outer transition (for IPM/SQP inner loop tracking).
    pub(crate) inner_step_count: usize,
    /// Count of consecutive outer cycles without meaningful improvement.
    pub(crate) stalled_outer_count: usize,
    /// Total steps taken across all outer iterations.
    pub(crate) total_step_count: usize,
}

impl Optimizer {
    /// Create a new stateful LM optimizer.
    ///
    /// This is the single unified constructor. Bundle costs and constraints into an
    /// [`OptProblem`]:
    /// - Plain NLLS: `OptProblem::costs_only(costs)`
    /// - With equality constraints only: set `eq_constraints`
    /// - With inequality constraints only: set `ineq_constraints`, configure `params.ineq_method`
    /// - With both: set both `eq_constraints` and `ineq_constraints`
    ///
    /// The inequality method (IPM or SQP) is selected via `params.ineq_method`.
    pub fn new(
        variables: VarFamilies,
        problem: OptProblem,
        params: OptParams,
    ) -> Result<Self, NllsError> {
        let OptProblem {
            costs: cost_fns,
            eq_constraints: eq_fns,
            ineq_constraints: barriers,
        } = problem;
        validate_solver_config(&variables, &cost_fns, &eq_fns, params)?;
        if barriers.is_empty() {
            // Plain NLLS or equality-constrained NLLS path.
            let mut smooth_cost_system = CostSystem::new(&variables, cost_fns, params)
                .map_err(|e| NllsError::NllsCostSystemError { source: e })?;
            let eq_system = EqSystem::new(&variables, eq_fns, params)
                .map_err(|e| NllsError::NllsEqConstraintSystemError { source: e })?;
            let merit = helpers::calc_merit(&variables, &mut smooth_cost_system, params)?;

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
                strategy: IneqStrategy::None,
                inner_step_count: 0,
                stalled_outer_count: 0,
                total_step_count: 0,
            })
        } else {
            Self::new_ineq_internal(variables, cost_fns, eq_fns, barriers, params)
        }
    }

    /// Internal: Create a new inequality-constrained optimizer (IPM or SQP).
    ///
    /// Shared setup for both strategies: sort barriers, check feasibility,
    /// build cost/eq systems, compute initial merit, then dispatch to
    /// strategy-specific initialization.
    pub(crate) fn new_ineq_internal(
        variables: VarFamilies,
        smooth_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsCostFn>>,
        eq_fns: alloc::vec::Vec<alloc::boxed::Box<dyn IsEqConstraintsFn>>,
        mut barriers: alloc::vec::Vec<alloc::boxed::Box<dyn IsIneqBarrierFn>>,
        params: OptParams,
    ) -> Result<Self, NllsError> {
        // Sort barriers (computes reduction ranges, clears lambdas).
        for b in barriers.iter_mut() {
            b.sort(&variables);
        }

        // Compute initial smooth cost for adaptive barrier scaling.
        let initial_smooth_cost = {
            let mut sc = 0.0_f64;
            for cf in smooth_fns.iter() {
                sc += cf
                    .calc_total_cost(&variables, params.parallelize)
                    .map_err(|e| NllsError::NllsCostSystemError { source: e })?;
            }
            sc
        };

        // Count total inequality constraints for adaptive scaling.
        let num_ineq: usize = barriers
            .iter()
            .map(|b| b.eval_h_values(&variables).len())
            .sum();

        // Check feasibility.
        let any_infeasible = barriers
            .iter()
            .any(|b| b.eval_h_values(&variables).iter().any(|&h| h < 0.0));

        // Adaptive barrier parameter: μ = σ · f₀ / m (IPOPT-style).
        let sigma = 0.1_f64;
        let mu = if num_ineq > 0 && initial_smooth_cost > 1e-12 {
            (sigma * initial_smooth_cost / num_ineq as f64).max(1e-8)
        } else {
            1e-8
        };

        let tau = match params.ineq_method {
            IneqMethod::Ipm { tau, .. } | IneqMethod::Sqp { tau, .. } => tau,
        };

        // Strategy-specific initialization.
        let strategy = if any_infeasible {
            // ── Elastic feasibility (Wächter & Biegler §3.3) ────────────────
            //
            // Add elastic slacks v_j ≥ 0 so s_j = h_j + v_j > 0 always.
            // The smooth cost is included from the start (not deferred).
            // Fixed μ/ρ — no outer transitions during elastic phase.
            // Transitions to IPM/SQP when all h_j > 0.
            let max_violation: f64 = barriers
                .iter()
                .flat_map(|b| b.eval_h_values(&variables))
                .filter(|&h| h < 0.0)
                .map(|h| h.abs())
                .fold(0.0_f64, f64::max);

            // μ must be large enough for the barrier gradient (λ = μ/s) to compete
            // with the smooth cost gradient pulling away from feasibility.
            let elastic_mu = (initial_smooth_cost / num_ineq.max(1) as f64)
                .max(mu)
                .max(max_violation)
                .max(1e-4);

            // ρ: ℓ₁ penalty weight on elastic slacks. For the penalty to be exact
            // (driving v→0 at optimality), ρ must exceed the optimal Lagrange
            // multipliers. Scale proportionally to the cost-per-violation ratio.
            let rho = (1000.0 * initial_smooth_cost
                / (num_ineq.max(1) as f64 * max_violation.max(1e-6)))
            .max(100.0 * max_violation)
            .max(100.0);

            info!(
                "Elastic: infeasible start, μ={elastic_mu:.4e}, ρ={rho:.4e}, \
                 max_violation={max_violation:.4e}, f₀={initial_smooth_cost:.4e}, m={num_ineq}"
            );

            let all_slacks =
                helpers::compute_all_elastic_slacks(&barriers, &variables, elastic_mu, rho);
            helpers::set_elastic_state(&barriers, &all_slacks, &variables, elastic_mu);

            IneqStrategy::Elastic(ElasticState {
                barriers,
                mu: elastic_mu,
                rho,
                tau,
            })
        } else {
            // ── Standard IPM or SQP (feasible start) ───────────────────────
            match params.ineq_method {
                IneqMethod::Ipm { tau, .. } => {
                    info!(
                        "IPM: adaptive μ₀ = {mu:.4e} (σ={sigma}, f₀={initial_smooth_cost:.4e}, m={num_ineq})"
                    );
                    for b in barriers.iter() {
                        let h_values = b.eval_h_values(&variables);
                        let lambdas: alloc::vec::Vec<f64> = h_values
                            .iter()
                            .map(|&h| if h > 1e-6 { mu / h } else { mu })
                            .collect();
                        b.set_lambdas(lambdas);
                    }
                    // Initialize filter with theta_max based on initial infeasibility.
                    let theta_0 = helpers::compute_theta(&barriers, &variables);
                    let filter = super::filter::Filter::new(1e4 * theta_0.max(1.0));
                    IneqStrategy::Ipm(IpmState {
                        barriers,
                        mu,
                        tau,
                        filter,
                    })
                }
                IneqMethod::Sqp { tau, .. } => {
                    info!(
                        "SQP: adaptive μ₀ = {mu:.4e} (σ={sigma}, f₀={initial_smooth_cost:.4e}, m={num_ineq})"
                    );
                    let frozen_lins = barriers
                        .iter()
                        .map(|b| b.eval_constraint_linearizations(&variables))
                        .collect();
                    let slacks: alloc::vec::Vec<alloc::vec::Vec<f64>> = barriers
                        .iter()
                        .map(|b| b.eval_h_values(&variables))
                        .collect();
                    let duals = slacks
                        .iter()
                        .map(|sv| sv.iter().map(|&s| mu / s.max(1e-6)).collect())
                        .collect();
                    IneqStrategy::Sqp(SqpState {
                        barriers,
                        frozen_lins,
                        duals,
                        slacks,
                        mu,
                        tau,
                    })
                }
            }
        };

        // ── Shared: build systems, compute merit, build pattern ────────────
        let smooth_cost_system = CostSystem::new(&variables, smooth_fns, params)
            .map_err(|e| NllsError::NllsCostSystemError { source: e })?;
        let eq_system = EqSystem::new(&variables, eq_fns, params)
            .map_err(|e| NllsError::NllsEqConstraintSystemError { source: e })?;

        let barriers_ref = strategy.barriers();
        let (smooth_cost, merit) = match &strategy {
            IneqStrategy::Ipm(_) => helpers::calc_ipm_merit(
                &variables,
                &smooth_cost_system,
                barriers_ref,
                &eq_system,
                params.parallelize,
            )
            .map_err(|e| NllsError::NllsCostSystemError { source: e })?,
            IneqStrategy::Sqp(sqp) => helpers::calc_sqp_merit(
                &variables,
                &smooth_cost_system,
                barriers_ref,
                &sqp.duals,
                &eq_system,
                params.parallelize,
            )
            .map_err(|e| NllsError::NllsCostSystemError { source: e })?,
            IneqStrategy::Elastic(e) => {
                let all_slacks =
                    helpers::compute_all_elastic_slacks(&e.barriers, &variables, e.mu, e.rho);
                helpers::calc_elastic_merit(
                    &variables,
                    &smooth_cost_system,
                    &e.barriers,
                    &all_slacks,
                    e.mu,
                    e.rho,
                    params.parallelize,
                )
                .map_err(|e| NllsError::NllsCostSystemError { source: e })?
            }
            _ => unreachable!(),
        };

        let current_pattern = helpers::maybe_build_ineq_pattern(
            &variables,
            &smooth_cost_system,
            barriers_ref,
            &eq_system,
        );

        Ok(Self {
            variables,
            smooth_cost_system,
            eq_system,
            smooth_cost,
            merit,
            lm_damping: params.initial_lm_damping,
            consecutive_rejects: 0,
            params,
            current_pattern,
            strategy,
            inner_step_count: 0,
            stalled_outer_count: 0,
            total_step_count: 0,
        })
    }

    /// Run one optimizer iteration, preserving damping and cached structures across calls.
    ///
    /// For IPM/SQP strategies, this method automatically manages inner/outer loop
    /// transitions: when the inner loop converges or reaches its step budget,
    /// it performs the outer transition (decay lambdas/mu, reset damping, refreeze
    /// Jacobians for SQP) and resets the inner counter.
    ///
    /// Returns [`StepInfo`] with current costs, inner step index, and termination status.
    pub fn step(&mut self) -> Result<StepInfo, NllsError> {
        // Elastic feasibility: run elastic step, check for transition to IPM/SQP.
        if matches!(self.strategy, IneqStrategy::Elastic(_)) {
            return step_fns::step_elastic(self);
        }

        let inner_result = match &self.strategy {
            IneqStrategy::None => step_fns::step_nlls_inner(self),
            IneqStrategy::Elastic(_) => unreachable!(),
            IneqStrategy::Ipm(_) => step_fns::step_ipm_inner(self),
            IneqStrategy::Sqp(_) => step_fns::step_sqp_inner(self),
        }?;

        self.total_step_count += 1;
        self.inner_step_count += 1;

        // For plain NLLS, no outer loop management.
        if matches!(self.strategy, IneqStrategy::None) {
            let termination = if self.total_step_count >= self.params.num_iterations {
                Some(inner_result.unwrap_or(TerminationReason::MaxIterations))
            } else {
                inner_result
            };
            return Ok(StepInfo {
                merit: self.merit,
                smooth_cost: self.smooth_cost,
                did_outer_step: false,
                inner_step: self.inner_step_count - 1,
                lm_damping: self.lm_damping,
                in_phase1: false,
                termination,
            });
        }

        // IPM/SQP: check if outer transition is needed.
        let (inner_iters, decay) = match self.params.ineq_method {
            IneqMethod::Ipm {
                inner_iters,
                lambda_decay,
                ..
            } => (inner_iters, lambda_decay),
            IneqMethod::Sqp {
                inner_iters,
                mu_decay,
                ..
            } => (inner_iters, mu_decay),
        };

        let inner_converged = matches!(
            &inner_result,
            Some(TerminationReason::FunctionTolerance { .. })
                | Some(TerminationReason::ParameterTolerance { .. })
                | Some(TerminationReason::GradientTolerance { .. })
        );

        let inner_stalled = matches!(
            &inner_result,
            Some(TerminationReason::MaxConsecutiveRejects { .. })
        );
        let needs_outer = inner_converged || inner_stalled || self.inner_step_count >= inner_iters;

        let mut did_outer_step = false;
        let mut termination = None;
        if needs_outer {
            let smooth_before = self.smooth_cost;
            step_fns::do_outer_transition(self, decay, inner_converged)?;
            did_outer_step = true;

            // Track whether the outer cycle made progress on the actual cost.
            let rel =
                (smooth_before - self.smooth_cost) / (smooth_before.abs() + f64::MIN_POSITIVE);
            if rel < self.params.function_tolerance {
                self.stalled_outer_count += 1;
            } else {
                self.stalled_outer_count = 0;
            }

            // When the outer loop stalls (no smooth cost progress), the iterate
            // is at the current barrier subproblem optimum. Force a μ decrease
            // to continue toward the true constrained optimum.
            // After 3 consecutive stalled cycles, force the decrease.
            if self.stalled_outer_count >= 3
                && self.stalled_outer_count % 3 == 0
                && !inner_converged
            {
                step_fns::do_outer_transition(self, decay, true)?; // force decrease
            }

            const MAX_STALLED_OUTER_CYCLES: usize = 20;
            if self.stalled_outer_count >= MAX_STALLED_OUTER_CYCLES {
                termination = Some(TerminationReason::FunctionTolerance {
                    rel_improvement: rel,
                    threshold: self.params.function_tolerance,
                });
            }
        }

        // Check total step budget.
        if self.total_step_count >= self.params.num_iterations {
            termination = Some(TerminationReason::MaxIterations);
        }

        Ok(StepInfo {
            merit: self.merit,
            smooth_cost: self.smooth_cost,
            did_outer_step,
            inner_step: if did_outer_step {
                0
            } else {
                self.inner_step_count - 1
            },
            lm_damping: self.lm_damping,
            in_phase1: false,
            termination,
        })
    }

    /// Accept a step: decrease LM damping using Nielsen's formula, reset reject counter.
    pub(crate) fn accept_step(&mut self, gain_ratio: f64) {
        let rho_factor = 1.0 - (2.0 * gain_ratio - 1.0).powi(3);
        self.lm_damping *= rho_factor.max(1.0 / 3.0);
        self.consecutive_rejects = 0;
    }

    /// Reject a step: increase LM damping, track consecutive rejects, possibly terminate.
    pub(crate) fn reject_step(&mut self) -> Option<TerminationReason> {
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

    /// Handle a linear solver failure: increase damping, track rejects, possibly terminate.
    pub(crate) fn handle_solve_failure(&mut self) -> Option<TerminationReason> {
        self.reject_step()
    }

    /// Check function tolerance: returns `Some(FunctionTolerance{..})` if converged.
    pub(crate) fn check_function_tolerance(&self, new_merit: f64) -> Option<TerminationReason> {
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
    pub(crate) fn check_gradient_tolerance(
        &self,
        linear_system: &LinearSystem,
    ) -> Option<TerminationReason> {
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

    /// Check gradient tolerance unless equality constraints are active.
    ///
    /// Skips the check when eq constraints are present, because the augmented KKT
    /// gradient includes Lagrange multiplier contributions that can be small even
    /// when the objective is not yet minimized.
    pub(crate) fn check_gradient_tolerance_unless_eq(
        &self,
        linear_system: &LinearSystem,
    ) -> Option<TerminationReason> {
        if !self.eq_system.partitions.is_empty() {
            return None;
        }
        self.check_gradient_tolerance(linear_system)
    }

    /// Project the current variables onto the equality constraint manifold.
    ///
    /// Evaluates the equality constraints and applies the first-order correction
    /// δx = -Gᵀ(GGᵀ)⁻¹c(x). No-ops when there are no equality constraints.
    pub(crate) fn project_onto_eq_manifold(&mut self) -> Result<(), NllsError> {
        if self.eq_system.partitions.is_empty() {
            return Ok(());
        }
        self.eq_system
            .eval(&self.variables, EvalMode::CalculateDerivatives, self.params)
            .map_err(|e| NllsError::NllsEqConstraintSystemError { source: e })?;
        if let Some(correction) = self.eq_system.project_correction(&self.variables) {
            self.variables = self.variables.update(&correction);
        }
        Ok(())
    }

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
        helpers::calc_merit(variables, &mut self.smooth_cost_system, self.params)
    }

    /// Current merit value (smooth cost + barrier terms for IPM/SQP).
    pub fn merit(&self) -> f64 {
        self.merit
    }

    /// Current smooth cost only (without barrier/dual terms).
    ///
    /// For unconstrained problems, this equals `merit()`.
    /// For IPM/SQP, this is the actual objective value.
    pub fn smooth_cost(&self) -> f64 {
        self.smooth_cost
    }

    /// Current LM damping parameter.
    pub fn lm_damping(&self) -> f64 {
        self.lm_damping
    }

    /// Current barrier/dual parameter, or `None` for plain NLLS.
    ///
    /// - IPM: returns the mean lambda across all constraints.
    /// - SQP: returns mu.
    /// - NLLS: returns `None`.
    pub fn barrier_param(&self) -> Option<f64> {
        match &self.strategy {
            IneqStrategy::None => None,
            IneqStrategy::Elastic(e) => Some(e.mu),
            IneqStrategy::Ipm(ipm) => Some(ipm.mu),
            IneqStrategy::Sqp(sqp) => Some(sqp.mu),
        }
    }

    /// Build the final `OptimizationSolution` after the optimization loop has finished.
    ///
    /// Evaluates a final linear system (with zero damping unless `skip_final_hessian`
    /// is set) to obtain the Hessian and gradient for covariance queries.
    ///
    /// Only valid for unconstrained / equality-constrained NLLS (`IneqStrategy::None`).
    /// For IPM/SQP the smooth cost system is the right one to use for hessian queries,
    /// but the same approach works (barrier terms are not included in the hessian).
    pub fn build_solution(mut self) -> Result<OptimizationSolution, NllsError> {
        let final_linear_system = if self.params.skip_final_hessian {
            // Re-evaluate at current point (with damping to avoid issues).
            helpers::evaluate_cost_and_build_linear_system(
                &self.variables,
                &mut self.smooth_cost_system,
                &mut self.eq_system,
                self.params,
                self.current_pattern.take(),
                false,
            )?
            .0
        } else {
            // Evaluate with zero damping to get the true Hessian.
            self.smooth_cost_system.lm_damping = 0.0;
            helpers::evaluate_cost_and_build_linear_system(
                &self.variables,
                &mut self.smooth_cost_system,
                &mut self.eq_system,
                self.params,
                self.current_pattern.take(),
                false,
            )?
            .0
        };

        let (final_neg_gradient, final_damped_hessian) =
            final_linear_system.into_gradient_and_hessian();
        let constraint_jacobian = self.eq_system.dense_constraint_jacobian(&self.variables);
        Ok(OptimizationSolution::new(
            self.variables,
            self.merit,
            final_neg_gradient,
            final_damped_hessian,
            constraint_jacobian,
        ))
    }
}

/// Validate solver configuration before optimization.
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
