extern crate alloc;

use log::{
    debug,
    info,
};

use super::{
    optimizer::LineSearchResult,
    optimizer_helpers as helpers,
    *,
};

/// Elastic feasibility step (Wächter & Biegler §3.3): IPM-like inner step with
/// elastic slacks. Transitions to IPM/SQP when all h_j > 0.
///
/// During elastic phase: fixed μ/ρ, no outer transitions. When the inner loop
/// stalls, just reset LM damping and keep going. The elastic barrier gradient
/// drives the iterate toward feasibility while also minimizing the smooth cost.
pub(crate) fn step_elastic(opt: &mut Optimizer) -> Result<StepInfo, NllsError> {
    // Take up to ELASTIC_SUBSTEPS inner steps per call to accelerate feasibility.
    // Each substep is cheap (same sparsity pattern, warm-started).
    const ELASTIC_SUBSTEPS: usize = 10;

    let mut inner_result = None;
    for _ in 0..ELASTIC_SUBSTEPS {
        inner_result = Some(step_elastic_inner(opt)?);
        opt.total_step_count += 1;
        opt.inner_step_count += 1;

        // Check feasibility after each substep.
        let IneqStrategy::Elastic(ref e) = opt.strategy else {
            unreachable!();
        };
        let min_h: f64 = e
            .barriers
            .iter()
            .flat_map(|b| b.eval_h_values(&opt.variables))
            .fold(f64::INFINITY, f64::min);
        if min_h > 0.0 || inner_result.as_ref().unwrap().is_some() {
            break;
        }
        if opt.total_step_count >= opt.params.num_iterations {
            break;
        }
    }
    let inner_result = inner_result.unwrap();

    // Check if all constraints are now feasible.
    let IneqStrategy::Elastic(ref e) = opt.strategy else {
        unreachable!();
    };
    let all_h = helpers::eval_h_values_all(&e.barriers, &opt.variables);
    let min_h: f64 = all_h
        .iter()
        .flat_map(|hs| hs.iter())
        .cloned()
        .fold(f64::INFINITY, f64::min);

    if min_h > 0.0 {
        info!(
            "elastic: feasible (min_h={min_h:.4e}) after {} steps, transitioning to phase-2",
            opt.total_step_count
        );
        transition_elastic_to_phase2(opt)?;
        return Ok(StepInfo {
            merit: opt.merit,
            smooth_cost: opt.smooth_cost,
            did_outer_step: true,
            inner_step: 0,
            lm_damping: opt.lm_damping,
            in_phase1: true,
            termination: None,
        });
    }

    // When elastic inner loop stalls (converges to the current barrier subproblem
    // optimum but still infeasible), increase μ to strengthen the barrier gradient.
    // This is the opposite of the IPM outer transition (which decreases μ).
    // The larger μ widens the barrier, creating a steeper gradient that pushes
    // the iterate toward feasibility more aggressively.
    let inner_stalled = matches!(
        &inner_result,
        Some(TerminationReason::FunctionTolerance { .. })
            | Some(TerminationReason::ParameterTolerance { .. })
            | Some(TerminationReason::GradientTolerance { .. })
            | Some(TerminationReason::MaxConsecutiveRejects { .. })
    );
    let mut did_outer_step = false;
    if inner_stalled {
        if let IneqStrategy::Elastic(ref mut e) = opt.strategy {
            // Increase μ by 2x to strengthen barrier gradient toward feasibility.
            e.mu *= 2.0;
            // Recompute elastic slacks and lambdas with the new μ.
            let all_slacks =
                helpers::compute_all_elastic_slacks(&e.barriers, &opt.variables, e.mu, e.rho);
            helpers::set_elastic_state(&e.barriers, &all_slacks, &opt.variables, e.mu);
            // Recompute merit with new parameters.
            (opt.smooth_cost, opt.merit) = helpers::calc_elastic_merit(
                &opt.variables,
                &opt.smooth_cost_system,
                &e.barriers,
                &all_slacks,
                e.mu,
                e.rho,
                opt.params.parallelize,
            )
            .map_err(|e| NllsError::NllsCostSystemError { source: e })?;
        }
        opt.lm_damping = opt.params.initial_lm_damping;
        opt.consecutive_rejects = 0;
        opt.inner_step_count = 0;
        did_outer_step = true;
    }

    let termination = if opt.total_step_count >= opt.params.num_iterations {
        Some(TerminationReason::MaxIterations)
    } else {
        None
    };

    Ok(StepInfo {
        merit: opt.merit,
        smooth_cost: opt.smooth_cost,
        did_outer_step,
        inner_step: if did_outer_step {
            0
        } else {
            opt.inner_step_count - 1
        },
        lm_damping: opt.lm_damping,
        in_phase1: true,
        termination,
    })
}

/// Elastic inner step: IPM-like step with elastic slacks set on barriers.
///
/// The barrier evaluates with shifted constraints s_j = h_j + v_j (always > 0).
/// Step acceptance uses filter-like criteria: accept if merit OR smooth cost improves.
fn step_elastic_inner(opt: &mut Optimizer) -> Result<Option<TerminationReason>, NllsError> {
    let IneqStrategy::Elastic(ref e) = opt.strategy else {
        unreachable!();
    };

    // Evaluate barrier costs (they have elastic slacks set, so they use s_j).
    let barrier_evals: alloc::vec::Vec<alloc::boxed::Box<dyn IsEvaluatedCost>> = e
        .barriers
        .iter()
        .map(|b| b.eval(&opt.variables, EvalMode::CalculateDerivatives, false))
        .collect::<Result<_, _>>()
        .map_err(|e| NllsError::NllsCostSystemError { source: e })?;

    let solve_result = helpers::build_and_solve_ineq_system(opt, barrier_evals);
    let (delta, linear_system) = match solve_result? {
        Some(pair) => pair,
        None => {
            return Ok(opt.handle_solve_failure());
        }
    };

    if let Some(reason) = opt.check_gradient_tolerance_unless_eq(&linear_system) {
        return Ok(Some(reason));
    }

    let IneqStrategy::Elastic(ref e) = opt.strategy else {
        unreachable!();
    };
    let tau = e.tau;
    let mu = e.mu;
    let rho = e.rho;

    // Compute old s values (h + v) for fraction-to-boundary.
    let h_olds = helpers::eval_h_values_all(&e.barriers, &opt.variables);
    let old_elastic: alloc::vec::Vec<alloc::vec::Vec<f64>> =
        e.barriers.iter().map(|b| b.elastic_slacks()).collect();
    let s_olds: alloc::vec::Vec<alloc::vec::Vec<f64>> = h_olds
        .iter()
        .zip(old_elastic.iter())
        .map(|(hs, vs)| hs.iter().zip(vs.iter()).map(|(&h, &v)| h + v).collect())
        .collect();

    let result = helpers::elastic_line_search(opt, &delta, &s_olds, tau, mu, rho)?;

    match result {
        Some(search_result) => {
            let neg_grad = linear_system.neg_gradient.scalar_vector();
            let gain_ratio = helpers::compute_gain_ratio(
                opt.lm_damping,
                &delta,
                neg_grad,
                opt.merit,
                search_result.merit,
            );

            // Filter-like acceptance: accept if merit OR smooth cost improves.
            let cost_improved =
                search_result.smooth_cost < opt.smooth_cost - 1e-4 * opt.smooth_cost.abs().max(1.0);
            let merit_improved = search_result.merit < opt.merit;
            // Also accept if infeasibility decreased (θ-type acceptance).
            let theta_old = helpers::compute_theta(opt.strategy.barriers(), &opt.variables);
            let theta_new =
                helpers::compute_theta(opt.strategy.barriers(), &search_result.proposed);
            let theta_improved = theta_new < (1.0 - 1e-5) * theta_old;
            let accept = (gain_ratio > 0.0 && merit_improved) || cost_improved || theta_improved;

            if accept {
                opt.accept_step(gain_ratio.max(0.1));
                let reason = opt.check_function_tolerance(search_result.merit);

                let updated_lambdas = opt.eq_system.update_lambdas(&opt.variables, &delta);
                opt.eq_system.lambda = updated_lambdas;

                opt.variables = search_result.proposed;

                // Recompute elastic slacks at the new point.
                let IneqStrategy::Elastic(ref e) = opt.strategy else {
                    unreachable!();
                };
                let new_slacks =
                    helpers::compute_all_elastic_slacks(&e.barriers, &opt.variables, mu, rho);
                helpers::set_elastic_state(&e.barriers, &new_slacks, &opt.variables, mu);

                opt.project_onto_eq_manifold()?;

                opt.smooth_cost = search_result.smooth_cost;
                opt.merit = search_result.merit;
                Ok(reason)
            } else {
                Ok(opt.reject_step())
            }
        }
        None => Ok(opt.handle_solve_failure()),
    }
}

/// Transition from elastic feasibility to IPM or SQP.
///
/// The smooth cost system and eq system are already built (they were included
/// during the elastic phase). Just need to clear elastic slacks and set up
/// the IPM/SQP strategy on the existing barriers.
pub(crate) fn transition_elastic_to_phase2(opt: &mut Optimizer) -> Result<(), NllsError> {
    let IneqStrategy::Elastic(elastic) = core::mem::replace(
        &mut opt.strategy,
        IneqStrategy::None, // temporary
    ) else {
        unreachable!();
    };

    let ElasticState { barriers, .. } = elastic;

    // Clear elastic slacks on all barriers.
    for b in barriers.iter() {
        b.set_elastic_slacks(alloc::vec![]);
    }

    // Compute adaptive mu for the IPM/SQP phase.
    let initial_smooth_cost = {
        let mut sc = 0.0_f64;
        for cf in opt.smooth_cost_system.cost_fns.iter() {
            if let Ok(v) = cf.calc_total_cost(&opt.variables, opt.params.parallelize) {
                sc += v;
            }
        }
        sc
    };
    let num_ineq: usize = barriers
        .iter()
        .map(|b| b.eval_h_values(&opt.variables).len())
        .sum();
    let sigma = 0.1_f64;
    let adaptive_mu = if num_ineq > 0 && initial_smooth_cost > 1e-12 {
        (sigma * initial_smooth_cost / num_ineq as f64).max(1e-8)
    } else {
        1e-8
    };

    // Set up IPM or SQP strategy.
    match opt.params.ineq_method {
        IneqMethod::Ipm { tau, .. } => {
            for b in barriers.iter() {
                let h_values = b.eval_h_values(&opt.variables);
                let lambdas: alloc::vec::Vec<f64> = h_values
                    .iter()
                    .map(|&h| {
                        if h > 1e-6 {
                            adaptive_mu / h
                        } else {
                            adaptive_mu
                        }
                    })
                    .collect();
                b.set_lambdas(lambdas);
            }
            let (sc, m) = helpers::calc_ipm_merit(
                &opt.variables,
                &opt.smooth_cost_system,
                &barriers,
                &opt.eq_system,
                opt.params.parallelize,
            )
            .map_err(|e| NllsError::NllsCostSystemError { source: e })?;
            opt.smooth_cost = sc;
            opt.merit = m;
            let theta_0 = helpers::compute_theta(&barriers, &opt.variables);
            let filter = super::filter::Filter::new(1e4 * theta_0.max(1.0));
            opt.strategy = IneqStrategy::Ipm(IpmState {
                barriers,
                mu: adaptive_mu,
                tau,
                filter,
            });
        }
        IneqMethod::Sqp { tau, .. } => {
            let frozen_lins: alloc::vec::Vec<alloc::vec::Vec<ConstraintLinearization>> = barriers
                .iter()
                .map(|b| b.eval_constraint_linearizations(&opt.variables))
                .collect();
            let mut slacks: alloc::vec::Vec<alloc::vec::Vec<f64>> = alloc::vec![];
            let mut duals: alloc::vec::Vec<alloc::vec::Vec<f64>> = alloc::vec![];
            for b in barriers.iter() {
                let h = b.eval_h_values(&opt.variables);
                let d: alloc::vec::Vec<f64> =
                    h.iter().map(|&s| adaptive_mu / s.max(1e-6)).collect();
                duals.push(d);
                slacks.push(h);
            }
            let (sc, m) = helpers::calc_sqp_merit(
                &opt.variables,
                &opt.smooth_cost_system,
                &barriers,
                &duals,
                &opt.eq_system,
                opt.params.parallelize,
            )
            .map_err(|e| NllsError::NllsCostSystemError { source: e })?;
            opt.smooth_cost = sc;
            opt.merit = m;
            opt.strategy = IneqStrategy::Sqp(SqpState {
                barriers,
                frozen_lins,
                duals,
                slacks,
                mu: adaptive_mu,
                tau,
            });
        }
    }

    // Rebuild pattern and reset counters.
    opt.current_pattern = helpers::maybe_build_ineq_pattern(
        &opt.variables,
        &opt.smooth_cost_system,
        opt.strategy.barriers(),
        &opt.eq_system,
    );
    opt.lm_damping = opt.params.initial_lm_damping;
    opt.consecutive_rejects = 0;
    opt.inner_step_count = 0;
    opt.stalled_outer_count = 0;

    Ok(())
}

/// NLLS / equality-constrained inner step (no inequality constraints).
pub(crate) fn step_nlls_inner(opt: &mut Optimizer) -> Result<Option<TerminationReason>, NllsError> {
    debug!("lm-damping: {}", opt.lm_damping);
    opt.smooth_cost_system.lm_damping = opt.lm_damping;
    let (mut linear_system, next_pattern) = helpers::evaluate_cost_and_build_linear_system(
        &opt.variables,
        &mut opt.smooth_cost_system,
        &mut opt.eq_system,
        opt.params,
        opt.current_pattern.take(),
        false,
    )?;
    opt.current_pattern = next_pattern;

    // Check gradient tolerance before solving (skip when eq constraints are
    // present — the augmented KKT gradient includes Lagrange multiplier terms
    // that can appear small before the objective is minimized).
    if let Some(reason) = opt.check_gradient_tolerance_unless_eq(&linear_system) {
        return Ok(Some(reason));
    }

    let delta = match helpers::try_solve(&mut linear_system)? {
        Some(d) => d,
        None => return Ok(opt.handle_solve_failure()),
    };

    let updated_families = opt.variables.update(&delta);
    let updated_lambdas = opt.eq_system.update_lambdas(&opt.variables, &delta);
    let new_merit =
        helpers::calc_merit(&updated_families, &mut opt.smooth_cost_system, opt.params)?;

    // Check parameter tolerance (max component of tangent-space step).
    let max_step = delta.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    if max_step < opt.params.parameter_tolerance {
        if new_merit < opt.merit {
            opt.variables = updated_families;
            opt.eq_system.lambda = updated_lambdas;
            opt.smooth_cost = new_merit;
            opt.merit = new_merit;
        }
        return Ok(Some(TerminationReason::ParameterTolerance {
            max_step,
            threshold: opt.params.parameter_tolerance,
        }));
    }

    // Gain ratio ρ = (actual decrease) / (predicted decrease by the linear model).
    //   predicted = δᵀ(ν·δ + g), where g = -neg_gradient
    // This measures how well the quadratic model approximates the true cost.
    let neg_grad = linear_system.neg_gradient.scalar_vector();
    let gain_ratio =
        helpers::compute_gain_ratio(opt.lm_damping, &delta, neg_grad, opt.merit, new_merit);

    let reason;
    if gain_ratio > 0.0 && new_merit < opt.merit {
        // Accepted step — decrease damping using Nielsen's formula.
        opt.accept_step(gain_ratio);
        reason = opt.check_function_tolerance(new_merit);
        opt.variables = updated_families;
        opt.eq_system.lambda = updated_lambdas;
        opt.smooth_cost = new_merit;
        opt.merit = new_merit;
    } else {
        // Rejected step — increase damping.
        reason = opt.reject_step();
    }

    info!(
        "lm-damping: {:?}, merit {:?}, (new_merit {:?})",
        opt.lm_damping, opt.merit, new_merit
    );
    Ok(reason)
}

/// IPM inner step with filter line search (Wächter & Biegler, 2006).
///
/// Step acceptance uses a filter on (θ, φ) where θ = infeasibility measure
/// and φ = barrier objective. This replaces the simple gain-ratio check and
/// provides proven global convergence: the filter prevents cycling, while
/// the fraction-to-boundary rule ensures iterates stay interior.
pub(crate) fn step_ipm_inner(opt: &mut Optimizer) -> Result<Option<TerminationReason>, NllsError> {
    let IneqStrategy::Ipm(ref ipm) = opt.strategy else {
        unreachable!();
    };

    // Evaluate barrier costs as extra evaluated costs.
    let barrier_evals: alloc::vec::Vec<alloc::boxed::Box<dyn IsEvaluatedCost>> = ipm
        .barriers
        .iter()
        .map(|b| b.eval(&opt.variables, EvalMode::CalculateDerivatives, false))
        .collect::<Result<_, _>>()
        .map_err(|e| NllsError::NllsCostSystemError { source: e })?;

    let solve_result = helpers::build_and_solve_ineq_system(opt, barrier_evals);
    let (delta, linear_system) = match solve_result? {
        Some(pair) => pair,
        None => {
            return Ok(opt.handle_solve_failure());
        }
    };

    if let Some(reason) = opt.check_gradient_tolerance_unless_eq(&linear_system) {
        return Ok(Some(reason));
    }

    let IneqStrategy::Ipm(ref ipm) = opt.strategy else {
        unreachable!();
    };
    let tau = ipm.tau;
    let h_olds = helpers::eval_h_values_all(&ipm.barriers, &opt.variables);

    // Current (θ, φ) for filter check.
    let theta_current = helpers::compute_theta(opt.strategy.barriers(), &opt.variables);
    let phi_current = opt.merit;

    // Predicted decrease from the linearized model (for switching condition).
    let neg_grad = linear_system.neg_gradient.scalar_vector();
    let predicted_decrease = opt.lm_damping * delta.dot(&delta) + delta.dot(neg_grad);

    // Fraction-to-boundary line search with filter acceptance.
    let smooth = &opt.smooth_cost_system;
    let parallelize = opt.params.parallelize;
    let barriers = opt.strategy.barriers();
    let eq_sys = &opt.eq_system;

    // Backtracking line search: try α = 1, 0.5, 0.25, ... until filter accepts.
    let mut alpha = 1.0_f64;
    let mut accepted_result: Option<(LineSearchResult, bool)> = None; // (result, is_phi_type)
    loop {
        let scaled = &delta * alpha;
        let proposed = opt.variables.update(&scaled);
        let h_news = helpers::eval_h_values_all(barriers, &proposed);

        if helpers::fraction_to_boundary_ok(&h_olds, &h_news, tau) {
            let (new_smooth_cost, new_merit) =
                helpers::calc_ipm_merit(&proposed, smooth, barriers, eq_sys, parallelize)
                    .map_err(|e| NllsError::NllsCostSystemError { source: e })?;

            let theta_trial = helpers::compute_theta(barriers, &proposed);
            let phi_trial = new_merit;

            // Filter step classification (Wächter & Biegler Algorithm A).
            let IneqStrategy::Ipm(ref ipm) = opt.strategy else {
                unreachable!();
            };
            let filter_result = ipm.filter.check_step(
                theta_current,
                phi_current,
                theta_trial,
                phi_trial,
                predicted_decrease * alpha,
            );

            if let Some(is_phi_type) = filter_result {
                accepted_result = Some((
                    LineSearchResult {
                        proposed,
                        h_values: h_news,
                        smooth_cost: new_smooth_cost,
                        merit: new_merit,
                    },
                    is_phi_type,
                ));
                break;
            }

            // Fallback: when fully feasible (θ ≈ 0), the filter's φ-type
            // criterion can be too strict after μ changes (the merit function
            // itself changed). Fall back to gain-ratio acceptance.
            if theta_current < 1e-10 && theta_trial < 1e-10 {
                let trial_gain = helpers::compute_gain_ratio(
                    opt.lm_damping,
                    &delta,
                    neg_grad,
                    phi_current,
                    phi_trial,
                );
                let cost_improved =
                    new_smooth_cost < opt.smooth_cost - 1e-4 * opt.smooth_cost.abs().max(1.0);
                if (trial_gain > 0.0 && phi_trial < phi_current) || cost_improved {
                    accepted_result = Some((
                        LineSearchResult {
                            proposed,
                            h_values: h_news,
                            smooth_cost: new_smooth_cost,
                            merit: new_merit,
                        },
                        true, // treat as φ-type (no filter augmentation)
                    ));
                    break;
                }
            }
        }

        alpha *= 0.5;
        if alpha < 1e-12 {
            break;
        }
    }

    match accepted_result {
        Some((search_result, is_phi_type)) => {
            let gain_ratio = helpers::compute_gain_ratio(
                opt.lm_damping,
                &delta,
                neg_grad,
                opt.merit,
                search_result.merit,
            );
            opt.accept_step(gain_ratio.max(0.1));
            let reason = opt.check_function_tolerance(search_result.merit);

            // If θ-type step, augment the filter with current (θ, φ).
            // This prevents the algorithm from cycling back to this region.
            if !is_phi_type {
                let IneqStrategy::Ipm(ref mut ipm) = opt.strategy else {
                    unreachable!();
                };
                ipm.filter.augment(theta_current, phi_current);
            }

            // Update equality constraint lambdas.
            let updated_lambdas = opt.eq_system.update_lambdas(&opt.variables, &delta);
            opt.eq_system.lambda = updated_lambdas;

            // Multiplicative barrier lambda update.
            let IneqStrategy::Ipm(ref ipm) = opt.strategy else {
                unreachable!();
            };
            for (b, (ho, hn)) in ipm
                .barriers
                .iter()
                .zip(h_olds.iter().zip(search_result.h_values.iter()))
            {
                let new_lambdas: alloc::vec::Vec<f64> = b
                    .lambdas()
                    .iter()
                    .zip(ho.iter().zip(hn.iter()))
                    .map(|(&l, (&ho_j, &hn_j))| (l * ho_j / hn_j.max(1e-10)).min(1e8))
                    .collect();
                b.set_lambdas(new_lambdas);
            }

            opt.variables = search_result.proposed;
            opt.project_onto_eq_manifold()?;
            opt.smooth_cost = search_result.smooth_cost;
            opt.merit = search_result.merit;
            Ok(reason)
        }
        None => {
            // Filter rejected all step sizes (Wächter & Biegler §3.3).
            //
            // If the current iterate has significant constraint violation (θ > 0),
            // enter feasibility restoration: switch to elastic mode which
            // aggressively reduces infeasibility. The elastic phase will
            // transition back to IPM once all constraints are satisfied.
            //
            // If the iterate is feasible (θ ≈ 0), the rejection is due to
            // the objective — increase LM damping for a more conservative step.
            if theta_current > 1e-8 {
                info!(
                    "IPM filter rejection with θ={theta_current:.4e}: entering feasibility restoration"
                );
                enter_feasibility_restoration(opt, theta_current, phi_current)?;
                Ok(None) // no termination, restoration phase will take over
            } else {
                Ok(opt.reject_step())
            }
        }
    }
}

/// Enter feasibility restoration from IPM (Wächter & Biegler §3.3).
///
/// When the filter rejects all step sizes and the iterate is infeasible,
/// switch to elastic mode to aggressively reduce constraint violation.
/// The current (θ, φ) is added to the filter to prevent cycling back.
///
/// The elastic phase runs with the same smooth cost system (no rebuild needed),
/// and transitions back to IPM via `transition_elastic_to_phase2` once feasible.
fn enter_feasibility_restoration(
    opt: &mut Optimizer,
    theta_current: f64,
    phi_current: f64,
) -> Result<(), NllsError> {
    // Extract barriers from current IPM state.
    let IneqStrategy::Ipm(ipm) = core::mem::replace(&mut opt.strategy, IneqStrategy::None) else {
        unreachable!();
    };

    let IpmState {
        barriers,
        mu,
        tau,
        mut filter,
    } = ipm;

    // Augment filter with current (θ, φ) to prevent cycling back to this region.
    filter.augment(theta_current, phi_current);

    // Compute elastic parameters.
    // μ should be large enough for the barrier to push toward feasibility.
    let max_violation: f64 = barriers
        .iter()
        .flat_map(|b| b.eval_h_values(&opt.variables))
        .filter(|&h| h < 0.0)
        .map(|h| h.abs())
        .fold(0.0_f64, f64::max);

    let elastic_mu = mu.max(max_violation).max(1e-4);
    let rho = (100.0 * elastic_mu).max(100.0 * max_violation).max(100.0);

    // Set elastic slacks on barriers.
    let all_slacks =
        helpers::compute_all_elastic_slacks(&barriers, &opt.variables, elastic_mu, rho);
    helpers::set_elastic_state(&barriers, &all_slacks, &opt.variables, elastic_mu);

    // Recompute merit.
    (opt.smooth_cost, opt.merit) = helpers::calc_elastic_merit(
        &opt.variables,
        &opt.smooth_cost_system,
        &barriers,
        &all_slacks,
        elastic_mu,
        rho,
        opt.params.parallelize,
    )
    .map_err(|e| NllsError::NllsCostSystemError { source: e })?;

    opt.strategy = IneqStrategy::Elastic(ElasticState {
        barriers,
        mu: elastic_mu,
        rho,
        tau,
    });

    // Reset LM damping for the restoration phase.
    opt.lm_damping = opt.params.initial_lm_damping;
    opt.consecutive_rejects = 0;
    opt.inner_step_count = 0;

    Ok(())
}

/// SQP inner step with frozen Jacobians + gain ratio.
pub(crate) fn step_sqp_inner(opt: &mut Optimizer) -> Result<Option<TerminationReason>, NllsError> {
    let IneqStrategy::Sqp(ref sqp) = opt.strategy else {
        unreachable!();
    };

    // Build frozen barrier contribution as a single extra cost.
    let frozen_cost: alloc::boxed::Box<dyn IsEvaluatedCost> = alloc::boxed::Box::new(
        helpers::build_frozen_barrier_cost(&sqp.frozen_lins, &sqp.duals, &sqp.slacks),
    );

    let (delta, linear_system) =
        match helpers::build_and_solve_ineq_system(opt, alloc::vec![frozen_cost])? {
            Some(pair) => pair,
            None => return Ok(opt.handle_solve_failure()),
        };

    // Check gradient tolerance (skip for combined eq+ineq).
    if let Some(reason) = opt.check_gradient_tolerance_unless_eq(&linear_system) {
        return Ok(Some(reason));
    }

    let IneqStrategy::Sqp(ref sqp) = opt.strategy else {
        unreachable!();
    };
    let tau = sqp.tau;
    let mu = sqp.mu;
    let h_olds = helpers::eval_h_values_all(&sqp.barriers, &opt.variables);

    // Capture references for the merit closure.
    let smooth = &opt.smooth_cost_system;
    let parallelize = opt.params.parallelize;
    let barriers = opt.strategy.barriers();
    let eq_sys = &opt.eq_system;
    let duals = match &opt.strategy {
        IneqStrategy::Sqp(sqp) => &sqp.duals,
        _ => unreachable!(),
    };

    let result =
        helpers::fraction_to_boundary_line_search(opt, &delta, &h_olds, tau, |proposed| {
            helpers::calc_sqp_merit(proposed, smooth, barriers, duals, eq_sys, parallelize)
                .map_err(|e| NllsError::NllsCostSystemError { source: e })
        })?;

    match result {
        Some(search_result) => {
            // Gain ratio.
            let neg_grad = linear_system.neg_gradient.scalar_vector();
            let gain_ratio = helpers::compute_gain_ratio(
                opt.lm_damping,
                &delta,
                neg_grad,
                opt.merit,
                search_result.merit,
            );

            if gain_ratio > 0.0 && search_result.merit < opt.merit {
                // Accepted.
                opt.accept_step(gain_ratio);

                let reason = opt.check_function_tolerance(search_result.merit);

                // Update equality constraint lambdas.
                let updated_lambdas = opt.eq_system.update_lambdas(&opt.variables, &delta);
                opt.eq_system.lambda = updated_lambdas;

                // Update slacks and duals from new actual h values.
                let IneqStrategy::Sqp(ref mut sqp) = opt.strategy else {
                    unreachable!();
                };
                for i in 0..sqp.barriers.len() {
                    sqp.slacks[i] = search_result.h_values[i].clone();
                    for (z, &s) in sqp.duals[i]
                        .iter_mut()
                        .zip(search_result.h_values[i].iter())
                    {
                        *z = mu / s.max(1e-6);
                    }
                }

                opt.variables = search_result.proposed;

                // Project back onto the equality constraint manifold.
                opt.project_onto_eq_manifold()?;

                opt.smooth_cost = search_result.smooth_cost;
                opt.merit = search_result.merit;
                Ok(reason)
            } else {
                // Rejected.
                Ok(opt.reject_step())
            }
        }
        None => Ok(opt.handle_solve_failure()),
    }
}

/// Perform an outer transition: decay parameters, reset damping, refreeze (SQP).
///
/// `inner_converged` indicates whether the inner loop actually converged
/// (FunctionTolerance/ParameterTolerance/GradientTolerance) vs just stalling
/// (MaxConsecutiveRejects) or hitting the step budget. μ is only decreased
/// when the current barrier subproblem was approximately solved.
pub(crate) fn do_outer_transition(
    opt: &mut Optimizer,
    decay: f64,
    inner_converged: bool,
) -> Result<(), NllsError> {
    match &mut opt.strategy {
        IneqStrategy::None | IneqStrategy::Elastic(_) => {}
        IneqStrategy::Ipm(ipm) => {
            // Decrease μ only when the inner loop actually converged
            // (the current barrier subproblem was approximately solved).
            // When the inner loop just stalled (rejected steps) or hit the
            // budget, keep μ and let the inner loop try again with reset damping.
            if inner_converged {
                let new_mu = (ipm.mu * decay).max(opt.params.function_tolerance / 10.0);
                ipm.mu = new_mu;

                // Set lambdas to μ/h_j (re-center on the new barrier subproblem).
                for b in ipm.barriers.iter() {
                    let h_values = b.eval_h_values(&opt.variables);
                    let new_lam: alloc::vec::Vec<f64> = h_values
                        .iter()
                        .map(|&h| if h > 1e-6 { new_mu / h } else { new_mu })
                        .collect();
                    b.set_lambdas(new_lam);
                }

                // Reset filter when barrier parameter changes (Wächter & Biegler §3.1).
                ipm.filter.reset();
            }
        }
        IneqStrategy::Sqp(sqp) => {
            // Re-freeze constraint Jacobians at current outer iterate.
            sqp.frozen_lins = sqp
                .barriers
                .iter()
                .map(|b| b.eval_constraint_linearizations(&opt.variables))
                .collect();

            // Re-initialize slacks from actual h values; reset duals to mu/s.
            sqp.slacks = sqp
                .barriers
                .iter()
                .map(|b| b.eval_h_values(&opt.variables))
                .collect();
            for (sv, dv) in sqp.slacks.iter().zip(sqp.duals.iter_mut()) {
                for (s, z) in sv.iter().zip(dv.iter_mut()) {
                    *z = sqp.mu / s.max(1e-6);
                }
            }

            // Decay barrier parameter.
            sqp.mu *= decay;

            // Recompute merit with freshly initialized duals.
            (opt.smooth_cost, opt.merit) = helpers::calc_sqp_merit(
                &opt.variables,
                &opt.smooth_cost_system,
                &sqp.barriers,
                &sqp.duals,
                &opt.eq_system,
                opt.params.parallelize,
            )
            .map_err(|e| NllsError::NllsCostSystemError { source: e })?;
        }
    }

    // Reset LM damping for the next inner loop.
    opt.lm_damping = opt.params.initial_lm_damping;
    opt.consecutive_rejects = 0;
    opt.inner_step_count = 0;

    Ok(())
}
