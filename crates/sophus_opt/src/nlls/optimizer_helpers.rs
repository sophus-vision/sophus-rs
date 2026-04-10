extern crate alloc;

use log::debug;
use sophus_solver::matrix::block_sparse::BlockSparseSymmetricMatrixPattern;

use super::{
    optimizer::LineSearchResult,
    *,
};

/// Evaluate h values at the given variables for all barriers.
pub(crate) fn eval_h_values_all(
    barriers: &[alloc::boxed::Box<dyn IsIneqBarrierFn>],
    variables: &VarFamilies,
) -> alloc::vec::Vec<alloc::vec::Vec<f64>> {
    barriers
        .iter()
        .map(|b| b.eval_h_values(variables))
        .collect()
}

/// Fraction-to-boundary check: all new h values satisfy h_new >= (1 - tau) * h_old.
pub(crate) fn fraction_to_boundary_ok(
    h_olds: &[alloc::vec::Vec<f64>],
    h_news: &[alloc::vec::Vec<f64>],
    tau: f64,
) -> bool {
    h_olds.iter().zip(h_news.iter()).all(|(ho, hn)| {
        ho.iter()
            .zip(hn.iter())
            .all(|(ho_j, hn_j)| *hn_j >= (1.0 - tau) * ho_j)
    })
}

/// Compute the total smooth cost (sum of all smooth cost function terms).
pub(crate) fn calc_smooth_cost(
    smooth: &CostSystem,
    variables: &VarFamilies,
    parallelize: bool,
) -> Result<f64, CostError> {
    let mut sc = 0.0_f64;
    for cf in smooth.cost_fns.iter() {
        sc += cf.calc_total_cost(variables, parallelize)?;
    }
    Ok(sc)
}

/// Compute IPM merit: smooth cost + barrier cost + equality penalty.
/// Returns `(smooth_cost, merit)`.
pub(crate) fn calc_ipm_merit(
    variables: &VarFamilies,
    smooth: &CostSystem,
    barriers: &[alloc::boxed::Box<dyn IsIneqBarrierFn>],
    eq_system: &EqSystem,
    parallelize: bool,
) -> Result<(f64, f64), CostError> {
    let sc = calc_smooth_cost(smooth, variables, parallelize)?;
    let mut m = sc;
    for b in barriers.iter() {
        m += b.calc_total_cost(variables, parallelize)?;
    }
    m += eq_system.calc_eq_penalty(variables);
    Ok((sc, m))
}

/// Compute SQP merit: smooth cost + sum(-z_j * ln(h_j)) + equality penalty.
/// Returns `(smooth_cost, merit)`.
pub(crate) fn calc_sqp_merit(
    variables: &VarFamilies,
    smooth: &CostSystem,
    barriers: &[alloc::boxed::Box<dyn IsIneqBarrierFn>],
    duals: &[alloc::vec::Vec<f64>],
    eq_system: &EqSystem,
    parallelize: bool,
) -> Result<(f64, f64), CostError> {
    let sc = calc_smooth_cost(smooth, variables, parallelize)?;
    let mut m = sc;
    const EPSILON: f64 = 1e-6;
    for (b_idx, b) in barriers.iter().enumerate() {
        let h_vals = b.eval_h_values(variables);
        for (j, &h) in h_vals.iter().enumerate() {
            let z = duals[b_idx][j];
            if h > EPSILON {
                m += -z * h.ln();
            } else {
                let violation = h - EPSILON;
                m += z * violation * violation / (2.0 * EPSILON * EPSILON);
            }
        }
    }
    m += eq_system.calc_eq_penalty(variables);
    Ok((sc, m))
}

/// Build a FrozenBarrierEvaluatedCost from frozen linearizations and current duals/slacks.
pub(crate) fn build_frozen_barrier_cost(
    frozen_lins: &[alloc::vec::Vec<ConstraintLinearization>],
    duals: &[alloc::vec::Vec<f64>],
    slacks: &[alloc::vec::Vec<f64>],
) -> FrozenBarrierEvaluatedCost {
    let mut entries = alloc::vec![];
    for (b_idx, lins) in frozen_lins.iter().enumerate() {
        for (j, lin) in lins.iter().enumerate() {
            let z = duals[b_idx][j];
            let s = slacks[b_idx][j];
            entries.push((lin.clone(), z, s));
        }
    }
    FrozenBarrierEvaluatedCost { entries }
}

/// Build the sparsity pattern for inequality-constrained problems (IPM/SQP).
pub(crate) fn build_ineq_pattern(
    variables: &VarFamilies,
    smooth: &CostSystem,
    barriers: &[alloc::boxed::Box<dyn IsIneqBarrierFn>],
    eq_system: &EqSystem,
) -> BlockSparseSymmetricMatrixPattern {
    use sophus_solver::matrix::{
        PartitionSet,
        block_sparse::BlockSparseSymmetricSymbolicBuilder,
    };
    let mut partition_specs = variables.build_partition_specs();
    partition_specs.extend(eq_system.partitions.clone());
    let partitions = PartitionSet::new(partition_specs);
    let mut sym_builder = BlockSparseSymmetricSymbolicBuilder::new(partitions);

    for cf in smooth.cost_fns.iter() {
        cf.populate_symbolic(variables, &mut sym_builder);
    }
    for b in barriers.iter() {
        b.populate_symbolic(variables, &mut sym_builder);
    }
    sym_builder.into_pattern()
}

/// Build the sparsity pattern for inequality-constrained problems, unless equality
/// constraints are present (in which case the pattern cannot be pre-built).
///
/// Returns `Some(pattern)` when there are no equality constraints, `None` otherwise.
pub(crate) fn maybe_build_ineq_pattern(
    variables: &VarFamilies,
    smooth: &CostSystem,
    barriers: &[alloc::boxed::Box<dyn IsIneqBarrierFn>],
    eq_system: &EqSystem,
) -> Option<BlockSparseSymmetricMatrixPattern> {
    if eq_system.partitions.is_empty() {
        Some(build_ineq_pattern(variables, smooth, barriers, eq_system))
    } else {
        None
    }
}

/// Compute the gain ratio (actual / predicted cost decrease) for LM step acceptance.
///
/// `neg_grad` is the negated gradient vector from the linear system.
pub(crate) fn compute_gain_ratio(
    lm_damping: f64,
    delta: &nalgebra::DVector<f64>,
    neg_grad: &nalgebra::DVector<f64>,
    old_merit: f64,
    new_merit: f64,
) -> f64 {
    let predicted_decrease = lm_damping * delta.dot(delta) + delta.dot(neg_grad);
    let actual_decrease = old_merit - new_merit;
    if predicted_decrease.abs() > f64::MIN_POSITIVE {
        actual_decrease / predicted_decrease
    } else {
        if actual_decrease > 0.0 { 1.0 } else { 0.0 }
    }
}

/// Solve the linear system, handling solver failures uniformly.
pub(crate) fn try_solve(
    linear_system: &mut LinearSystem,
) -> Result<Option<nalgebra::DVector<f64>>, NllsError> {
    match linear_system.solve() {
        Ok(d) => Ok(Some(d)),
        Err(NllsError::LinearSolver { .. }) => Ok(None),
        Err(NllsError::SchurComplementFailed) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Build the augmented linear system (smooth + barrier + eq constraints) and solve it.
///
/// Returns `(delta, linear_system)` on success, `None` on solver failure.
/// Temporarily appends `extra_costs` to the evaluated costs, builds the normal
/// equations, then truncates back to the original length.
pub(crate) fn build_and_solve_ineq_system(
    opt: &mut Optimizer,
    extra_costs: alloc::vec::Vec<alloc::boxed::Box<dyn IsEvaluatedCost>>,
) -> Result<Option<(nalgebra::DVector<f64>, LinearSystem)>, NllsError> {
    // Evaluate smooth costs.
    opt.smooth_cost_system.lm_damping = opt.lm_damping;
    opt.smooth_cost_system
        .eval(&opt.variables, EvalMode::CalculateDerivatives, opt.params)
        .map_err(|e| NllsError::NllsCostSystemError { source: e })?;

    // Evaluate equality constraints (critical for combined eq+ineq).
    opt.eq_system
        .eval(&opt.variables, EvalMode::CalculateDerivatives, opt.params)
        .map_err(|e| NllsError::NllsEqConstraintSystemError { source: e })?;

    let original_len = opt.smooth_cost_system.evaluated_costs.len();
    opt.smooth_cost_system.evaluated_costs.extend(extra_costs);

    let (mut linear_system, next_pattern) = LinearSystem::from_families_costs_and_constraints(
        &opt.variables,
        &opt.smooth_cost_system.evaluated_costs,
        opt.lm_damping,
        &opt.eq_system,
        opt.params.solver,
        opt.params.parallelize,
        opt.current_pattern.take(),
    )
    .map_err(|e| NllsError::NllsCostSystemError { source: e })?;
    opt.current_pattern = next_pattern;
    opt.smooth_cost_system
        .evaluated_costs
        .truncate(original_len);

    match try_solve(&mut linear_system)? {
        Some(delta) => Ok(Some((delta, linear_system))),
        None => Ok(None),
    }
}

/// Fraction-to-boundary line search for IPM/SQP.
///
/// Returns `(proposed_vars, h_news, new_smooth_cost, new_merit)` on success,
/// or `None` if line search exhausted (alpha < 1e-12).
pub(crate) fn fraction_to_boundary_line_search(
    opt: &Optimizer,
    delta: &nalgebra::DVector<f64>,
    h_olds: &[alloc::vec::Vec<f64>],
    tau: f64,
    calc_merit_fn: impl Fn(&VarFamilies) -> Result<(f64, f64), NllsError>,
) -> Result<Option<LineSearchResult>, NllsError> {
    let barriers = opt.strategy.barriers();
    let mut alpha = 1.0_f64;
    loop {
        let scaled: nalgebra::DVector<f64> = delta * alpha;
        let proposed = opt.variables.update(&scaled);
        let h_news = eval_h_values_all(barriers, &proposed);

        if fraction_to_boundary_ok(h_olds, &h_news, tau) {
            let (new_smooth_cost, new_merit) = calc_merit_fn(&proposed)?;
            return Ok(Some(LineSearchResult {
                proposed,
                h_values: h_news,
                smooth_cost: new_smooth_cost,
                merit: new_merit,
            }));
        }

        alpha *= 0.5;
        if alpha < 1e-12 {
            return Ok(None);
        }
    }
}

pub(crate) fn evaluate_cost_and_build_linear_system(
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

/// Calculate the merit value — at this point just the smooth cost.
pub(crate) fn calc_merit(
    variables: &VarFamilies,
    cost_system: &mut CostSystem,
    params: OptParams,
) -> Result<f64, NllsError> {
    let mut c = 0.0;
    for cost_fn in cost_system.cost_fns.iter() {
        c += cost_fn
            .calc_total_cost(variables, params.parallelize)
            .map_err(|e| NllsError::NllsCostSystemError { source: e })?;
    }
    Ok(c)
}

// ── Adaptive μ update (Wächter & Biegler §3.1 Eq. 16) ───────────────────────

/// Compute the barrier KKT error `E_μ` for the current barrier subproblem.
///
/// `E_μ = max(‖∇_x L‖∞, max_j |λ_j · h_j − μ|)`
///
/// The first term is the gradient norm (stationarity), which we approximate
/// from the linear system's neg_gradient. The second term is the complementarity
/// error: at the barrier subproblem optimum, `λ_j · h_j = μ` for all j.
///
/// When `E_μ < κ_ε · μ`, the current barrier subproblem is approximately solved
/// and μ should be decreased.
pub(crate) fn compute_barrier_kkt_error(
    barriers: &[alloc::boxed::Box<dyn IsIneqBarrierFn>],
    variables: &VarFamilies,
    mu: f64,
    gradient_max: f64,
) -> f64 {
    let mut max_complementarity = 0.0_f64;
    for b in barriers.iter() {
        let h_vals = b.eval_h_values(variables);
        let lambdas = b.lambdas();
        for (j, &h) in h_vals.iter().enumerate() {
            let lambda = if j < lambdas.len() { lambdas[j] } else { 1.0 };
            let complementarity = (lambda * h.max(1e-14) - mu).abs();
            max_complementarity = max_complementarity.max(complementarity);
        }
    }
    gradient_max.max(max_complementarity)
}

/// Compute the next barrier parameter μ using the superlinear update
/// (Wächter & Biegler §3.1 Eq. 16):
///
/// `μ_{k+1} = max(ε_tol/10, min(κ_μ · μ_k, μ_k^{θ_μ}))`
///
/// where `κ_μ ∈ (0,1)` controls the linear rate and `θ_μ ∈ (1,2)` controls
/// the superlinear rate. The max with `ε_tol/10` prevents μ from becoming
/// smaller than needed for the final tolerance.
pub(crate) fn compute_next_mu(mu: f64, kappa_mu: f64, theta_mu: f64, eps_tol: f64) -> f64 {
    let mu_linear = kappa_mu * mu;
    let mu_superlinear = mu.powf(theta_mu);
    (eps_tol / 10.0).max(mu_linear.min(mu_superlinear))
}

// ── Elastic feasibility helpers (Wächter & Biegler §3.3) ─────────────────────

/// Compute elastic slack `v_j` from the KKT condition `ρ = μ/s_j + μ/v_j`.
///
/// Solves `ρ v² + (ρh − 2μ) v − μh = 0` (positive root).
/// - Feasible `h > 0`: `v ≈ μ/ρ` (tiny).
/// - Infeasible `h < 0`: `v ≈ |h| + μ/ρ` (absorbs violation).
pub(crate) fn compute_elastic_slack(h: f64, mu: f64, rho: f64) -> f64 {
    let a = rho;
    let b = rho * h - 2.0 * mu;
    let c = -mu * h;
    let disc = (b * b - 4.0 * a * c).max(0.0);
    let v = (-b + disc.sqrt()) / (2.0 * a);
    v.max(1e-14)
}

/// Compute elastic slacks for all constraints in all barriers.
pub(crate) fn compute_all_elastic_slacks(
    barriers: &[alloc::boxed::Box<dyn IsIneqBarrierFn>],
    variables: &VarFamilies,
    mu: f64,
    rho: f64,
) -> alloc::vec::Vec<alloc::vec::Vec<f64>> {
    barriers
        .iter()
        .map(|b| {
            b.eval_h_values(variables)
                .iter()
                .map(|&h| compute_elastic_slack(h, mu, rho))
                .collect()
        })
        .collect()
}

/// Set elastic slacks and corresponding lambdas (`λ_j = μ / s_j`) on all barriers.
pub(crate) fn set_elastic_state(
    barriers: &[alloc::boxed::Box<dyn IsIneqBarrierFn>],
    all_slacks: &[alloc::vec::Vec<f64>],
    variables: &VarFamilies,
    mu: f64,
) {
    for (b_idx, b) in barriers.iter().enumerate() {
        let h_vals = b.eval_h_values(variables);
        let slacks = &all_slacks[b_idx];
        let lambdas: alloc::vec::Vec<f64> = h_vals
            .iter()
            .zip(slacks.iter())
            .map(|(&h, &v)| {
                let s = (h + v).max(1e-14);
                mu / s
            })
            .collect();
        b.set_elastic_slacks(slacks.clone());
        b.set_lambdas(lambdas);
    }
}

/// Compute elastic merit: `f(x) − μ Σ ln(s_j) + ρ Σ v_j − μ Σ ln(v_j)`.
/// Returns `(smooth_cost, merit)`.
pub(crate) fn calc_elastic_merit(
    variables: &VarFamilies,
    smooth: &CostSystem,
    barriers: &[alloc::boxed::Box<dyn IsIneqBarrierFn>],
    all_slacks: &[alloc::vec::Vec<f64>],
    mu: f64,
    rho: f64,
    parallelize: bool,
) -> Result<(f64, f64), CostError> {
    let sc = calc_smooth_cost(smooth, variables, parallelize)?;
    let mut m = sc;
    // Barrier cost: −λ_j ln(s_j) (computed by calc_total_cost with elastic slacks set).
    for b in barriers.iter() {
        m += b.calc_total_cost(variables, parallelize)?;
    }
    // Elastic penalty: ρ Σ v_j − μ Σ ln(v_j).
    for slacks in all_slacks.iter() {
        for &v in slacks.iter() {
            m += rho * v;
            if v > 1e-14 {
                m += -mu * v.ln();
            }
        }
    }
    Ok((sc, m))
}

// ── Infeasibility measure ────────────────────────────────────────────────────

/// Compute the infeasibility measure θ(x) = max(0, max_j(−h_j)).
///
/// Returns 0 when all constraints are satisfied.
pub(crate) fn compute_theta(
    barriers: &[alloc::boxed::Box<dyn IsIneqBarrierFn>],
    variables: &VarFamilies,
) -> f64 {
    barriers
        .iter()
        .flat_map(|b| b.eval_h_values(variables))
        .map(|h| (-h).max(0.0))
        .fold(0.0_f64, f64::max)
}

/// Elastic line search: backtracking on `s_j = h_j + v_j` with fraction-to-boundary.
///
/// At each trial alpha, recomputes elastic slacks at the proposed point.
pub(crate) fn elastic_line_search(
    opt: &Optimizer,
    delta: &nalgebra::DVector<f64>,
    s_olds: &[alloc::vec::Vec<f64>],
    tau: f64,
    mu: f64,
    rho: f64,
) -> Result<Option<LineSearchResult>, NllsError> {
    let barriers = opt.strategy.barriers();
    let parallelize = opt.params.parallelize;
    let mut alpha = 1.0_f64;
    loop {
        let scaled = delta * alpha;
        let proposed = opt.variables.update(&scaled);

        let new_slacks = compute_all_elastic_slacks(barriers, &proposed, mu, rho);
        let s_news: alloc::vec::Vec<alloc::vec::Vec<f64>> = barriers
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let h_vals = b.eval_h_values(&proposed);
                h_vals
                    .iter()
                    .zip(new_slacks[i].iter())
                    .map(|(&h, &v)| h + v)
                    .collect()
            })
            .collect();

        if fraction_to_boundary_ok(s_olds, &s_news, tau) {
            set_elastic_state(barriers, &new_slacks, &proposed, mu);
            let (new_smooth_cost, new_merit) = calc_elastic_merit(
                &proposed,
                &opt.smooth_cost_system,
                barriers,
                &new_slacks,
                mu,
                rho,
                parallelize,
            )
            .map_err(|e| NllsError::NllsCostSystemError { source: e })?;

            return Ok(Some(LineSearchResult {
                proposed,
                h_values: s_news,
                smooth_cost: new_smooth_cost,
                merit: new_merit,
            }));
        }

        alpha *= 0.5;
        if alpha < 1e-12 {
            return Ok(None);
        }
    }
}
