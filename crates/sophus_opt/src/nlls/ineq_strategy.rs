//! Inequality constraint strategies for the unified `Optimizer`.
//!
//! Four variants:
//! - [`IneqStrategy::None`]: standard NLLS / equality-constrained only.
//! - [`IneqStrategy::Elastic`]: elastic feasibility phase — slacks ensure `s_j = h_j + v_j > 0`.
//! - [`IneqStrategy::Ipm`]: primal-dual IPM (Nocedal & Wright Ch. 19).
//! - [`IneqStrategy::Sqp`]: SQP with frozen Jacobians (Nocedal & Wright Ch. 18).

use sophus_solver::matrix::{
    PartitionBlockIndex,
    SymmetricMatrixBuilderEnum,
    block::BlockVector,
};

use super::cost::ineq_barrier_fn::{
    ConstraintLinearization,
    IsIneqBarrierFn,
};
use crate::variables::VarFamilies;

extern crate alloc;

/// Inequality constraint strategy for the unified optimizer.
pub enum IneqStrategy {
    /// No inequality constraints (standard NLLS / equality-constrained).
    None,
    /// Elastic feasibility: slacks `v_j >= 0` ensure `s_j = h_j + v_j > 0`.
    Elastic(ElasticState),
    /// Primal-dual IPM (Nocedal & Wright Ch. 19).
    Ipm(IpmState),
    /// SQP with frozen Jacobians (Nocedal & Wright Ch. 18).
    Sqp(SqpState),
}

/// State for elastic feasibility (replaces phase-1).
///
/// Elastic slacks `v_j >= 0` ensure `s_j = h_j + v_j > 0` even when infeasible.
/// The smooth cost is included from the start. Once all `h_j > 0`, transitions
/// to IPM or SQP.
pub struct ElasticState {
    /// Barrier cost functions with elastic slacks set.
    pub barriers: alloc::vec::Vec<alloc::boxed::Box<dyn IsIneqBarrierFn>>,
    /// Barrier parameter mu (fixed during elastic phase).
    pub mu: f64,
    /// Elastic penalty weight rho (fixed during elastic phase).
    pub rho: f64,
    /// Fraction-to-boundary safety factor.
    pub tau: f64,
}

/// State for the primal-dual IPM strategy with filter line search
/// (Wächter & Biegler, 2006).
pub struct IpmState {
    /// Barrier cost functions with per-constraint dual variables.
    pub barriers: alloc::vec::Vec<alloc::boxed::Box<dyn IsIneqBarrierFn>>,
    /// Current barrier parameter μ.
    pub mu: f64,
    /// Fraction-to-boundary safety factor.
    pub tau: f64,
    /// Filter for step acceptance (prevents cycling, ensures global convergence).
    pub filter: super::filter::Filter,
}

/// State for the SQP strategy.
pub struct SqpState {
    /// Barrier cost functions (used for h-value evaluation and Jacobian freezing).
    pub barriers: alloc::vec::Vec<alloc::boxed::Box<dyn IsIneqBarrierFn>>,
    /// Frozen constraint Jacobians at the current outer iterate.
    pub frozen_lins: alloc::vec::Vec<alloc::vec::Vec<ConstraintLinearization>>,
    /// Per-barrier, per-constraint dual variables z_j >= 0.
    pub duals: alloc::vec::Vec<alloc::vec::Vec<f64>>,
    /// Per-barrier, per-constraint slack s_j = h_j(x).
    pub slacks: alloc::vec::Vec<alloc::vec::Vec<f64>>,
    /// Barrier parameter mu.
    pub mu: f64,
    /// Fraction-to-boundary safety factor.
    pub tau: f64,
}

impl IneqStrategy {
    /// Get references to the barrier functions, if present.
    pub fn barriers(&self) -> &[alloc::boxed::Box<dyn IsIneqBarrierFn>] {
        match self {
            IneqStrategy::Ipm(ipm) => &ipm.barriers,
            IneqStrategy::Sqp(sqp) => &sqp.barriers,
            IneqStrategy::Elastic(e) => &e.barriers,
            IneqStrategy::None => &[],
        }
    }
}

// ── FrozenBarrierEvaluatedCost ────────────────────────────────────────────────

/// Per-outer-iteration frozen barrier contribution for the SQP inner LM system.
///
/// For each constraint `j` (with frozen Jacobian `J_j` at `x_k`) contributes:
/// - **Hessian**: `(zⱼ / sⱼ) · Jⱼ Jⱼᵀ`  (frozen `Jⱼ`, updated `z/s`)
/// - **neg-gradient**: `zⱼ · Jⱼ`  (KKT stationarity: `∇f = Σ zⱼ ∇hⱼ`)
#[derive(Debug, Clone)]
pub(crate) struct FrozenBarrierEvaluatedCost {
    /// Per-constraint: (frozen Jacobian linearization, dual `z_j`, slack `s_j`).
    pub entries: alloc::vec::Vec<(ConstraintLinearization, f64, f64)>,
}

impl super::IsEvaluatedCost for FrozenBarrierEvaluatedCost {
    fn calc_square_error(&self) -> f64 {
        // Merit is tracked separately by the optimizer.
        0.0
    }

    fn populate_upper_triangular_normal_equation(
        &self,
        variables: &VarFamilies,
        _nu: f64, // LM damping applied only to smooth costs
        hessian_block_triplet: &mut SymmetricMatrixBuilderEnum,
        neg_grad: &mut BlockVector,
        _parallelize: bool,
    ) {
        for (lin, z, s) in &self.entries {
            let scale = z / s.max(1e-10);
            let n = lin.family_names.len();

            // Resolve block indices and scalar offsets for each argument.
            let mut indices: alloc::vec::Vec<Option<PartitionBlockIndex>> =
                alloc::vec::Vec::with_capacity(n);
            let mut scalar_starts: alloc::vec::Vec<i64> = alloc::vec::Vec::with_capacity(n);

            for arg_id in 0..n {
                let name = &lin.family_names[arg_id];
                let entity_idx = lin.entity_indices[arg_id];
                let family = variables.collection.get(name).unwrap();
                let scalar_start = family.get_scalar_start_indices()[entity_idx];
                let block_start = family.get_block_start_indices()[entity_idx];
                scalar_starts.push(scalar_start);
                if scalar_start == -1 {
                    indices.push(None);
                } else {
                    let family_id = variables.index(name).unwrap();
                    indices.push(Some(PartitionBlockIndex {
                        partition: variables.partition_idx_by_family[family_id],
                        block: block_start as usize,
                    }));
                }
            }

            for arg_id_alpha in 0..n {
                let idx_alpha = match indices[arg_id_alpha] {
                    Some(idx) => idx,
                    None => continue,
                };
                let j_alpha = &lin.jac_blocks[arg_id_alpha];

                // Diagonal Hessian block: (z/s) * J_alpha^T J_alpha
                let hessian_diag = scale * j_alpha * j_alpha.transpose();
                hessian_block_triplet.add_lower_block(
                    idx_alpha,
                    idx_alpha,
                    &hessian_diag.as_view(),
                );

                // neg-gradient contribution: += z * J_alpha^T
                neg_grad.axpy_block(idx_alpha, &(*z * j_alpha).as_view(), 1.0);

                // Off-diagonal Hessian blocks (lower-triangular only).
                for arg_id_beta in 0..n {
                    if arg_id_alpha == arg_id_beta {
                        continue;
                    }
                    let idx_beta = match indices[arg_id_beta] {
                        Some(idx) => idx,
                        None => continue,
                    };
                    // Keep lower triangular: alpha (row) must have scalar_start >= beta (col).
                    if scalar_starts[arg_id_beta] > scalar_starts[arg_id_alpha] {
                        continue;
                    }
                    let j_beta = &lin.jac_blocks[arg_id_beta];
                    // Off-diagonal block (alpha, beta): (z/s) * J_alpha^T J_beta
                    let hessian_off = scale * j_alpha * j_beta.transpose();
                    hessian_block_triplet.add_lower_block(
                        idx_alpha,
                        idx_beta,
                        &hessian_off.as_view(),
                    );
                }
            }
        }
    }
}
