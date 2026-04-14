use core::{
    fmt,
    marker::PhantomData,
};
use std::sync::{
    Arc,
    Mutex,
};

use sophus_solver::matrix::block_sparse::BlockSparseSymmetricSymbolicBuilder;

use super::evaluated_cost::IsEvaluatedCost;
use crate::{
    block::{
        BlockGradient,
        BlockHessian,
    },
    nlls::{
        IsCostFn,
        constraint::ineq_constraint::{
            HasIneqConstraintFn,
            IneqConstraints,
        },
        cost::{
            cost_fn::CostError,
            evaluated_cost::EvaluatedCost,
            evaluated_term::EvaluatedCostTerm,
        },
        linear_system::EvalMode,
    },
    robust_kernel::RobustKernel,
    variables::{
        IsVarTuple,
        VarFamilies,
        VarKind,
    },
};

extern crate alloc;

/// Pre-evaluated constraint linearization at a fixed point `x_k`.
///
/// Stores `h_j(x_k)` and the dense Jacobian blocks `∂h_j/∂x_arg` for each argument.
/// Used by the SQP inner LM loop where constraint Jacobians are frozen at `x_k`.
#[derive(Debug, Clone)]
pub struct ConstraintLinearization {
    /// Constraint value `h_j(x_k) ≥ 0`.
    pub h: f64,
    /// Variable family names, one per argument.
    pub family_names: alloc::vec::Vec<String>,
    /// Variable indices within each family, one per argument.
    pub entity_indices: alloc::vec::Vec<usize>,
    /// Dense Jacobian blocks: `∂h_j/∂x_arg[k]` as a row vector (length = DOF of that arg).
    /// One entry per argument, in the same order as `family_names`.
    pub jac_blocks: alloc::vec::Vec<nalgebra::DVector<f64>>,
}

/// Object-safe trait for inequality barrier cost functions (used by both IPM and SQP strategies).
///
/// Extends [`IsCostFn`] with the extra methods the [`crate::nlls::Optimizer`] IPM and SQP
/// strategies need: evaluating constraint values `h_j`, reading, and writing dual variables `λ_j`.
///
/// Implementations use interior mutability so that the lambda updates can go through
/// `&self` (required because `Box<dyn IsIneqBarrierFn>` is held by the optimizer while
/// the optimizer itself calls `step(&mut self)`).
pub trait IsIneqBarrierFn: IsCostFn + fmt::Debug {
    /// Evaluate the scalar constraint value `h_j(x) ≥ 0` for every constraint in the
    /// sorted collection.  Returns one entry per entry in `constraints.collection`
    /// (collection order matches the lambda vector after `sort()` has been called).
    ///
    /// All variables are treated as conditioned — no Jacobians are computed.
    fn eval_h_values(&self, variables: &VarFamilies) -> alloc::vec::Vec<f64>;

    /// Return a snapshot of the current `λ_j` values (one per constraint in collection order).
    fn lambdas(&self) -> alloc::vec::Vec<f64>;

    /// Replace the full lambda vector.  Must have the same length as `num_lambdas()`.
    fn set_lambdas(&self, lambdas: alloc::vec::Vec<f64>);

    /// Number of lambdas (= `constraints.collection.len()`, fixed after `sort()`).
    fn num_lambdas(&self) -> usize;

    /// Set elastic slacks `v_j >= 0` for elastic feasibility mode.
    /// When non-empty, the barrier uses `s_j = h_j + v_j` instead of `h_j`.
    fn set_elastic_slacks(&self, slacks: alloc::vec::Vec<f64>);

    /// Return current elastic slacks (empty if not in elastic mode).
    fn elastic_slacks(&self) -> alloc::vec::Vec<f64>;

    /// Evaluate and freeze constraint linearizations at the current `variables`.
    ///
    /// Returns one [`ConstraintLinearization`] per constraint in collection order.
    /// The Jacobian blocks are evaluated at `variables` and stored for use by the
    /// SQP inner LM loop (where constraint Jacobians must remain fixed).
    fn eval_constraint_linearizations(
        &self,
        variables: &VarFamilies,
    ) -> alloc::vec::Vec<ConstraintLinearization>;
}

/// Inequality barrier cost wrapping a set of inequality constraints (used by both IPM and SQP).
///
/// For each constraint `h_j(x) ≥ 0` this contributes:
///
/// - **Hessian**: `(λⱼ / hⱼ) · ∇hⱼᵀ · ∇hⱼ`
/// - **neg-gradient**: `λⱼ · ∇hⱼᵀ`  (stored negated; the normal equation assembler subtracts)
/// - **Cost (merit)**: `−λⱼ · ln(hⱼ)`
///
/// For IPM, the optimizer updates `λ_j` multiplicatively after each accepted step:
/// `λ_j ← λ_j · h_j_old / h_j_new`.
/// For SQP, the lambdas correspond to the frozen dual variables `z_j`; the SQP strategy
/// uses `eval_constraint_linearizations` rather than `eval` directly for its inner loop.
///
/// The lambda and elastic-slack vectors use `Arc<Mutex<…>>` for interior mutability:
/// the `IsIneqBarrierFn` trait requires `&self` on mutating methods (`set_lambdas`,
/// `set_elastic_slacks`) because the trait must be object-safe and `Send + Sync`,
/// and barriers are borrowed immutably through `&[Box<dyn IsIneqBarrierFn>]` while
/// the optimizer simultaneously holds `&mut self` for other fields.
#[derive(Clone)]
pub struct IneqBarrierCostFn<
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync,
    Constraint: HasIneqConstraintFn<INPUT_DIM, N, GlobalConstants, Args>,
    Args: IsVarTuple<N> + 'static,
> {
    global_constants: GlobalConstants,
    constraints: IneqConstraints<INPUT_DIM, N, GlobalConstants, Args, Constraint>,
    /// Per-constraint dual variables `λ_j`.
    ///
    /// Indexed by position in `constraints.collection` (which is sorted after `sort()`).
    /// Empty until the optimizer calls `set_lambdas` after `sort()`.
    pub lambdas: Arc<Mutex<alloc::vec::Vec<f64>>>,
    /// Per-constraint elastic slacks `v_j >= 0` (elastic mode only). Empty otherwise.
    elastic_slacks: Arc<Mutex<alloc::vec::Vec<f64>>>,
    phantom: PhantomData<Args>,
}

impl<
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync,
    Constraint: HasIneqConstraintFn<INPUT_DIM, N, GlobalConstants, Args>,
    Args: IsVarTuple<N> + 'static,
> fmt::Debug for IneqBarrierCostFn<INPUT_DIM, N, GlobalConstants, Constraint, Args>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IneqBarrierCostFn")
            .field("num_constraints", &self.constraints.collection.len())
            .field("num_lambdas", &self.lambdas.lock().unwrap().len())
            .finish()
    }
}

impl<
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync + Clone,
    Constraint: HasIneqConstraintFn<INPUT_DIM, N, GlobalConstants, Args> + Clone + 'static,
    Args: IsVarTuple<N> + Clone + 'static,
> IneqBarrierCostFn<INPUT_DIM, N, GlobalConstants, Constraint, Args>
{
    /// Create a new IPM barrier cost function, boxed as `Box<dyn IsIneqBarrierFn>`.
    ///
    /// Lambdas are initialised to an empty vector; call `set_lambdas` (or rely on
    /// `Optimizer::new` to initialise them) after the first `sort()`.
    pub fn new_boxed(
        global_constants: GlobalConstants,
        constraints: IneqConstraints<INPUT_DIM, N, GlobalConstants, Args, Constraint>,
    ) -> alloc::boxed::Box<dyn IsIneqBarrierFn> {
        alloc::boxed::Box::new(Self {
            global_constants,
            constraints,
            lambdas: Arc::new(Mutex::new(alloc::vec![])),
            elastic_slacks: Arc::new(Mutex::new(alloc::vec![])),
            phantom: PhantomData,
        })
    }
}

impl<
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync,
    Constraint: HasIneqConstraintFn<INPUT_DIM, N, GlobalConstants, Args> + 'static,
    Args: IsVarTuple<N> + 'static,
> IsCostFn for IneqBarrierCostFn<INPUT_DIM, N, GlobalConstants, Constraint, Args>
{
    fn eval(
        &self,
        variables: &VarFamilies,
        eval_mode: EvalMode,
        _parallelize: bool,
    ) -> Result<alloc::boxed::Box<dyn IsEvaluatedCost>, CostError> {
        let mut var_kind_array =
            Args::var_kind_array(variables, self.constraints.family_names.clone());

        if eval_mode == EvalMode::DontCalculateDerivatives {
            var_kind_array = var_kind_array.map(|_| VarKind::Conditioned);
        }

        let dims: [usize; N] = core::array::from_fn(|i| {
            variables
                .collection
                .get(&self.constraints.family_names[i])
                .unwrap()
                .free_or_marg_dof()
        });

        let lambdas = self.lambdas.lock().unwrap();
        let elastic = self.elastic_slacks.lock().unwrap();
        let has_elastic = !elastic.is_empty();

        let var_family_tuple =
            Args::ref_var_family_tuple(variables, self.constraints.family_names.clone())
                .map_err(|e| CostError::VariableFamilyError { source: e })?;

        let reduction_ranges = self.constraints.reduction_ranges.as_ref().unwrap();
        let mut evaluated_terms = EvaluatedCost::new(self.constraints.family_names.clone());
        evaluated_terms.terms.reserve(reduction_ranges.len());

        const EPSILON: f64 = 1e-6;

        for range in reduction_ranges.iter() {
            let mut evaluated_term_sum: Option<EvaluatedCostTerm<INPUT_DIM, N>> = None;

            for j in range.start..range.end {
                let constraint = &self.constraints.collection[j];
                let eval_result = constraint.eval(
                    &self.global_constants,
                    *constraint.idx_ref(),
                    Args::extract(&var_family_tuple, *constraint.idx_ref()),
                    var_kind_array,
                );

                let h = eval_result.h;
                let j_h = eval_result.jacobian.mat;
                // Fall back to λ=1 when lambdas haven't been initialised yet.
                let lambda = if j < lambdas.len() {
                    lambdas[j]
                } else {
                    1.0_f64
                };

                // In elastic mode, use s_j = h_j + v_j instead of h_j.
                let s = if has_elastic && j < elastic.len() {
                    h + elastic[j]
                } else {
                    h
                };

                let mut hessian = BlockHessian::<INPUT_DIM, N>::new(&dims);
                let mut gradient = BlockGradient::<INPUT_DIM, N>::new(&dims);
                let cost;

                if s > EPSILON {
                    // Interior feasible point — standard primal-dual update.
                    //   merit term: -λ · ln(s)
                    //   H_b  = (λ/s) · J_h^T · J_h   (∂s/∂x = ∂h/∂x)
                    //   neg-gradient contribution = +λ · J_h^T
                    //     → stored as gradient.vec = -λ · J_h^T
                    //       (populate_normal_equation does: neg_grad -= gradient, so
                    //        neg_grad += λ · J_h^T — correct KKT stationarity direction)
                    let scale_h = lambda / s;
                    if eval_mode == EvalMode::CalculateDerivatives {
                        hessian.mat = scale_h * j_h.transpose() * j_h;
                        gradient.vec = -lambda * j_h.transpose();
                    }
                    cost = -lambda * s.ln();
                } else {
                    // Near boundary or infeasible — C1-continuous quadratic extension.
                    //
                    // Matches the log barrier at s = ε in both value and first derivative:
                    //   cost = λ/(2ε²)·(s−ε)² − (λ/ε)·(s−ε) − λ·ln(ε)
                    //   ∂cost/∂s = λ·(s−ε)/ε² − λ/ε   →  at s=ε: −λ/ε  (matches −λ/s)
                    let violation = s - EPSILON;
                    let scale_h = lambda / (EPSILON * EPSILON);
                    let scale_g = scale_h * violation - lambda / EPSILON;
                    if eval_mode == EvalMode::CalculateDerivatives {
                        hessian.mat = scale_h * j_h.transpose() * j_h;
                        gradient.vec = scale_g * j_h.transpose();
                    }
                    cost = lambda * violation * violation / (2.0 * EPSILON * EPSILON)
                        - lambda * violation / EPSILON
                        - lambda * EPSILON.ln();
                };

                let term = EvaluatedCostTerm {
                    hessian,
                    gradient,
                    cost,
                    idx: eval_result.idx,
                    num_sub_terms: 1,
                };

                match evaluated_term_sum {
                    Some(mut sum) => {
                        sum.reduce(term);
                        evaluated_term_sum = Some(sum);
                    }
                    None => evaluated_term_sum = Some(term),
                }
            }

            evaluated_terms
                .terms
                .push(evaluated_term_sum.expect("reduction range must have ≥1 constraint"));
        }

        Ok(alloc::boxed::Box::new(evaluated_terms))
    }

    fn sort(&mut self, variables: &VarFamilies) {
        self.constraints.sort_and_reduce(variables);
        // Clear lambdas — they must be re-initialised after sort because the constraint
        // ordering has changed.  Optimizer::new() will call set_lambdas() next.
        *self.lambdas.lock().unwrap() = alloc::vec![];
        *self.elastic_slacks.lock().unwrap() = alloc::vec![];
    }

    fn cost_family_names(&self) -> alloc::vec::Vec<String> {
        self.constraints.family_names.to_vec()
    }

    fn robust_kernel(&self) -> Option<RobustKernel> {
        None
    }

    fn calc_total_cost(
        &self,
        variables: &VarFamilies,
        _parallelize: bool,
    ) -> Result<f64, CostError> {
        let lambdas = self.lambdas.lock().unwrap();
        let elastic = self.elastic_slacks.lock().unwrap();
        let has_elastic = !elastic.is_empty();
        let var_family_tuple =
            Args::ref_var_family_tuple(variables, self.constraints.family_names.clone())
                .map_err(|e| CostError::VariableFamilyError { source: e })?;
        let var_kind_array: [VarKind; N] = core::array::from_fn(|_| VarKind::Conditioned);
        const EPSILON: f64 = 1e-6;
        let mut total = 0.0_f64;
        for (j, constraint) in self.constraints.collection.iter().enumerate() {
            let eval_result = constraint.eval(
                &self.global_constants,
                *constraint.idx_ref(),
                Args::extract(&var_family_tuple, *constraint.idx_ref()),
                var_kind_array,
            );
            let h = eval_result.h;
            let lambda = if j < lambdas.len() {
                lambdas[j]
            } else {
                1.0_f64
            };
            let s = if has_elastic && j < elastic.len() {
                h + elastic[j]
            } else {
                h
            };
            if s > EPSILON {
                total += -lambda * s.ln();
            } else {
                let violation = s - EPSILON;
                total += lambda * violation * violation / (2.0 * EPSILON * EPSILON)
                    - lambda * violation / EPSILON
                    - lambda * EPSILON.ln();
            }
        }
        Ok(total)
    }

    fn populate_symbolic(
        &self,
        variables: &VarFamilies,
        sym_builder: &mut BlockSparseSymmetricSymbolicBuilder,
    ) {
        self.constraints.populate_symbolic(variables, sym_builder);
    }
}

impl<
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync + Clone,
    Constraint: HasIneqConstraintFn<INPUT_DIM, N, GlobalConstants, Args> + Clone + 'static,
    Args: IsVarTuple<N> + Clone + 'static,
> IsIneqBarrierFn for IneqBarrierCostFn<INPUT_DIM, N, GlobalConstants, Constraint, Args>
{
    fn eval_h_values(&self, variables: &VarFamilies) -> alloc::vec::Vec<f64> {
        let Ok(var_family_tuple) =
            Args::ref_var_family_tuple(variables, self.constraints.family_names.clone())
        else {
            return alloc::vec![];
        };
        let var_kind_array: [VarKind; N] = core::array::from_fn(|_| VarKind::Conditioned);
        self.constraints
            .collection
            .iter()
            .map(|constraint| {
                constraint
                    .eval(
                        &self.global_constants,
                        *constraint.idx_ref(),
                        Args::extract(&var_family_tuple, *constraint.idx_ref()),
                        var_kind_array,
                    )
                    .h
            })
            .collect()
    }

    fn lambdas(&self) -> alloc::vec::Vec<f64> {
        self.lambdas.lock().unwrap().clone()
    }

    fn set_lambdas(&self, lambdas: alloc::vec::Vec<f64>) {
        *self.lambdas.lock().unwrap() = lambdas;
    }

    fn num_lambdas(&self) -> usize {
        self.constraints.collection.len()
    }

    fn set_elastic_slacks(&self, slacks: alloc::vec::Vec<f64>) {
        *self.elastic_slacks.lock().unwrap() = slacks;
    }

    fn elastic_slacks(&self) -> alloc::vec::Vec<f64> {
        self.elastic_slacks.lock().unwrap().clone()
    }

    fn eval_constraint_linearizations(
        &self,
        variables: &VarFamilies,
    ) -> alloc::vec::Vec<ConstraintLinearization> {
        let Ok(var_family_tuple) =
            Args::ref_var_family_tuple(variables, self.constraints.family_names.clone())
        else {
            return alloc::vec![];
        };
        // Evaluate with full derivatives so we get the Jacobian.
        let var_kind_array = Args::var_kind_array(variables, self.constraints.family_names.clone());

        self.constraints
            .collection
            .iter()
            .map(|constraint| {
                let eval = constraint.eval(
                    &self.global_constants,
                    *constraint.idx_ref(),
                    Args::extract(&var_family_tuple, *constraint.idx_ref()),
                    var_kind_array,
                );

                let h = eval.h;
                let jac = &eval.jacobian; // BlockJacobian<1, INPUT_DIM, N>

                // Extract per-argument Jacobian blocks as dense DVector rows.
                let n = self.constraints.family_names.len();
                let mut jac_blocks = alloc::vec::Vec::with_capacity(n);
                let mut dofs = alloc::vec::Vec::with_capacity(n);
                for arg_id in 0..n {
                    let dof = variables
                        .collection
                        .get(&self.constraints.family_names[arg_id])
                        .unwrap()
                        .free_or_marg_dof();
                    dofs.push(dof);
                }

                // jac.mat is SMatrix<f64, 1, INPUT_DIM>: a single row.
                // The block layout is: arg 0 occupies cols 0..dof0, arg 1 occupies dof0..dof0+dof1,
                // etc.
                let mat = jac.mat;
                let mut col_offset = 0;
                for arg_id in 0..n {
                    let dof = dofs[arg_id];
                    // Extract the dof columns for this argument as a DVector.
                    let block: nalgebra::DVector<f64> = nalgebra::DVector::from_iterator(
                        dof,
                        (col_offset..col_offset + dof).map(|c| mat[(0, c)]),
                    );
                    jac_blocks.push(block);
                    col_offset += dof;
                }

                ConstraintLinearization {
                    h,
                    family_names: self.constraints.family_names.to_vec(),
                    entity_indices: constraint.idx_ref().to_vec(),
                    jac_blocks,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use sophus_autodiff::linalg::VecF64;

    use crate::{
        nlls::{
            IneqBarrierCostFn,
            IneqConstraints,
            ineq_constraints::ScalarHalfPlaneConstraint,
            linear_system::EvalMode,
        },
        variables::{
            VarBuilder,
            VarFamily,
            VarKind,
        },
    };

    const POINT: &str = "point";

    /// Build a single half-plane barrier (h = -x + offset) at a given point.
    fn build_barrier_and_vars(
        x: f64,
        offset: f64,
        lambda: f64,
    ) -> (
        Box<dyn super::IsIneqBarrierFn>,
        crate::variables::VarFamilies,
    ) {
        let constraints = vec![ScalarHalfPlaneConstraint {
            normal: VecF64::<2>::new(-1.0, 0.0),
            offset,
            entity_indices: [0],
        }];
        let mut barrier =
            IneqBarrierCostFn::new_boxed((), IneqConstraints::new([POINT], constraints));
        let vars = VarBuilder::new()
            .add_family(
                POINT,
                VarFamily::new(VarKind::Free, vec![VecF64::<2>::new(x, 0.0)]),
            )
            .build();
        barrier.sort(&vars);
        barrier.set_lambdas(vec![lambda]);
        (barrier, vars)
    }

    /// Verify that `eval` and `calc_total_cost` return the same cost
    /// for a range of s values spanning the interior, boundary, and infeasible regions.
    #[test]
    fn eval_and_calc_total_cost_agree() {
        let lambda = 2.5;
        // h(p) = -x + 5, so s = -x + 5 = 5 - x.
        // Test at x values yielding s = {4.0, 1.0, 1e-6 + 1e-8, 1e-6, 0.5e-6, -1.0}.
        let test_cases: Vec<(f64, &str)> = vec![
            (1.0, "deep interior s=4.0"),
            (4.0, "near boundary s=1.0"),
            (5.0 - 1e-6 - 1e-8, "just above EPSILON"),
            (5.0 - 1e-6, "exactly at EPSILON"),
            (5.0 - 0.5e-6, "just below EPSILON"),
            (6.0, "infeasible s=-1.0"),
        ];

        for (x, label) in test_cases {
            let (barrier, vars) = build_barrier_and_vars(x, 5.0, lambda);

            let eval_result = barrier.eval(&vars, EvalMode::DontCalculateDerivatives, false);
            let eval_cost = eval_result.unwrap().calc_square_error();

            let calc_cost = barrier.calc_total_cost(&vars, false).unwrap();

            assert!(
                (eval_cost - calc_cost).abs() < 1e-12,
                "eval vs calc_total_cost mismatch at {label}: eval={eval_cost:.15e}, calc={calc_cost:.15e}, diff={:.2e}",
                (eval_cost - calc_cost).abs()
            );
        }
    }

    /// Verify C1 continuity of the barrier cost at the EPSILON boundary.
    ///
    /// The log barrier `-λ ln(s)` and its quadratic extension must match
    /// in both value and first derivative at s = EPSILON.
    #[test]
    fn barrier_c1_continuous_at_epsilon() {
        let lambda = 3.0;
        let epsilon = 1e-6_f64;
        // h(p) = -x + offset, so s = offset - x. Set offset = 5.0, so x = 5.0 - s.
        let offset = 5.0;

        // Evaluate directly at s = ε via both formulas to check C0 continuity.
        // The log-barrier value at s = ε: -λ ln(ε)
        let log_value = -lambda * epsilon.ln();
        // The quadratic extension at s = ε: violation = 0, so
        //   λ·0²/(2ε²) - λ·0/ε - λ·ln(ε) = -λ·ln(ε)
        let quad_value = lambda * 0.0 / (2.0 * epsilon * epsilon)
            - lambda * 0.0 / epsilon
            - lambda * epsilon.ln();
        assert!(
            (log_value - quad_value).abs() < 1e-12,
            "C0: value mismatch at s=ε: log={log_value:.10e}, quad={quad_value:.10e}"
        );

        // Derivative continuity: d/ds(-λ ln(s)) = -λ/s.  At s = ε: -λ/ε.
        // d/ds(quad extension) = λ(s-ε)/ε² - λ/ε.  At s = ε: -λ/ε.  Same.
        // Check numerically via finite differences on calc_total_cost.
        let h = 1e-9;
        // Gradient from the interior side (s slightly above ε)
        let (b1, v1) = build_barrier_and_vars(offset - epsilon - 2.0 * h, offset, lambda);
        let (b2, v2) = build_barrier_and_vars(offset - epsilon - h, offset, lambda);
        let c1 = b1.calc_total_cost(&v1, false).unwrap();
        let c2 = b2.calc_total_cost(&v2, false).unwrap();
        // dcost/dx; since s = offset - x, dcost/ds = -dcost/dx
        let grad_interior = (c1 - c2) / h;

        // Gradient from the extension side (s slightly below ε)
        let (b3, v3) = build_barrier_and_vars(offset - epsilon + h, offset, lambda);
        let (b4, v4) = build_barrier_and_vars(offset - epsilon + 2.0 * h, offset, lambda);
        let c3 = b3.calc_total_cost(&v3, false).unwrap();
        let c4 = b4.calc_total_cost(&v4, false).unwrap();
        let grad_extension = (c3 - c4) / h;

        let grad_diff = (grad_interior - grad_extension).abs();
        let grad_scale = grad_interior.abs().max(grad_extension.abs()).max(1.0);
        assert!(
            grad_diff / grad_scale < 1e-2,
            "C1 discontinuity at EPSILON: gradients differ by {grad_diff:.2e} \
             (interior={grad_interior:.6e}, extension={grad_extension:.6e})"
        );
    }
}
