use core::marker::PhantomData;
use std::fmt::Debug;

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

/// Feasibility cost function for inequality constraints (phase-1).
///
/// For each violated constraint `h_j(x) < 0`, adds a quadratic penalty
/// `½ h_j(x)²` to push it toward feasibility. Feasible constraints (`h_j ≥ 0`)
/// contribute zero cost.
///
/// Uses Gauss-Newton with residual `r_j = -h_j` (when violated):
/// - Hessian: `∂h_jᵀ · ∂h_j`
/// - Neg-gradient: `h_j · ∂h_jᵀ`  (pushes h_j positive)
///
/// This is used as a phase-1 method: run `Optimizer` with only this cost
/// (no smooth/prior costs) until all constraints are satisfied, then switch
/// to the barrier optimizer for phase-2.
#[derive(Debug, Clone)]
pub struct IneqFeasibilityCostFn<
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync,
    Constraint: HasIneqConstraintFn<INPUT_DIM, N, GlobalConstants, Args>,
    Args: IsVarTuple<N> + 'static,
> {
    global_constants: GlobalConstants,
    constraints: IneqConstraints<INPUT_DIM, N, GlobalConstants, Args, Constraint>,
    phantom: PhantomData<Args>,
}

impl<
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync,
    Constraint: HasIneqConstraintFn<INPUT_DIM, N, GlobalConstants, Args> + 'static,
    Args: IsVarTuple<N> + 'static,
> IneqFeasibilityCostFn<INPUT_DIM, N, GlobalConstants, Constraint, Args>
{
    /// Create a new feasibility cost function, boxed as [`IsCostFn`].
    pub fn new_boxed(
        global_constants: GlobalConstants,
        constraints: IneqConstraints<INPUT_DIM, N, GlobalConstants, Args, Constraint>,
    ) -> alloc::boxed::Box<dyn IsCostFn> {
        alloc::boxed::Box::new(Self {
            global_constants,
            constraints,
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
> IsCostFn for IneqFeasibilityCostFn<INPUT_DIM, N, GlobalConstants, Constraint, Args>
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

        let mut evaluated_terms = EvaluatedCost::new(self.constraints.family_names.clone());

        let var_family_tuple =
            Args::ref_var_family_tuple(variables, self.constraints.family_names.clone())
                .map_err(|e| CostError::VariableFamilyError { source: e })?;

        let reduction_ranges = self.constraints.reduction_ranges.as_ref().unwrap();
        evaluated_terms.terms.reserve(reduction_ranges.len());

        for range in reduction_ranges.iter() {
            let mut evaluated_term_sum: Option<EvaluatedCostTerm<INPUT_DIM, N>> = None;

            for constraint in self.constraints.collection[range.start..range.end].iter() {
                let eval_result = constraint.eval(
                    &self.global_constants,
                    *constraint.idx_ref(),
                    Args::extract(&var_family_tuple, *constraint.idx_ref()),
                    var_kind_array,
                );

                let h = eval_result.h;

                // Only penalize violated constraints (h < 0).
                if h >= 0.0 {
                    continue;
                }

                let j_h = eval_result.jacobian.mat;

                let mut hessian = BlockHessian::<INPUT_DIM, N>::new(&dims);
                let mut gradient = BlockGradient::<INPUT_DIM, N>::new(&dims);

                // Cost: ½ h² (minimize violation when h < 0)
                // H = J_hᵀ J_h
                // neg_grad = h · J_hᵀ  (pushes h toward positive)
                let cost = 0.5 * h * h;

                if eval_mode == EvalMode::CalculateDerivatives {
                    hessian.mat = j_h.transpose() * j_h;
                    gradient.vec = h * j_h.transpose();
                }

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

            if let Some(term) = evaluated_term_sum {
                evaluated_terms.terms.push(term);
            }
        }

        Ok(alloc::boxed::Box::new(evaluated_terms))
    }

    fn sort(&mut self, variables: &VarFamilies) {
        self.constraints.sort_and_reduce(variables);
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
        let var_family_tuple =
            Args::ref_var_family_tuple(variables, self.constraints.family_names.clone())
                .map_err(|e| CostError::VariableFamilyError { source: e })?;

        let var_kind_array: [VarKind; N] = core::array::from_fn(|_| VarKind::Conditioned);

        let mut total = 0.0_f64;
        for constraint in self.constraints.collection.iter() {
            let eval_result = constraint.eval(
                &self.global_constants,
                *constraint.idx_ref(),
                Args::extract(&var_family_tuple, *constraint.idx_ref()),
                var_kind_array,
            );
            let h = eval_result.h;
            if h < 0.0 {
                total += 0.5 * h * h;
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
