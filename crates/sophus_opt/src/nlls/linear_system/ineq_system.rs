use sophus_geo::region::IsNonEmptyRegion;

use super::EvalMode;
use crate::{
    block::{
        block_sparse_matrix_builder::BlockSparseMatrixBuilder, block_vector::BlockVector,
        PartitionSpec,
    },
    nlls::{
        constraint::{
            evaluated_ineq_set::{EvaluatedIneqSet, IsEvaluatedIneqConstraintSet},
            ineq_constraint_fn::IsIneqConstraintsFn,
        },
        OptParams,
    },
    variables::var_families::VarFamilies,
};

extern crate alloc;

impl IneqSystem {
    pub(crate) fn new(
        variables: &VarFamilies,
        mut ineq_constraints_fns: Vec<alloc::boxed::Box<dyn IsIneqConstraintsFn>>,
        params: OptParams,
    ) -> Result<IneqSystem, crate::nlls::constraint::ineq_constraint_fn::IneqConstraintError> {
        for ineq_functors in ineq_constraints_fns.iter_mut() {
            // sort to achieve more efficient evaluation and reduction
            ineq_functors.sort(variables);
        }
        let mut partitions = vec![];
        for ineq_constraint in ineq_constraints_fns.iter_mut() {
            partitions.push(PartitionSpec {
                num_blocks: 1,
                block_dim: ineq_constraint.residual_dim(),
            });
        }

        let mut eq_system = IneqSystem {
            ineq_constraints_fns,
            evaluated_ineq_constraints: vec![],
            partitions,
        };
        eq_system.eval(variables, EvalMode::DontCalculateDerivatives, params)?;
        Ok(eq_system)
    }

    pub(crate) fn eval(
        &mut self,
        variables: &VarFamilies,
        eval_mode: EvalMode,
        _params: OptParams,
    ) -> Result<(), crate::nlls::constraint::ineq_constraint_fn::IneqConstraintError> {
        self.evaluated_ineq_constraints.clear();
        for eq_constraint_fn in self.ineq_constraints_fns.iter() {
            self.evaluated_ineq_constraints
                .push(eq_constraint_fn.eval(variables, eval_mode)?);
        }
        Ok(())
    }
}

/// Linear inequality system
///
/// ```ascii
/// -------------------------------------------
/// |    l - d(x)   <=   H * x  <=  u - d(x)  |
/// -------------------------------------------
/// ```
///
/// where d is the evaluated of value, H is the corresponding Jacobian, l is the lower bound, and
/// u is the upper bound of the non-linear inequality constraint.
///
/// Note, in the linearized system, the bounds are shifted. The upper bound is "u - d(x)" and the
/// lower bound is "l - d(x)".
pub struct IneqSystem {
    pub(crate) ineq_constraints_fns: Vec<alloc::boxed::Box<dyn IsIneqConstraintsFn>>,
    pub(crate) evaluated_ineq_constraints:
        alloc::vec::Vec<alloc::boxed::Box<dyn IsEvaluatedIneqConstraintSet>>,
    pub(crate) partitions: alloc::vec::Vec<PartitionSpec>,
}

impl<const RESIDUAL_DIM: usize, const INPUT_DIM: usize, const NUM_ARGS: usize>
    IsEvaluatedIneqConstraintSet for EvaluatedIneqSet<RESIDUAL_DIM, INPUT_DIM, NUM_ARGS>
{
    fn populate_linear_constraint_system(
        &self,
        variables: &VarFamilies,
        constraint_idx: usize,
        constraint_matrix: &mut BlockSparseMatrixBuilder,
        lower: &mut BlockVector,
        upper: &mut BlockVector,
    ) {
        let num_args = self.family_names.len();

        let mut scalar_start_indices_per_arg = alloc::vec::Vec::new();
        let mut block_start_indices_per_arg = alloc::vec::Vec::new();

        let mut dof_per_arg = alloc::vec::Vec::new();
        let mut arg_ids = alloc::vec::Vec::new();
        for name in self.family_names.iter() {
            let family = variables.collection.get(name).unwrap();
            scalar_start_indices_per_arg.push(family.get_scalar_start_indices().clone());
            block_start_indices_per_arg.push(family.get_block_start_indices().clone());
            dof_per_arg.push(family.free_or_marg_dof());
            arg_ids.push(variables.index(name).unwrap());
        }

        for constraint in self.evaluated_constraints.iter() {
            let idx = constraint.idx;

            // l - d
            lower.add_block(
                constraint_idx,
                0,
                &((constraint.bounds.lower() - constraint.constraint_value).as_view()),
            );
            // u - d
            upper.add_block(
                constraint_idx,
                0,
                &((constraint.bounds.upper() - constraint.constraint_value).as_view()),
            );

            for arg_id_alpha in 0..num_args {
                let dof_alpha = dof_per_arg[arg_id_alpha];

                if dof_alpha == 0 {
                    continue;
                }
                let family_alpha = arg_ids[arg_id_alpha];
                let var_idx_alpha = idx[arg_id_alpha];

                let scalar_start_idx_alpha =
                    scalar_start_indices_per_arg[arg_id_alpha][var_idx_alpha];
                let block_start_idx_alpha =
                    block_start_indices_per_arg[arg_id_alpha][var_idx_alpha];

                if scalar_start_idx_alpha == -1 {
                    continue;
                }

                let block_start_idx_alpha = block_start_idx_alpha as usize;

                // H
                constraint_matrix.add_block(
                    &[family_alpha, constraint_idx],
                    [block_start_idx_alpha, 0],
                    &constraint.jacobian.block(arg_id_alpha).as_view(),
                );
            }
        }
    }

    fn calc_le_sum_of_l1_norms(&self) -> f64 {
        // Measures violation of constraints that say: residual >= lower
        // i.e., if residual < lower, the violation is (lower - residual); else 0.
        let mut l1_norms = 0.0;
        for evaluated_constraint in self.evaluated_constraints.iter() {
            for i in 0..RESIDUAL_DIM {
                let diff = evaluated_constraint.bounds.lower()[i]
                    - evaluated_constraint.constraint_value[i];
                let violation = diff.max(0.0);
                l1_norms += violation;
            }
        }
        l1_norms
    }

    fn calc_ge_sum_of_l1_norms(&self) -> f64 {
        // Measures violation of constraints that say: residual <= upper
        // i.e., if residual > upper, the violation is (residual - upper); else 0.
        let mut l1_norms = 0.0;
        for evaluated_constraint in self.evaluated_constraints.iter() {
            for i in 0..RESIDUAL_DIM {
                let diff = evaluated_constraint.constraint_value[i]
                    - evaluated_constraint.bounds.upper()[i];
                let violation = diff.max(0.0);
                l1_norms += violation;
            }
        }
        l1_norms
    }
}
