use super::EvalMode;
use crate::{
    block::{
        PartitionSpec,
        block_vector::BlockVector,
        symmetric_block_sparse_matrix_builder::SymmetricBlockSparseMatrixBuilder,
    },
    nlls::{
        OptParams,
        constraint::evaluated_eq_set::{
            EvaluatedEqSet,
            IsEvaluatedEqConstraintSet,
        },
    },
    prelude::*,
    variables::VarFamilies,
};

extern crate alloc;

impl EqSystem {
    pub(crate) fn new(
        variables: &VarFamilies,
        mut eq_constraints_fns: Vec<alloc::boxed::Box<dyn IsEqConstraintsFn>>,
        params: OptParams,
    ) -> Result<EqSystem, crate::nlls::constraint::eq_constraint_fn::EqConstraintError> {
        for eq_functors in eq_constraints_fns.iter_mut() {
            // sort to achieve more efficient evaluation and reduction
            eq_functors.sort(variables);
        }
        let mut partitions = vec![];
        for eq_constraint in eq_constraints_fns.iter_mut() {
            partitions.push(PartitionSpec {
                num_blocks: 1,
                block_dim: eq_constraint.residual_dim(),
            });
        }
        let lambda = BlockVector::zero(&partitions);

        let mut eq_system = EqSystem {
            lambda,
            eq_constraints_fns,
            evaluated_eq_constraints: vec![],
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
    ) -> Result<(), crate::nlls::constraint::eq_constraint_fn::EqConstraintError> {
        self.evaluated_eq_constraints.clear();
        for eq_constraint_fn in self.eq_constraints_fns.iter() {
            self.evaluated_eq_constraints
                .push(eq_constraint_fn.eval(variables, eval_mode)?);
        }
        Ok(())
    }

    // update lambdas
    pub(crate) fn update_lambdas(
        &self,
        variables: &VarFamilies,
        delta: &nalgebra::DVector<f64>,
    ) -> BlockVector {
        let mut updated_lambdas = self.lambda.clone();
        let lambda_num_rows = updated_lambdas.scalar_vector().shape().0;
        *updated_lambdas.scalar_vector_mut() -=
            delta.rows(variables.num_free_scalars(), lambda_num_rows);
        updated_lambdas
    }
}

/// ```ascii
/// --------------------------------------------------------------
/// |    *           G'     |  *              |   * + G'lambda   |
/// |    G           0      | -d lambda       |  -c              |
/// --------------------------------------------------------------
/// ```
///
/// where c is the residual of the equality constraints, G is the Jacobian of the equality
/// constraints, and lambda is the Lagrange multiplier.
pub struct EqSystem {
    pub(crate) eq_constraints_fns: Vec<alloc::boxed::Box<dyn IsEqConstraintsFn>>,
    pub(crate) lambda: BlockVector,
    pub(crate) evaluated_eq_constraints:
        alloc::vec::Vec<alloc::boxed::Box<dyn IsEvaluatedEqConstraintSet>>,
    pub(crate) partitions: alloc::vec::Vec<PartitionSpec>,
}

impl<const RESIDUAL_DIM: usize, const INPUT_DIM: usize, const N: usize> IsEvaluatedEqConstraintSet
    for EvaluatedEqSet<RESIDUAL_DIM, INPUT_DIM, N>
{
    fn populate_upper_triangular_kkt_mat(
        &self,
        variables: &VarFamilies,
        lambda: &BlockVector,
        constraint_idx: usize,
        block_triplet: &mut SymmetricBlockSparseMatrixBuilder,
        block_vec: &mut BlockVector,
    ) {
        let num_args = self.family_names.len();

        let num_families = variables.collection.len();
        let region_idx = num_families + constraint_idx;

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

            let lambda = lambda.get_block(constraint_idx, 0);

            // -c
            block_vec.add_block(region_idx, 0, &((-constraint.residual).as_view()));

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

                let mat_g_times_lambda =
                    constraint.jacobian.block(arg_id_alpha).transpose() * lambda;

                // + G'lambda
                block_vec.add_block(
                    family_alpha,
                    block_start_idx_alpha,
                    &(mat_g_times_lambda.as_view()),
                );

                // G'
                block_triplet.add_block(
                    &[family_alpha, region_idx],
                    [block_start_idx_alpha, 0],
                    &constraint
                        .jacobian
                        .block(arg_id_alpha)
                        .transpose()
                        .as_view(),
                );

                // no need to set G since we are only setting the upper triangular system
            }
        }
    }

    fn calc_sum_of_l1_norms(&self) -> f64 {
        let mut l1_norms = 0.0;
        for evaluated_constraint in self.evaluated_constraints.iter() {
            for residual in evaluated_constraint.residual.iter() {
                l1_norms += residual.abs();
            }
        }
        l1_norms
    }
}
