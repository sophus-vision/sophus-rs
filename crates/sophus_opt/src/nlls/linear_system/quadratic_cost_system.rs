use super::EvalMode;
use crate::{
    block::{
        block_vector::BlockVector,
        symmetric_block_sparse_matrix_builder::SymmetricBlockSparseMatrixBuilder,
    },
    nlls::{
        quadratic_cost::evaluated_cost::EvaluatedCost,
        OptParams,
    },
    prelude::*,
    variables::var_families::VarFamilies,
};

extern crate alloc;

impl CostSystem {
    pub(crate) fn new(
        variables: &VarFamilies,
        mut cost_fns: Vec<alloc::boxed::Box<dyn IsCostFn>>,
        params: OptParams,
    ) -> Self {
        for cost_functors in cost_fns.iter_mut() {
            // sort to achieve more efficient evaluation and reduction
            cost_functors.sort(variables);
        }
        let mut cost_system = CostSystem {
            lm_damping: params.initial_lm_damping,
            cost_fns,
            evaluated_costs: vec![],
        };
        cost_system.eval(variables, EvalMode::DontCalculateDerivatives, params);
        cost_system
    }

    pub(crate) fn eval(&mut self, variables: &VarFamilies, eval_mode: EvalMode, params: OptParams) {
        self.evaluated_costs.clear();
        for cost_functors in self.cost_fns.iter() {
            self.evaluated_costs
                .push(cost_functors.eval(variables, eval_mode, params.parallelize));
        }
    }
}

/// ```ascii
/// --------------------------------------------------------------
/// | J'J + nuI      *      | dx              |  -J'r + *        |
/// |     *          *      | *               |   *              |
/// --------------------------------------------------------------
/// ```
///
/// where r is the residual, J is the Jacobian, dx is the incremental update for the
/// variables, and nu is the Levenberg-Marquardt damping parameter.
pub struct CostSystem {
    pub(crate) lm_damping: f64,
    pub(crate) cost_fns: Vec<alloc::boxed::Box<dyn IsCostFn>>,
    pub(crate) evaluated_costs: alloc::vec::Vec<alloc::boxed::Box<dyn IsEvaluatedCost>>,
}

impl<const NUM: usize, const NUM_ARGS: usize> IsEvaluatedCost for EvaluatedCost<NUM, NUM_ARGS> {
    fn populate_upper_triangulatr_normal_equation(
        &self,
        variables: &VarFamilies,
        nu: f64,
        hessian_block_triplet: &mut SymmetricBlockSparseMatrixBuilder,
        neg_grad: &mut BlockVector,
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

        for evaluated_term in self.terms.iter() {
            let idx = evaluated_term.idx;
            assert_eq!(idx.len(), num_args);

            for arg_id_alpha in 0..num_args {
                let dof_alpha = dof_per_arg[arg_id_alpha];
                let family_alpha = arg_ids[arg_id_alpha];
                if dof_alpha == 0 {
                    continue;
                }
                let var_idx_alpha = idx[arg_id_alpha];
                let scalar_start_idx_alpha =
                    scalar_start_indices_per_arg[arg_id_alpha][var_idx_alpha];
                let block_start_idx_alpha =
                    block_start_indices_per_arg[arg_id_alpha][var_idx_alpha];
                if scalar_start_idx_alpha == -1 {
                    continue;
                }
                let scalar_start_idx_alpha = scalar_start_idx_alpha as usize;

                let grad_block = evaluated_term.gradient.block(arg_id_alpha);
                let block_start_idx_alpha = block_start_idx_alpha as usize;
                assert_eq!(dof_alpha, grad_block.nrows());

                // -J'r
                neg_grad.add_block(
                    family_alpha,
                    block_start_idx_alpha,
                    &(-grad_block).as_view(),
                );
                let hessian_block = evaluated_term.hessian.block(arg_id_alpha, arg_id_alpha);
                assert_eq!(dof_alpha, hessian_block.nrows());
                assert_eq!(dof_alpha, hessian_block.ncols());

                // block diagonal
                // J'J + nuI
                hessian_block_triplet.add_block(
                    &[family_alpha, family_alpha],
                    [block_start_idx_alpha, block_start_idx_alpha],
                    &(hessian_block + nu * nalgebra::DMatrix::identity(dof_alpha, dof_alpha))
                        .as_view(),
                );

                // off diagonal hessian
                for arg_id_beta in 0..num_args {
                    let family_beta = arg_ids[arg_id_beta];

                    // skip diagonal blocks
                    if arg_id_alpha == arg_id_beta {
                        continue;
                    }
                    let dof_beta = dof_per_arg[arg_id_beta];
                    if dof_beta == 0 {
                        continue;
                    }

                    let var_idx_beta = idx[arg_id_beta];
                    let scalar_start_idx_beta =
                        scalar_start_indices_per_arg[arg_id_beta][var_idx_beta];
                    if scalar_start_idx_beta == -1 {
                        continue;
                    }
                    let scalar_start_idx_beta = scalar_start_idx_beta as usize;
                    if scalar_start_idx_beta < scalar_start_idx_alpha {
                        // upper triangular only, hence skip lower triangular
                        continue;
                    }
                    let block_start_idx_beta =
                        block_start_indices_per_arg[arg_id_beta][var_idx_beta] as usize;

                    let hessian_block_alpha_beta =
                        evaluated_term.hessian.block(arg_id_alpha, arg_id_beta);

                    // J'J
                    hessian_block_triplet.add_block(
                        &[family_alpha, family_beta],
                        [block_start_idx_alpha, block_start_idx_beta],
                        &hessian_block_alpha_beta.as_view(),
                    );
                }
            }
        }
    }

    fn calc_square_error(&self) -> f64 {
        let mut error = 0.0;
        for term in self.terms.iter() {
            error += term.cost;
        }
        error
    }
}
