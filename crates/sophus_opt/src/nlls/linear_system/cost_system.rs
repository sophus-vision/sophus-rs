use sophus_solver::matrix::{
    PartitionBlockIndex,
    SymmetricMatrixBuilderEnum,
    block::BlockVector,
    block_sparse::BlockSparseSymmetricMatrixPattern,
};

use super::EvalMode;
use crate::{
    nlls::{
        CostError,
        EvaluatedCost,
        OptParams,
    },
    prelude::*,
    variables::VarFamilies,
};

extern crate alloc;

impl CostSystem {
    pub(crate) fn new(
        variables: &VarFamilies,
        mut cost_fns: Vec<alloc::boxed::Box<dyn IsCostFn>>,
        params: OptParams,
    ) -> Result<Self, CostError> {
        for cost_functors in cost_fns.iter_mut() {
            // sort to achieve more efficient evaluation and reduction
            cost_functors.sort(variables);
        }
        let mut cost_system = CostSystem {
            lm_damping: params.initial_lm_damping,
            cost_fns,
            evaluated_costs: vec![],
        };
        cost_system.eval(variables, EvalMode::DontCalculateDerivatives, params)?;
        Ok(cost_system)
    }

    pub(crate) fn eval(
        &mut self,
        variables: &VarFamilies,
        eval_mode: EvalMode,
        params: OptParams,
    ) -> Result<(), CostError> {
        self.evaluated_costs.clear();
        for cost_functors in self.cost_fns.iter() {
            self.evaluated_costs.push(cost_functors.eval(
                variables,
                eval_mode,
                params.parallelize,
            )?);
        }
        Ok(())
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

impl<const INPUT_DIM: usize, const N: usize> IsEvaluatedCost for EvaluatedCost<INPUT_DIM, N> {
    fn populate_upper_triangular_normal_equation(
        &self,
        variables: &VarFamilies,
        nu: f64,
        hessian_block_triplet: &mut SymmetricMatrixBuilderEnum,
        neg_grad: &mut BlockVector,
        parallelize: bool,
    ) {
        let num_args = self.family_names.len();
        let mut scalar_start_indices_per_arg = alloc::vec::Vec::new();
        let mut block_start_indices_per_arg = alloc::vec::Vec::new();
        let mut dof_per_arg = alloc::vec::Vec::new();
        let mut arg_ids = alloc::vec::Vec::new();
        for name in self.family_names.iter() {
            let family = variables
                .collection
                .get(name)
                .unwrap_or_else(|| panic!("cost family '{name}' not in variables"));
            scalar_start_indices_per_arg.push(family.get_scalar_start_indices().clone());
            block_start_indices_per_arg.push(family.get_block_start_indices().clone());
            dof_per_arg.push(family.free_or_marg_dof());
            arg_ids.push(
                variables
                    .index(name)
                    .unwrap_or_else(|| panic!("cost family '{name}' not in variables")),
            );
        }

        // Pre-compute nu * I_d for each unique DOF.
        let mut nu_id_cache: alloc::vec::Vec<Option<nalgebra::DMatrix<f64>>> =
            alloc::vec![None; 65];
        if nu != 0.0 {
            for &dof in &dof_per_arg {
                if dof < nu_id_cache.len() && nu_id_cache[dof].is_none() {
                    nu_id_cache[dof] = Some(nalgebra::DMatrix::<f64>::identity(dof, dof) * nu);
                }
            }
        }

        // ── Parallel path: BlockSparsePattern only ──────────────────────────
        #[cfg(not(target_arch = "wasm32"))]
        if parallelize
            && let SymmetricMatrixBuilderEnum::BlockSparsePattern(main_pat, _) =
                hessian_block_triplet
        {
            use rayon::prelude::*;

            let n_grad = neg_grad.scalar_vector().len();
            // Pre-extract partition mapping — avoids capturing VarFamilies (non-Sync).
            let partition_idx_by_family = variables.partition_idx_by_family.clone();

            // Cap at 4 chunks to bound clone/merge memory traffic.
            let max_chunks = rayon::current_num_threads().clamp(1, 4);
            let chunk_size = self.terms.len().div_ceil(max_chunks);
            let chunk_size = chunk_size.max(1);

            // Split terms into chunks (borrows self.terms).
            let term_chunks: alloc::vec::Vec<&[crate::nlls::EvaluatedCostTerm<INPUT_DIM, N>]> =
                self.terms.chunks(chunk_size).collect();
            let n_chunks = term_chunks.len();

            // Ensure pre-allocated workers exist and are zeroed.
            // First iteration: clones the pattern structure (one-time allocation).
            // Subsequent iterations: only zeroes the storage (memset, no allocation).
            main_pat.ensure_workers(n_chunks);
            let mut workers = main_pat.take_workers();

            // Pre-allocate gradient accumulator slots (small: n_grad floats each).
            // These are reset inside the parallel loop together with the workers.
            let mut grad_workers: alloc::vec::Vec<nalgebra::DVector<f64>> = (0..n_chunks)
                .map(|_| nalgebra::DVector::<f64>::zeros(n_grad))
                .collect();

            // Process each chunk in parallel.
            // Each worker resets itself first (parallel memset), then accumulates.
            // This avoids sequential pre-reset overhead while keeping zero allocation.
            workers[..n_chunks]
                .par_iter_mut()
                .zip(term_chunks.into_par_iter())
                .zip(grad_workers.par_iter_mut())
                .for_each(|((worker, chunk), grad)| {
                    worker.reset(); // parallel reset: no sequential bottleneck
                    grad.fill(0.0);
                    for term in chunk {
                        accumulate_term(
                            term,
                            num_args,
                            &scalar_start_indices_per_arg,
                            &block_start_indices_per_arg,
                            &dof_per_arg,
                            &arg_ids,
                            &partition_idx_by_family,
                            nu,
                            &nu_id_cache,
                            worker,
                            grad,
                        );
                    }
                });

            // Merge all workers back into main_pat and neg_grad.
            for i in 0..n_chunks {
                main_pat.merge_from(&workers[i]);
                *neg_grad.scalar_vector_mut() += &grad_workers[i];
            }

            // Return workers to main_pat for reuse next iteration.
            main_pat.return_workers(workers);

            return;
        }
        let _ = parallelize;

        // ── Sequential path ─────────────────────────────────────────────────
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
                let idx_alpha = PartitionBlockIndex {
                    partition: variables.partition_idx_by_family[family_alpha],
                    block: block_start_idx_alpha,
                };

                // neg_grad -= J'r
                neg_grad.axpy_block(idx_alpha, &grad_block.as_view(), -1.0);
                let hessian_block = evaluated_term.hessian.block(arg_id_alpha, arg_id_alpha);
                assert_eq!(dof_alpha, hessian_block.nrows());
                assert_eq!(dof_alpha, hessian_block.ncols());

                // block diagonal: J'J
                hessian_block_triplet.add_lower_block(
                    idx_alpha,
                    idx_alpha,
                    &hessian_block.as_view(),
                );

                // LM damping: nu * I_d
                if nu != 0.0
                    && dof_alpha < nu_id_cache.len()
                    && let Some(ref ni) = nu_id_cache[dof_alpha]
                {
                    hessian_block_triplet.add_lower_block(idx_alpha, idx_alpha, &ni.as_view());
                }

                // off-diagonal hessian
                for arg_id_beta in 0..num_args {
                    let family_beta = arg_ids[arg_id_beta];
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
                    if scalar_start_idx_beta > scalar_start_idx_alpha {
                        continue;
                    }
                    let block_start_idx_beta =
                        block_start_indices_per_arg[arg_id_beta][var_idx_beta] as usize;
                    let hessian_block_alpha_beta =
                        evaluated_term.hessian.block(arg_id_alpha, arg_id_beta);
                    let idx_beta = PartitionBlockIndex {
                        partition: variables.partition_idx_by_family[family_beta],
                        block: block_start_idx_beta,
                    };
                    hessian_block_triplet.add_lower_block(
                        idx_alpha,
                        idx_beta,
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

/// Accumulate a single evaluated term into a `BlockSparseSymmetricMatrixPattern` and
/// a scalar gradient vector.
///
/// Extracted from `populate_upper_triangular_normal_equation` so it can be called
/// from rayon's parallel fold (where the accumulator is a thread-local pattern + DVector).
#[allow(clippy::too_many_arguments)]
fn accumulate_term<const INPUT_DIM: usize, const N: usize>(
    term: &crate::nlls::EvaluatedCostTerm<INPUT_DIM, N>,
    num_args: usize,
    scalar_start_indices_per_arg: &[alloc::vec::Vec<i64>],
    block_start_indices_per_arg: &[alloc::vec::Vec<i64>],
    dof_per_arg: &[usize],
    arg_ids: &[usize],
    partition_idx_by_family: &[usize],
    nu: f64,
    nu_id_cache: &[Option<nalgebra::DMatrix<f64>>],
    local_pat: &mut BlockSparseSymmetricMatrixPattern,
    local_grad: &mut nalgebra::DVector<f64>,
) {
    let idx = term.idx;
    debug_assert_eq!(idx.len(), num_args);

    for arg_id_alpha in 0..num_args {
        let dof_alpha = dof_per_arg[arg_id_alpha];
        let family_alpha = arg_ids[arg_id_alpha];
        if dof_alpha == 0 {
            continue;
        }
        let var_idx_alpha = idx[arg_id_alpha];
        let scalar_start_idx_alpha = scalar_start_indices_per_arg[arg_id_alpha][var_idx_alpha];
        let block_start_idx_alpha = block_start_indices_per_arg[arg_id_alpha][var_idx_alpha];
        if scalar_start_idx_alpha == -1 {
            continue;
        }
        let scalar_start_idx_alpha = scalar_start_idx_alpha as usize;
        let block_start_idx_alpha = block_start_idx_alpha as usize;

        let grad_block = term.gradient.block(arg_id_alpha);
        let idx_alpha = PartitionBlockIndex {
            partition: partition_idx_by_family[family_alpha],
            block: block_start_idx_alpha,
        };

        // local_grad -= J'r  (explicit DVectorView type to resolve as_view() ambiguity)
        let grad_view: nalgebra::DVectorView<'_, f64> = grad_block.as_view();
        local_grad
            .rows_mut(scalar_start_idx_alpha, dof_alpha)
            .axpy(-1.0, &grad_view, 1.0);

        // block diagonal: J'J
        let hessian_block = term.hessian.block(arg_id_alpha, arg_id_alpha);
        let h_view: nalgebra::DMatrixView<'_, f64> = hessian_block.as_view();
        local_pat.add_lower_block(idx_alpha, idx_alpha, &h_view);

        // LM damping: nu * I_d
        if nu != 0.0
            && dof_alpha < nu_id_cache.len()
            && let Some(ref ni) = nu_id_cache[dof_alpha]
        {
            local_pat.add_lower_block(idx_alpha, idx_alpha, &ni.as_view());
        }

        // off-diagonal hessian
        for arg_id_beta in 0..num_args {
            let family_beta = arg_ids[arg_id_beta];
            if arg_id_alpha == arg_id_beta {
                continue;
            }
            let dof_beta = dof_per_arg[arg_id_beta];
            if dof_beta == 0 {
                continue;
            }
            let var_idx_beta = idx[arg_id_beta];
            let scalar_start_idx_beta = scalar_start_indices_per_arg[arg_id_beta][var_idx_beta];
            if scalar_start_idx_beta == -1 {
                continue;
            }
            let scalar_start_idx_beta = scalar_start_idx_beta as usize;
            if scalar_start_idx_beta > scalar_start_idx_alpha {
                continue;
            }
            let block_start_idx_beta =
                block_start_indices_per_arg[arg_id_beta][var_idx_beta] as usize;
            let hessian_block_ab = term.hessian.block(arg_id_alpha, arg_id_beta);
            let idx_beta = PartitionBlockIndex {
                partition: partition_idx_by_family[family_beta],
                block: block_start_idx_beta,
            };
            let hab_view: nalgebra::DMatrixView<'_, f64> = hessian_block_ab.as_view();
            local_pat.add_lower_block(idx_alpha, idx_beta, &hab_view);
        }
    }
}
