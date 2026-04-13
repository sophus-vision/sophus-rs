use core::{
    marker::PhantomData,
    ops::Range,
};
use std::fmt::Debug;

use snafu::Snafu;
use sophus_solver::matrix::block_sparse::BlockSparseSymmetricSymbolicBuilder;

use super::{
    cost_term::{
        CostTerms,
        HasResidualFn,
    },
    evaluated_cost::IsEvaluatedCost,
};
use crate::{
    nlls::{
        cost::{
            compare_idx::{
                CompareIdx,
                c_from_var_kind,
            },
            evaluated_cost::EvaluatedCost,
            evaluated_term::EvaluatedCostTerm,
        },
        linear_system::EvalMode,
    },
    robust_kernel::RobustKernel,
    variables::{
        IsVarTuple,
        VarFamilies,
        VarFamilyError,
        VarKind,
    },
};

extern crate alloc;

/// Quadratic cost function of the non-linear least squares problem.
pub trait IsCostFn {
    /// Evaluate the cost function.
    fn eval(
        &self,
        var_pool: &VarFamilies,
        eval_mode: EvalMode,
        parallelize: bool,
    ) -> Result<alloc::boxed::Box<dyn IsEvaluatedCost>, CostError>;

    /// Sort the terms of the cost function (to ensure more efficient evaluation and reduction over
    /// conditioned variables).
    fn sort(&mut self, variables: &VarFamilies);

    /// Get the robust kernel function.
    fn robust_kernel(&self) -> Option<RobustKernel>;

    /// Names of the variable families this cost function touches.
    fn cost_family_names(&self) -> alloc::vec::Vec<String>;

    /// Compute the total squared cost without building the full EvaluatedCost structure.
    ///
    /// Used by merit evaluation in the LM loop: all variables are treated as conditioned
    /// (no Jacobians needed), so this is much cheaper than `eval`.
    fn calc_total_cost(&self, var_pool: &VarFamilies, parallelize: bool) -> Result<f64, CostError>;

    /// Record only the sparsity structure into a symbolic builder — no matrix values computed.
    ///
    /// Iterates over the RAW (pre-reduction) cost terms and records which (row, col) block
    /// positions will be written during a numeric pass.  Used to pre-build the
    /// `BlockSparseSymmetricMatrixPattern` before the first optimizer iteration so that
    /// iteration 0 uses the same fast parallel path as all subsequent ones.
    fn populate_symbolic(
        &self,
        variables: &VarFamilies,
        sym_builder: &mut BlockSparseSymmetricSymbolicBuilder,
    );
}

/// Generic cost function of the non-linear least squares problem.
///
/// It represents a cost function that is a sum of squares of multiple residual function terms:
///
/// `f(x) = ∑ᵢ [g(Vⁱ₀, Vⁱ₁, ..., Vⁱₙ₋₁)]²`
///
/// All terms are based on a common residual function `g` and a set of input arguments
/// `Vⁱ₀, Vⁱ₁, ...,  Vⁱₙ₋₁`.
///
/// This struct is passed as a box of [IsEvaluatedCost] to the optimizer.
///
/// ## Generic parameters
///
///  * `INPUT_DIM`
///    - Total input dimension of the common residual function `g`. It is the sum of argument
///      dimensions: |Vⁱ₀| + |Vⁱ₁| + ... + |Vⁱₙ₋₁|.
///  * `N`
///    - Number of arguments of the common residual function `g`.
///  * `GlobalConstants`
///    - Type of the global constants which are passed to the residual function. If no global
///      constants are needed, use `()`.
///  * `ResidualFn`
///    - The common residual function `g`.
///  * `Args`
///    - Tuple of input argument types: `(Vⁱ₀, Vⁱ₁, ..., Vⁱₙ₋₁)`.
#[derive(Debug, Clone)]
pub struct CostFn<
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync,
    ResidualFn: HasResidualFn<INPUT_DIM, N, GlobalConstants, Args>,
    Args: IsVarTuple<N> + 'static,
> {
    global_constants: GlobalConstants,
    cost_terms: CostTerms<INPUT_DIM, N, GlobalConstants, Args, ResidualFn>,
    robust_kernel: Option<RobustKernel>,
    phantom: PhantomData<Args>,
}

impl<
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync,
    ResidualFn: HasResidualFn<INPUT_DIM, N, GlobalConstants, Args> + 'static,
    Args: IsVarTuple<N> + 'static,
> CostFn<INPUT_DIM, N, GlobalConstants, ResidualFn, Args>
{
    /// Create from global constants and cost terms.
    pub fn new_boxed(
        global_constants: GlobalConstants,
        terms: CostTerms<INPUT_DIM, N, GlobalConstants, Args, ResidualFn>,
    ) -> alloc::boxed::Box<dyn IsCostFn> {
        alloc::boxed::Box::new(Self {
            global_constants,
            cost_terms: terms,
            robust_kernel: None,
            phantom: PhantomData,
        })
    }

    /// Create from global constants, cost terms, and a robust kernel.
    pub fn new_boxed_robust(
        global_constants: GlobalConstants,
        terms: CostTerms<INPUT_DIM, N, GlobalConstants, Args, ResidualFn>,
        robust_kernel: RobustKernel,
    ) -> alloc::boxed::Box<dyn IsCostFn> {
        alloc::boxed::Box::new(Self {
            global_constants,
            cost_terms: terms,
            robust_kernel: Some(robust_kernel),
            phantom: PhantomData,
        })
    }
}

/// Errors that can occur when working with variables as cost arguments.
#[derive(Snafu, Debug)]
pub enum CostError {
    /// Error when working with variable as cost arguments.
    #[snafu(display("CostError({})", source))]
    VariableFamilyError {
        /// source
        source: VarFamilyError,
    },
}

impl<
    const INPUT_DIM: usize,
    const N: usize,
    GlobalConstants: 'static + Send + Sync,
    ResidualFn: HasResidualFn<INPUT_DIM, N, GlobalConstants, Args> + 'static,
    Args: IsVarTuple<N> + 'static,
> IsCostFn for CostFn<INPUT_DIM, N, GlobalConstants, ResidualFn, Args>
{
    fn eval(
        &self,
        variables: &VarFamilies,
        eval_mode: EvalMode,
        parallelize: bool,
    ) -> Result<alloc::boxed::Box<dyn IsEvaluatedCost>, CostError> {
        let mut var_kind_array =
            Args::var_kind_array(variables, self.cost_terms.family_names.clone());

        if eval_mode == EvalMode::DontCalculateDerivatives {
            var_kind_array = var_kind_array.map(|_x| VarKind::Conditioned)
        }

        let mut evaluated_terms = EvaluatedCost::new(self.cost_terms.family_names.clone());

        let var_family_tuple =
            Args::ref_var_family_tuple(variables, self.cost_terms.family_names.clone())
                .map_err(|e| CostError::VariableFamilyError { source: e })?;

        let eval_res = |term: &ResidualFn| {
            term.eval(
                &self.global_constants,
                *term.idx_ref(),
                Args::extract(&var_family_tuple, *term.idx_ref()),
                var_kind_array,
                self.robust_kernel,
            )
        };

        let reduction_ranges = self.cost_terms.reduction_ranges.as_ref().unwrap();

        #[cfg(not(target_arch = "wasm32"))]
        {
            #[derive(Debug)]
            enum ParallelizationStrategy {
                None,
                OuterLoop,
                InnerLoop,
            }
            const OUTER_LOOP_THRESHOLD: usize = 100;
            const INNER_LOOP_THRESHOLD: f64 = 100.0;
            const REDUCTION_RATIO_THRESHOLD: f64 = 1.0;

            let average_inner_loop_size =
                self.cost_terms.collection.len() as f64 / reduction_ranges.len() as f64;
            let reduction_ratio = average_inner_loop_size / reduction_ranges.len() as f64;

            let parallelization_strategy = match parallelize {
                true => {
                    if reduction_ranges.len() >= OUTER_LOOP_THRESHOLD
                        && reduction_ratio < REDUCTION_RATIO_THRESHOLD
                    {
                        // There are many outer terms, and significantly less inner terms on average
                        ParallelizationStrategy::OuterLoop
                    } else if average_inner_loop_size >= INNER_LOOP_THRESHOLD {
                        // There are many inner terms on average.
                        ParallelizationStrategy::InnerLoop
                    } else {
                        ParallelizationStrategy::None
                    }
                }
                false => ParallelizationStrategy::None,
            };

            match parallelization_strategy {
                ParallelizationStrategy::None => {
                    evaluated_terms.terms.reserve(reduction_ranges.len());
                    for range in reduction_ranges.iter() {
                        let mut evaluated_term_sum: Option<EvaluatedCostTerm<INPUT_DIM, N>> = None;

                        for term in self.cost_terms.collection[range.start..range.end].iter() {
                            match evaluated_term_sum {
                                Some(mut sum) => {
                                    sum.reduce(eval_res(term));
                                    evaluated_term_sum = Some(sum);
                                }
                                None => evaluated_term_sum = Some(eval_res(term)),
                            }
                        }

                        evaluated_terms.terms.push(evaluated_term_sum.unwrap());
                    }
                }
                ParallelizationStrategy::OuterLoop => {
                    use rayon::prelude::*;

                    evaluated_terms.terms = reduction_ranges
                        .par_iter() // parallelize over the outer terms
                        .map(|range| {
                            let evaluated_term_sum = self.cost_terms.collection
                                [range.start..range.end]
                                .iter() // sequential inner loop
                                .fold(
                                    None,
                                    |acc: Option<EvaluatedCostTerm<INPUT_DIM, N>>, term| {
                                        let evaluated_term = eval_res(term);
                                        match acc {
                                            Some(mut sum) => {
                                                sum.reduce(evaluated_term);
                                                Some(sum)
                                            }
                                            None => Some(evaluated_term),
                                        }
                                    },
                                );

                            evaluated_term_sum.unwrap()
                        })
                        .collect();
                }
                ParallelizationStrategy::InnerLoop => {
                    use rayon::prelude::*;

                    evaluated_terms.terms = reduction_ranges
                        .iter() // sequential outer loop
                        .map(|range| {
                            // We know on average there are many inner terms, however, there might
                            // be outliers.
                            //
                            // todo: Consider adding an if statement here and only parallelize the
                            //       inner loop if the range length is greater than some threshold.
                            let evaluated_term_sum = self.cost_terms.collection
                                [range.start..range.end]
                                .par_iter() // parallelize over the inner terms
                                .fold(
                                    || None,
                                    |acc: Option<EvaluatedCostTerm<INPUT_DIM, N>>, term| {
                                        let evaluated_term = eval_res(term);
                                        match acc {
                                            Some(mut sum) => {
                                                sum.reduce(evaluated_term);
                                                Some(sum)
                                            }
                                            None => Some(evaluated_term),
                                        }
                                    },
                                )
                                .reduce(
                                    || None,
                                    |acc, evaluated_term| match (acc, evaluated_term) {
                                        (Some(mut sum), Some(evaluated_term)) => {
                                            sum.reduce(evaluated_term);
                                            Some(sum)
                                        }
                                        (None, Some(evaluated_term)) => Some(evaluated_term),
                                        _ => None,
                                    },
                                );

                            evaluated_term_sum.unwrap()
                        })
                        .collect();
                }
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            let _ignore = parallelize;

            evaluated_terms.terms.reserve(reduction_ranges.len());
            for range in reduction_ranges.iter() {
                let mut evaluated_term_sum: Option<EvaluatedCostTerm<INPUT_DIM, N>> = None;

                for term in self.cost_terms.collection[range.start..range.end].iter() {
                    match evaluated_term_sum {
                        Some(mut sum) => {
                            sum.reduce(eval_res(term));
                            evaluated_term_sum = Some(sum);
                        }
                        None => evaluated_term_sum = Some(eval_res(term)),
                    }
                }

                evaluated_terms.terms.push(evaluated_term_sum.unwrap());
            }
        }

        Ok(alloc::boxed::Box::new(evaluated_terms))
    }

    fn sort(&mut self, variables: &VarFamilies) {
        let var_kind_array = &Args::var_kind_array(variables, self.cost_terms.family_names.clone());

        let c_array = c_from_var_kind(var_kind_array);

        let less = CompareIdx::new(&c_array);

        assert!(!self.cost_terms.collection.is_empty());

        self.cost_terms
            .collection
            .sort_by(|a, b| less.le_than(*a.idx_ref(), *b.idx_ref()));

        for t in 0..self.cost_terms.collection.len() - 1 {
            assert!(
                less.le_than(
                    *self.cost_terms.collection[t].idx_ref(),
                    *self.cost_terms.collection[t + 1].idx_ref()
                ) != core::cmp::Ordering::Greater
            );
        }

        let mut reduction_ranges: alloc::vec::Vec<Range<usize>> = alloc::vec![];
        let mut i = 0;
        while i < self.cost_terms.collection.len() {
            let outer_term = &self.cost_terms.collection[i];
            let outer_term_idx = i;
            while i < self.cost_terms.collection.len()
                && less.free_vars_equal(
                    outer_term.idx_ref(),
                    self.cost_terms.collection[i].idx_ref(),
                )
            {
                i += 1;
            }
            reduction_ranges.push(outer_term_idx..i);
        }

        self.cost_terms.reduction_ranges = Some(reduction_ranges);
    }

    fn cost_family_names(&self) -> alloc::vec::Vec<String> {
        self.cost_terms.family_names.to_vec()
    }

    fn robust_kernel(&self) -> Option<RobustKernel> {
        self.robust_kernel
    }

    fn calc_total_cost(
        &self,
        variables: &VarFamilies,
        parallelize: bool,
    ) -> Result<f64, CostError> {
        let var_family_tuple =
            Args::ref_var_family_tuple(variables, self.cost_terms.family_names.clone())
                .map_err(|e| CostError::VariableFamilyError { source: e })?;

        // Treat all variables as conditioned — no Jacobians computed.
        let var_kind_array: [VarKind; N] = core::array::from_fn(|_| VarKind::Conditioned);

        let eval_cost = |term: &ResidualFn| -> f64 {
            term.eval(
                &self.global_constants,
                *term.idx_ref(),
                Args::extract(&var_family_tuple, *term.idx_ref()),
                var_kind_array,
                self.robust_kernel,
            )
            .cost
        };

        #[cfg(not(target_arch = "wasm32"))]
        if parallelize {
            use rayon::prelude::*;
            return Ok(self.cost_terms.collection.par_iter().map(eval_cost).sum());
        }

        Ok(self.cost_terms.collection.iter().map(eval_cost).sum())
    }

    fn populate_symbolic(
        &self,
        variables: &VarFamilies,
        sym_builder: &mut BlockSparseSymmetricSymbolicBuilder,
    ) {
        let family_names = &self.cost_terms.family_names;
        let num_args = family_names.len();
        let mut scalar_start_indices_per_arg = alloc::vec::Vec::new();
        let mut block_start_indices_per_arg = alloc::vec::Vec::new();
        let mut dof_per_arg = alloc::vec::Vec::new();
        let mut arg_ids = alloc::vec::Vec::new();
        for name in family_names.iter() {
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

        // Iterate over RAW (pre-reduction) terms to capture all variable index combinations.
        for term in self.cost_terms.collection.iter() {
            let idx = term.idx_ref();
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
                let block_start_idx_alpha = block_start_idx_alpha as usize;
                let idx_alpha = sophus_solver::matrix::PartitionBlockIndex {
                    partition: variables.partition_idx_by_family[family_alpha],
                    block: block_start_idx_alpha,
                };

                // diagonal block
                sym_builder.add_lower_block(idx_alpha, idx_alpha);

                // off-diagonal blocks
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
                    let scalar_start_idx_alpha_usize = scalar_start_idx_alpha as usize;
                    let scalar_start_idx_beta = scalar_start_idx_beta as usize;
                    if scalar_start_idx_beta > scalar_start_idx_alpha_usize {
                        continue;
                    }
                    let block_start_idx_beta =
                        block_start_indices_per_arg[arg_id_beta][var_idx_beta] as usize;
                    let idx_beta = sophus_solver::matrix::PartitionBlockIndex {
                        partition: variables.partition_idx_by_family[family_beta],
                        block: block_start_idx_beta,
                    };
                    sym_builder.add_lower_block(idx_alpha, idx_beta);
                }
            }
        }
    }
}
