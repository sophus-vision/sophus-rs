pub(crate) mod cost_system;
pub(crate) mod eq_system;

use sophus_solver::{
    LinearSolverEnum,
    Schur,
    matrix::{
        PartitionSet,
        SymmetricMatrixBuilderEnum,
        SymmetricMatrixEnum,
        block::BlockVector,
        block_sparse::{
            BlockSparseSymmetricMatrixPattern,
            BlockSparseSymmetricSymbolicBuilder,
        },
    },
};

use super::{
    EqSystem,
    NllsError,
};
use crate::{
    nlls::{
        CostError,
        IsEvaluatedCost,
        damped_hessian::DampedHessian,
    },
    variables::{
        VarFamilies,
        VarKind,
    },
};

extern crate alloc;

/// Evaluation mode
#[derive(Clone, Debug, Copy, PartialEq)]
pub enum EvalMode {
    /// calculate the derivatives
    CalculateDerivatives,
    /// don't calculate the derivatives
    DontCalculateDerivatives,
}

/// Linear system of the non-linear least squares problem with equality constraints
pub struct LinearSystem {
    pub(crate) neg_gradient: BlockVector,
    /// The Hessian plus LM damping: `H + νI`, bundled with its solve state.
    damped_hessian: DampedHessian,
}

impl LinearSystem {
    /// Populates the complete linear system of the non-linear least squares problem with equality
    /// and >= constraints is given by the following KKT system:
    ///
    /// ```ascii
    /// --------------------------------------------------------------
    /// | J'J + nuI      G'     | dx              |  -J'r + G'lambda |
    /// |    G           0      | -d lambda       |  -c              |
    /// --------------------------------------------------------------
    /// ```
    ///
    /// where r is the residual, J is the Jacobian of the cost function, c is the residual and G
    /// is the Jacobian of the equality constraints, and lambda is the Lagrange multiplier.
    ///
    /// First, the normal equation system for the cost function is populated:
    ///
    /// ```ascii
    /// --------------------------------------------------------------
    /// | J'J + nuI      *      | dx              |  -J'r + *        |
    /// |     *          *      | *               |   *              |
    /// --------------------------------------------------------------
    /// ```
    ///
    /// Then, the normal equation system for the equality constraints is populated:
    ///
    /// ```ascii
    /// --------------------------------------------------------------
    /// |    *           G'     |  *              |   * + G'lambda   |
    /// |    G           0      | -d lambda       |  -c              |
    /// --------------------------------------------------------------
    /// ```
    /// Build the Hessian sparsity pattern from the raw (pre-reduction) cost function terms.
    ///
    /// Iterates over every raw cost term's variable index without computing any matrix values.
    /// Call this once after `CostSystem::new` to obtain a pre-built
    /// `BlockSparseSymmetricMatrixPattern` that can be passed as `pattern` to the very first
    /// optimizer iteration, avoiding the slow sequential `BlockSparseLower` bootstrap.
    pub fn build_initial_pattern(
        variables: &VarFamilies,
        cost_fns: &[alloc::boxed::Box<dyn crate::nlls::IsCostFn>],
        eq_system: &EqSystem,
    ) -> BlockSparseSymmetricMatrixPattern {
        // Mirror the partition layout from `from_families_costs_and_constraints`.
        let mut partition_specs = variables.build_partition_specs();
        partition_specs.extend(eq_system.partitions.clone());
        let partitions = PartitionSet::new(partition_specs);

        let mut sym_builder = BlockSparseSymmetricSymbolicBuilder::new(partitions);
        for cost_fn in cost_fns.iter() {
            cost_fn.populate_symbolic(variables, &mut sym_builder);
        }
        sym_builder.into_pattern()
    }

    /// Build the linear system from variable families, evaluated costs, and equality constraints.
    ///
    /// Returns `(LinearSystem, Option<BlockSparseSymmetricMatrixPattern>)`.  The pattern is `Some`
    /// when the builder was block-sparse (all cases without equality constraints); pass it back as
    /// `pattern` on the next iteration to skip the O(K log K) symbolic rebuild.
    pub fn from_families_costs_and_constraints(
        variables: &VarFamilies,
        evaluated_costs: &[alloc::boxed::Box<dyn IsEvaluatedCost>],
        nu: f64,
        eq_system: &EqSystem,
        mut solver: LinearSolverEnum,
        parallelize: bool,
        pattern: Option<BlockSparseSymmetricMatrixPattern>,
    ) -> Result<(LinearSystem, Option<BlockSparseSymmetricMatrixPattern>), CostError> {
        assert!(variables.num_of_kind(VarKind::Free) >= 1);

        solver.set_parallelize(parallelize);

        // Build partition specs: Free families first (BTreeMap order), then Marginalized.
        // This ordering must match the partition_idx_by_family mapping in VarFamilies.
        let mut partition_specs = variables.build_partition_specs();
        partition_specs.extend(eq_system.partitions.clone());

        let partitions = PartitionSet::new(partition_specs);
        let mut neg_grad = BlockVector::zero(partitions.specs());
        let mut block_triplets = if let Some(pat) = pattern {
            SymmetricMatrixBuilderEnum::from_block_sparse_pattern(pat, solver)
        } else {
            SymmetricMatrixBuilderEnum::zero(solver, partitions)
        };

        for evaluated_cost in evaluated_costs.iter() {
            evaluated_cost.populate_upper_triangular_normal_equation(
                variables,
                nu,
                &mut block_triplets,
                &mut neg_grad,
                parallelize,
            );
        }

        for (constraint_idx, eq_constraint_set) in
            eq_system.evaluated_eq_constraints.iter().enumerate()
        {
            eq_constraint_set.populate_lower_triangular_kkt_mat(
                variables,
                &eq_system.lambda,
                constraint_idx,
                &mut block_triplets,
                &mut neg_grad,
            );
        }

        // Augmented Lagrangian regularization: add -μI to the (λ,λ) diagonal
        // block of the KKT matrix to prevent singularity.
        // Uses the LM damping ν as the regularization parameter.
        if !eq_system.partitions.is_empty() {
            let num_var_partitions = variables.total_active_partition_count();
            for (ci, spec) in eq_system.partitions.iter().enumerate() {
                let partition = num_var_partitions + ci;
                for block in 0..spec.block_count {
                    let idx = sophus_solver::matrix::PartitionBlockIndex { partition, block };
                    let dim = spec.block_dim;
                    let mut reg = nalgebra::DMatrix::zeros(dim, dim);
                    for k in 0..dim {
                        reg[(k, k)] = -nu;
                    }
                    block_triplets.add_lower_block(idx, idx, &reg.as_view());
                }
            }
        }

        let (built_hessian, next_pattern) = block_triplets.build_with_pattern();

        let num_marg_scalars = variables.num_marg_scalars();
        let free_partition_count = variables.free_partition_count;

        // If a Schur solver was requested and there are marginalized variables,
        // wrap the block-sparse matrix in a Schur complement structure.
        let matrix = if solver.is_schur() && num_marg_scalars > 0 {
            let inner = built_hessian
                .into_block_sparse_lower()
                .expect("Schur solver requires BlockSparseLower matrix");
            SymmetricMatrixEnum::Schur(Schur::new(
                inner,
                free_partition_count,
                variables.total_active_partition_count(),
                solver.schur_inner_solver(),
                parallelize,
            ))
        } else {
            built_hessian
        };

        let damped_hessian =
            DampedHessian::new(matrix, nu, variables.total_active_partition_count(), solver);

        Ok((
            Self {
                damped_hessian,
                neg_gradient: neg_grad,
            },
            next_pattern,
        ))
    }

    /// The Hessian with LM damping applied: `H + νI`.
    ///
    /// Call `get_block`, `inverse_block`, or `matrix.as_schur()` directly on the returned value.
    pub fn damped_hessian(&self) -> &DampedHessian {
        &self.damped_hessian
    }

    /// Mutable access to `H + νI` — needed to call `inverse_block` (which caches state).
    pub fn damped_hessian_mut(&mut self) -> &mut DampedHessian {
        &mut self.damped_hessian
    }

    /// Consume the linear system, returning `(neg_gradient, damped_hessian)`.
    pub(crate) fn into_gradient_and_hessian(self) -> (BlockVector, DampedHessian) {
        (self.neg_gradient, self.damped_hessian)
    }

    pub(crate) fn solve(&mut self) -> Result<nalgebra::DVector<f64>, NllsError> {
        let g = self.neg_gradient.scalar_vector().clone();
        self.damped_hessian
            .solve(&g)
            .map_err(|e| NllsError::LinearSolver { source: e })
    }
}
