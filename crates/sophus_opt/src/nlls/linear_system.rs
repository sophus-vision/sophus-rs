use eq_system::EqSystem;
use linear_solvers::{
    sparse_lu::SparseLu,
    sparse_qr::SparseQr,
    SolveError,
};
use quadratic_cost_system::CostSystem;

use crate::{
    block::{
        block_vector::BlockVector,
        symmetric_block_sparse_matrix_builder::SymmetricBlockSparseMatrixBuilder,
        PartitionSpec,
    },
    nlls::{
        linear_system::linear_solvers::{
            dense_lu::DenseLu,
            sparse_ldlt::SparseLdlt,
            IsSparseSymmetricLinearSystem,
        },
        LinearSolverType,
    },
    variables::{
        var_families::VarFamilies,
        VarKind,
    },
};

extern crate alloc;

/// KKT sub-system for the equality-constraints
pub mod eq_system;
/// Linear solvers
pub mod linear_solvers;
/// Normal equation system for the non-linear least squares cost
pub mod quadratic_cost_system;

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
    pub(crate) sparse_hessian_plus_damping: SymmetricBlockSparseMatrixBuilder,
    pub(crate) neg_gradient: BlockVector,
    pub(crate) linear_solver: LinearSolverType,
}

impl LinearSystem {
    /// Populates the complete linear system of the non-linear least squares problem with equality
    /// and >= constraints is given by the following KKT system:
    ///
    /// ```ascii
    /// 
    ///   J'J + nuI      G'     | dx              /  -J'r + G'lambda \
    ///      G           0      | -d lambda       \  -c              /
    /// ```
    ///
    /// where r is the residual, J is the Jacobian of the cost function, c is the residual and G
    /// is the Jacobian of the equality constraints, and lambda is the Lagrange multiplier.
    ///
    /// First, the normal equation system for the cost function is populated:
    ///
    /// ```ascii
    /// 
    ///   J'J + nuI      *    | dx        /  -J'r   \
    ///       *          *    | *         \   *     /
    /// ```
    ///
    /// Then, the normal equation system for the equality constraints is populated:
    ///
    /// ```ascii
    /// 
    ///      *           G'          *          | *               /  * + G'lambda  \
    ///      G           0           *          | -d lambda  =    \ -c             /
    /// ```
    pub fn from_families_costs_and_constraints(
        variables: &VarFamilies,
        cost_system: &CostSystem,
        eq_system: &EqSystem,
        linear_solver: LinearSolverType,
    ) -> LinearSystem {
        assert!(variables.num_of_kind(VarKind::Marginalized) == 0);
        assert!(variables.num_of_kind(VarKind::Free) >= 1);

        // Note let's first focus on these special cases, before attempting a
        // general version covering all cases holistically. Also, it might not be trivial
        // to implement VarKind::Marginalized > 1.
        //  - Example, the the arrow-head sparsity uses a recursive application of the
        //    Schur-Complement.

        let mut partitions = vec![];
        for i in 0..variables.collection.len() {
            partitions.push(PartitionSpec {
                num_blocks: variables.free_vars()[i],
                block_dim: variables.dims()[i],
            });
        }

        partitions.extend(eq_system.partitions.clone());

        let mut block_triplets = SymmetricBlockSparseMatrixBuilder::zero(&partitions);
        let mut neg_grad = BlockVector::zero(&partitions);

        for cost in cost_system.evaluated_costs.iter() {
            cost.populate_upper_triangulatr_normal_equation(
                variables,
                cost_system.lm_damping,
                &mut block_triplets,
                &mut neg_grad,
            );
        }

        for (constraint_idx, eq_constraint_set) in
            eq_system.evaluated_eq_constraints.iter().enumerate()
        {
            eq_constraint_set.populate_upper_triangular_kkt_mat(
                variables,
                &eq_system.lambda,
                constraint_idx,
                &mut block_triplets,
                &mut neg_grad,
            );
        }
        Self {
            sparse_hessian_plus_damping: block_triplets,
            neg_gradient: neg_grad,
            linear_solver,
        }
    }

    pub(crate) fn solve(&mut self) -> Result<nalgebra::DVector<f64>, SolveError> {
        match self.linear_solver {
            LinearSolverType::SparseLdlt(ldlt_params) => SparseLdlt::new(ldlt_params).solve(
                &self.sparse_hessian_plus_damping,
                self.neg_gradient.scalar_vector_mut(),
            ),
            LinearSolverType::DenseLu => DenseLu {}.solve(
                &self.sparse_hessian_plus_damping,
                self.neg_gradient.scalar_vector(),
            ),
            LinearSolverType::SparseLu => SparseLu {}.solve(
                &self.sparse_hessian_plus_damping,
                self.neg_gradient.scalar_vector(),
            ),
            LinearSolverType::SparseQr => SparseQr {}.solve(
                &self.sparse_hessian_plus_damping,
                self.neg_gradient.scalar_vector(),
            ),
        }
    }
}
