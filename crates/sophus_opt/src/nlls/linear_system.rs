use crate::block::block_vector::BlockVector;
use crate::block::symmetric_block_sparse_matrix_builder::SymmetricBlockSparseMatrixBuilder;
use crate::block::PartitionSpec;
use crate::nlls::linear_system::linear_solvers::dense_lu::DenseLU;
use crate::nlls::linear_system::linear_solvers::sparse_ldlt::SparseLDLt;
use crate::nlls::linear_system::linear_solvers::IsSparseSymmetricLinearSystem;
use crate::nlls::LinearSolverType;
use crate::variables::var_families::VarFamilies;
use crate::variables::VarKind;
use eq_system::EqSystem;
use linear_solvers::sparse_lu::SparseLU;
use linear_solvers::sparse_qr::SparseQR;
use linear_solvers::SolveError;
use quadratic_cost_system::CostSystem;

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

// Linear system of the normal equation
pub(crate) struct LinearSystem {
    pub(crate) sparse_hessian_plus_damping: SymmetricBlockSparseMatrixBuilder,
    pub(crate) neg_gradient: BlockVector,
    pub(crate) linear_solver: LinearSolverType,
}

impl LinearSystem {
    pub(crate) fn from_families_costs_and_constraints(
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
        //  -  Example, the the arrow-head sparsity uses a recursive application of the Schur-Complement.

        // The complete linear system of the non-linear least squares problem with equality and
        // >= constraints is given by the following KKT system:
        //
        //   J'J + nuI      G'     | dx        /  -J'r + G'l \
        //      G           0      | -dl       \  -c         /
        //
        // where r is the residual, J is the Jacobian of the cost function, c is the residual and G
        // is the Jacobian of the equality constraints.
        //
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

        //   J'J + nuI      *    | dx        /  -J'r   \
        //      *           *    | *         \   *     /
        //
        // where r is the residual, J is the Jacobian, dx is the incremental update for the
        // variables, and nu is the Levenberg-Marquardt damping parameter.
        for cost in cost_system.evaluated_costs.iter() {
            cost.populate_upper_triangulatr_normal_equation(
                variables,
                cost_system.lm_damping,
                &mut block_triplets,
                &mut neg_grad,
            );
        }
        // Populates the KKT matrix for the equality constraints:
        //
        //      *           G'          *          | *         /  * + G'l  \
        //      G           0           *          | -dl  =    \ -c        /
        //
        // where c is the residual of the equality constraints, G is the Jacobian of the equality
        // constraints, and lambda is the Lagrange multiplier.
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
            LinearSolverType::SparseLDLt(ldlt_params) => SparseLDLt::new(ldlt_params).solve(
                &self.sparse_hessian_plus_damping,
                self.neg_gradient.scalar_vector_mut(),
            ),
            LinearSolverType::DenseLU => DenseLU {}.solve(
                &self.sparse_hessian_plus_damping,
                self.neg_gradient.scalar_vector(),
            ),
            LinearSolverType::SparseLU => SparseLU {}.solve(
                &self.sparse_hessian_plus_damping,
                self.neg_gradient.scalar_vector(),
            ),
            LinearSolverType::SparseQR => SparseQR {}.solve(
                &self.sparse_hessian_plus_damping,
                self.neg_gradient.scalar_vector(),
            ),
        }
    }
}
