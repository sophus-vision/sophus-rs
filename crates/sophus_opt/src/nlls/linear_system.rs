pub(crate) mod cost_system;
pub(crate) mod eq_system;

use sophus_solver::{
    BlockVector,
    IsLinearSolver as _,
    IsSymmetricMatrix,
    LinearSolverEnum,
    PartitionSpec,
    SymmetricMatrixBuilderEnum,
    SymmetricMatrixEnum,
};

use super::{
    CostSystem,
    EqSystem,
    NllsError,
};
use crate::variables::{
    VarFamilies,
    VarKind,
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
    pub(crate) sparse_hessian_plus_damping: SymmetricMatrixEnum,
    pub(crate) neg_gradient: BlockVector,
    pub(crate) solver: LinearSolverEnum,
    parallelize: bool,
}

///f
#[inline(always)]
pub fn phase(phase: &str) {
    let path = format!("{}", phase);
    tracing::trace!(path = path.as_str());
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
    pub fn from_families_costs_and_constraints(
        variables: &VarFamilies,
        cost_system: &CostSystem,
        eq_system: &EqSystem,
        solver: LinearSolverEnum,
        parallelize: bool,
    ) -> LinearSystem {
        assert!(variables.num_of_kind(VarKind::Marginalized) == 0);
        assert!(variables.num_of_kind(VarKind::Free) >= 1);

        phase("from_families_costs_and_constraints/befoe");

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
        phase("from_families_costs_and_constraints/partitions");

        let mut block_triplets = SymmetricMatrixBuilderEnum::zero(solver, &partitions);
        let mut neg_grad = BlockVector::zero(&partitions);
        phase("from_families_costs_and_constraints/zero");

        for cost in cost_system.evaluated_costs.iter() {
            cost.populate_upper_triangulatr_normal_equation(
                variables,
                cost_system.lm_damping,
                &mut block_triplets,
                &mut neg_grad,
            );
        }
        phase("from_families_costs_and_constraints/populate_upper_triangulatr_normal_equation");

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
        phase("from_families_costs_and_constraints/populate_lower_triangular_kkt_mat");

        Self {
            sparse_hessian_plus_damping: block_triplets.build(),
            neg_gradient: neg_grad,
            solver,
            parallelize,
        }
    }

    pub(crate) fn solve(&mut self) -> Result<nalgebra::DVector<f64>, NllsError> {
        let g = self
            .solver
            .solve(
                &self.sparse_hessian_plus_damping.compress(),
                self.neg_gradient.scalar_vector(),
            )
            .map_err(|e| NllsError::LinearSolver { source: e });
        phase("solve");

        g
    }
}
