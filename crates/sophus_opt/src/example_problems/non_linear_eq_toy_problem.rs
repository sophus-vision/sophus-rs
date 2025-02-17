use sophus_autodiff::linalg::VecF64;

use crate::{
    nlls::{
        costs::ExampleNonLinearCostTerm,
        eq_constraints::ExampleNonLinearEqConstraint,
        optimize_nlls_with_eq_constraints,
        CostFn,
        CostTerms,
        EqConstraintFn,
        EqConstraints,
        LinearSolverType,
        OptParams,
    },
    variables::{
        VarBuilder,
        VarFamily,
        VarKind,
    },
};
extern crate alloc;

/// non-linear == constraint toy problem
pub struct NonLinearEqToyProblem {}

impl Default for NonLinearEqToyProblem {
    fn default() -> Self {
        Self::new()
    }
}

impl NonLinearEqToyProblem {
    fn new() -> Self {
        Self {}
    }

    /// Test the non-linear equality constraint problem
    pub fn test(&self, solver: LinearSolverType) {
        use sophus_autodiff::linalg::EPS_F64;

        const VAR_X: &str = "x";
        const EQ_CONSTRAINT_RHS: f64 = 1.0;

        let initial_x = VecF64::<2>::new(2.0, 1.0);

        let variables = VarBuilder::new()
            .add_family(VAR_X, VarFamily::new(VarKind::Free, alloc::vec![initial_x]))
            .build();
        let cost_terms = CostTerms::new(
            [VAR_X],
            alloc::vec![ExampleNonLinearCostTerm {
                z: VecF64::<2>::new(1.0, 1.0),
                entity_indices: [0]
            },],
        );
        let eq_constraints = EqConstraints::new(
            [VAR_X],
            vec![ExampleNonLinearEqConstraint {
                lhs: EQ_CONSTRAINT_RHS,
                entity_indices: [0],
            }],
        );

        // illustrate that the initial guess is not feasible
        assert!(
            ExampleNonLinearEqConstraint::residual(initial_x, EQ_CONSTRAINT_RHS).norm() > EPS_F64
        );

        let refined_variables = optimize_nlls_with_eq_constraints(
            variables,
            alloc::vec![CostFn::new_boxed((), cost_terms.clone(),)],
            alloc::vec![EqConstraintFn::new_boxed((), eq_constraints,)],
            OptParams {
                num_iterations: 10,
                initial_lm_damping: EPS_F64,
                parallelize: true,
                solver,
            },
        )
        .unwrap()
        .variables;
        let refined_x = refined_variables.get_members::<VecF64<2>>(VAR_X)[0];

        approx::assert_abs_diff_eq!(refined_x, VecF64::<2>::new(0.0, 1.0), epsilon = 1e-6);

        // converged solution should satisfy the equality constraint
        approx::assert_abs_diff_eq!(
            ExampleNonLinearEqConstraint::residual(refined_x, EQ_CONSTRAINT_RHS)[0],
            0.0,
            epsilon = 1e-6
        );
    }
}

#[test]
fn normalize_opt_tests() {
    for solver in LinearSolverType::indefinite_solvers() {
        NonLinearEqToyProblem::new().test(solver);
    }
}
