use sophus_autodiff::linalg::VecF64;

use crate::{
    nlls::{
        constraint::{
            eq_constraint::EqConstraints,
            eq_constraint_fn::EqConstraintFn,
        },
        functor_library::{
            costs::example_non_linear_cost::ExampleNonLinearQuadraticCost,
            eq_constraints::small_non_linear_eq::SmallNonLinearEqConstraint,
        },
        optimize_with_eq_constraints,
        quadratic_cost::{
            cost_fn::CostFn,
            cost_term::CostTerms,
        },
        LinearSolverType,
        OptParams,
    },
    variables::{
        var_builder::VarBuilder,
        var_family::VarFamily,
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
            alloc::vec![ExampleNonLinearQuadraticCost {
                z: VecF64::<2>::new(1.0, 1.0),
                entity_indices: [0]
            },],
        );
        let eq_constraints = EqConstraints::new(
            [VAR_X],
            vec![SmallNonLinearEqConstraint {
                lhs: EQ_CONSTRAINT_RHS,
                entity_indices: [0],
            }],
        );

        // illustrate that the initial guess is not feasible
        assert!(
            SmallNonLinearEqConstraint::residual(initial_x, EQ_CONSTRAINT_RHS).norm() > EPS_F64
        );

        let refined_variables = optimize_with_eq_constraints(
            variables,
            alloc::vec![CostFn::new_box((), cost_terms.clone(),)],
            alloc::vec![EqConstraintFn::new_box((), eq_constraints,)],
            OptParams {
                num_iterations: 10,
                initial_lm_damping: EPS_F64,
                parallelize: true,
                linear_solver: solver,
            },
        )
        .unwrap()
        .variables;
        let refined_x = refined_variables.get_members::<VecF64<2>>(VAR_X)[0];

        approx::assert_abs_diff_eq!(refined_x, VecF64::<2>::new(0.0, 1.0), epsilon = 1e-6);

        // converged solution should satisfy the equality constraint
        approx::assert_abs_diff_eq!(
            SmallNonLinearEqConstraint::residual(refined_x, EQ_CONSTRAINT_RHS)[0],
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
