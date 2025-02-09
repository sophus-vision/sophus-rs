use sophus_autodiff::linalg::VecF64;
use sophus_lie::prelude::IsMatrix;

use crate::{
    nlls::{
        constraint::{
            eq_constraint::EqConstraints,
            eq_constraint_fn::EqConstraintFn,
        },
        cost::{
            cost_fn::CostFn,
            cost_term::CostTerms,
        },
        functor_library::{
            costs::quadratic1::Quadratic1CostTerm,
            eq_constraints::linear_eq::LinearEqConstraint1,
        },
        optimize_nlls_with_eq_constraints,
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

/// Simple linear equality constraint problem
pub struct LinearEqToyProblem {}

impl Default for LinearEqToyProblem {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearEqToyProblem {
    fn new() -> Self {
        Self {}
    }

    /// Test the linear equality constraint problem
    pub fn test(&self, solver: LinearSolverType) {
        use sophus_autodiff::linalg::EPS_F64;

        const VAR_X: &str = "x";

        const EQ_CONSTRAINT_RHS: f64 = 2.0;
        let initial_x0 = VecF64::<1>::from_f64(2.0);
        let initial_x1 = VecF64::<1>::from_f64(2.0);

        let variables = VarBuilder::new()
            .add_family(
                VAR_X,
                VarFamily::new(VarKind::Free, alloc::vec![initial_x0, initial_x1]),
            )
            .build();
        let cost_terms = CostTerms::new(
            [VAR_X],
            alloc::vec![
                Quadratic1CostTerm {
                    z: VecF64::<1>::new(1.0),
                    entity_indices: [0]
                },
                Quadratic1CostTerm {
                    z: VecF64::<1>::new(2.0),
                    entity_indices: [1]
                }
            ],
        );
        let eq_constraints = EqConstraints::new(
            [VAR_X, VAR_X],
            vec![LinearEqConstraint1 {
                lhs: EQ_CONSTRAINT_RHS,
                entity_indices: [0, 1],
            }],
        );

        // illustrate that the initial guess is not feasible
        assert!(
            LinearEqConstraint1::residual(initial_x0, initial_x1, EQ_CONSTRAINT_RHS).norm()
                > EPS_F64
        );

        let solution = optimize_nlls_with_eq_constraints(
            variables,
            alloc::vec![CostFn::new_box((), cost_terms.clone(),)],
            alloc::vec![EqConstraintFn::new_box((), eq_constraints,)],
            OptParams {
                num_iterations: 10,
                initial_lm_damping: EPS_F64,
                parallelize: true,
                solver,
            },
        )
        .unwrap();
        let refined_variables = solution.variables;
        let refined_x = refined_variables.get_members::<VecF64<1>>(VAR_X);

        let x0 = refined_x[0];
        let x1 = refined_x[1];
        approx::assert_abs_diff_eq!(x0[0], 0.5, epsilon = 1e-6);
        approx::assert_abs_diff_eq!(x1[0], 1.5, epsilon = 1e-6);

        // converged solution should satisfy the equality constraint
        approx::assert_abs_diff_eq!(
            LinearEqConstraint1::residual(x0, x1, EQ_CONSTRAINT_RHS)[0],
            0.0,
            epsilon = 1e-6
        );
    }
}

#[test]
fn normalize_opt_tests() {
    for solver in LinearSolverType::indefinite_solvers() {
        LinearEqToyProblem::new().test(solver);
    }
}
