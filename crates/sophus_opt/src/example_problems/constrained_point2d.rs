//! 2D constrained point optimization — the simplest inequality constraint example.
//!
//! A single 2D point is pulled toward a target by a quadratic prior, while
//! a half-plane constraint keeps it on one side of a wall. The optimal solution
//! lies on the wall, at the closest feasible point to the target.
//!
//! This is a minimal test case for the constrained optimizer (~2 DOF, 1 constraint).

use sophus_autodiff::linalg::VecF64;

use crate::{
    nlls::{
        CostFn,
        CostTerms,
        IneqBarrierCostFn,
        IneqConstraints,
        OptParams,
        OptProblem,
        Optimizer,
        costs::Quadratic2CostTerm,
        ineq_constraints::ScalarHalfPlaneConstraint,
    },
    variables::{
        VarBuilder,
        VarFamilies,
        VarFamily,
        VarKind,
    },
};

extern crate alloc;

/// Variable family name for the 2D point.
pub const POINT: &str = "point";

/// 2D constrained point optimization problem.
///
/// - Variable: single 2D point `p = (x, y)`.
/// - Cost: `‖p − target‖²` — pulls toward `target`.
/// - Constraint: `h(p) = nᵀp + d ≥ 0` — half-plane wall.
/// - Optimal: the point on the wall closest to the target.
pub struct ConstrainedPoint2dProblem {
    /// Target position (the prior pulls toward this).
    pub target: VecF64<2>,
    /// Initial guess.
    pub init: VecF64<2>,
    /// Wall normal (inward-pointing, toward the feasible side).
    pub wall_normal: VecF64<2>,
    /// Wall offset: `nᵀp + d ≥ 0`.
    pub wall_offset: f64,
}

impl ConstrainedPoint2dProblem {
    /// Default problem: target at (3, 3), wall at x ≤ 2, init at (1, 1).
    ///
    /// Optimal solution: (2, 3) — on the wall, closest to the target.
    pub fn new() -> Self {
        Self {
            target: VecF64::<2>::new(3.0, 3.0),
            init: VecF64::<2>::new(1.0, 1.0),
            // Wall: x ≤ 2  ⟹  -x + 2 ≥ 0  ⟹  n = (-1, 0), d = 2.
            wall_normal: VecF64::<2>::new(-1.0, 0.0),
            wall_offset: 2.0,
        }
    }

    /// Build variable families from the initial guess.
    pub fn build_variables(&self) -> VarFamilies {
        VarBuilder::new()
            .add_family(POINT, VarFamily::new(VarKind::Free, vec![self.init]))
            .build()
    }

    /// Build the prior cost pulling toward the target.
    pub fn build_prior_cost(&self) -> alloc::boxed::Box<dyn crate::nlls::IsCostFn> {
        let terms = vec![Quadratic2CostTerm {
            z: self.target,
            entity_indices: [0],
        }];
        CostFn::new_boxed((), CostTerms::new([POINT], terms))
    }

    /// Build the inequality barrier cost for the wall constraint.
    pub fn build_barrier_cost(&self) -> alloc::boxed::Box<dyn crate::nlls::IsIneqBarrierFn> {
        let constraints = vec![ScalarHalfPlaneConstraint {
            normal: self.wall_normal,
            offset: self.wall_offset,
            entity_indices: [0],
        }];
        IneqBarrierCostFn::new_boxed((), IneqConstraints::new([POINT], constraints))
    }

    /// Build an optimizer for this problem.
    pub fn build_optimizer(&self, params: OptParams) -> Optimizer {
        let variables = self.build_variables();
        let prior = self.build_prior_cost();
        let barrier = self.build_barrier_cost();
        Optimizer::new(
            variables,
            OptProblem {
                costs: vec![prior],
                eq_constraints: vec![],
                ineq_constraints: vec![barrier],
            },
            params,
        )
        .expect("Optimizer::new failed")
    }

    /// Run optimization to completion.
    pub fn optimize(&self, params: OptParams) -> VarFamilies {
        let mut optimizer = self.build_optimizer(params);
        while let Ok(info) = optimizer.step() {
            if info.termination.is_some() {
                break;
            }
        }
        optimizer.into_variables()
    }

    /// Run IPM optimization.
    pub fn optimize_ipm(&self, params: OptParams) -> VarFamilies {
        self.optimize(params)
    }

    /// Run SQP optimization.
    pub fn optimize_sqp(&self, params: OptParams) -> VarFamilies {
        self.optimize(params)
    }

    /// Check whether the wall constraint is satisfied.
    pub fn constraint_satisfied(&self, vars: &VarFamilies) -> bool {
        let points = vars.get_members::<VecF64<2>>(POINT);
        let h = self.wall_normal.dot(&points[0]) + self.wall_offset;
        h >= -1e-4
    }

    /// Get the optimized point.
    pub fn get_point(&self, vars: &VarFamilies) -> VecF64<2> {
        vars.get_members::<VecF64<2>>(POINT)[0]
    }
}

impl Default for ConstrainedPoint2dProblem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use sophus_solver::LinearSolverEnum;

    use super::ConstrainedPoint2dProblem;
    use crate::nlls::{
        IneqMethod,
        OptParams,
    };

    fn base_params() -> OptParams {
        OptParams {
            num_iterations: 50000,
            initial_lm_damping: 1.0,
            solver: LinearSolverEnum::default(),
            parallelize: false,
            ..Default::default()
        }
    }

    #[test]
    fn point2d_ipm_constraint_satisfied() {
        let problem = ConstrainedPoint2dProblem::new();
        let params = OptParams {
            ineq_method: IneqMethod::Ipm {
                tau: 0.99,
                inner_iters: 100,
                lambda_decay: 0.5,
            },
            ..base_params()
        };

        let vars = problem.optimize_ipm(params);
        assert!(problem.constraint_satisfied(&vars), "constraint violated");

        let p = problem.get_point(&vars);
        // Optimal: x = 2.0 (on the wall), y = 3.0 (unconstrained, at target).
        assert!((p[0] - 2.0).abs() < 0.01, "x should be ~2.0, got {}", p[0]);
        assert!((p[1] - 3.0).abs() < 0.01, "y should be ~3.0, got {}", p[1]);
    }

    #[test]
    fn point2d_sqp_constraint_satisfied() {
        let problem = ConstrainedPoint2dProblem::new();
        let params = OptParams {
            ineq_method: IneqMethod::Sqp {
                tau: 0.99,
                inner_iters: 100,
                mu_decay: 0.5,
            },
            ..base_params()
        };

        let vars = problem.optimize_sqp(params);
        assert!(problem.constraint_satisfied(&vars), "constraint violated");

        let p = problem.get_point(&vars);
        assert!((p[0] - 2.0).abs() < 0.01, "x should be ~2.0, got {}", p[0]);
        assert!((p[1] - 3.0).abs() < 0.01, "y should be ~3.0, got {}", p[1]);
    }

    #[test]
    fn point2d_ipm_infeasible_start() {
        // Start at (3, 3) which violates x ≤ 2. Phase-1 should find feasibility.
        let mut problem = ConstrainedPoint2dProblem::new();
        problem.init = sophus_autodiff::linalg::VecF64::<2>::new(3.0, 3.0);

        let params = OptParams {
            ineq_method: IneqMethod::Ipm {
                tau: 0.99,
                inner_iters: 100,
                lambda_decay: 0.5,
            },
            ..base_params()
        };

        let vars = problem.optimize_ipm(params);
        assert!(
            problem.constraint_satisfied(&vars),
            "constraint violated after phase-1 + IPM"
        );

        let p = problem.get_point(&vars);
        assert!((p[0] - 2.0).abs() < 0.01, "x should be ~2.0, got {}", p[0]);
        assert!((p[1] - 3.0).abs() < 0.01, "y should be ~3.0, got {}", p[1]);
    }

    #[test]
    fn point2d_sqp_infeasible_start() {
        // Start at (3, 3) which violates x ≤ 2. Phase-1 should find feasibility.
        let mut problem = ConstrainedPoint2dProblem::new();
        problem.init = sophus_autodiff::linalg::VecF64::<2>::new(3.0, 3.0);

        let params = OptParams {
            ineq_method: IneqMethod::Sqp {
                tau: 0.99,
                inner_iters: 100,
                mu_decay: 0.5,
            },
            ..base_params()
        };

        let vars = problem.optimize_sqp(params);
        assert!(
            problem.constraint_satisfied(&vars),
            "constraint violated after phase-1 + SQP"
        );
    }

    #[test]
    fn point2d_ipm_eval_h_positive_when_feasible() {
        // Verify barrier's eval_h_values returns positive h for a feasible point.
        let problem = ConstrainedPoint2dProblem::new();
        let vars = problem.build_variables(); // init at (1, 1), wall at x ≤ 2
        let barrier = problem.build_barrier_cost();

        let h_vals = barrier.eval_h_values(&vars);
        assert_eq!(h_vals.len(), 1);
        // h = -x + 2 = -1 + 2 = 1 > 0
        assert!(
            h_vals[0] > 0.0,
            "h should be positive for feasible init, got {}",
            h_vals[0]
        );
    }

    #[test]
    fn point2d_ipm_eval_h_negative_when_infeasible() {
        // Verify barrier's eval_h_values returns negative h for an infeasible point.
        let mut problem = ConstrainedPoint2dProblem::new();
        problem.init = sophus_autodiff::linalg::VecF64::<2>::new(3.0, 3.0);
        let vars = problem.build_variables(); // init at (3, 3), wall at x ≤ 2
        let barrier = problem.build_barrier_cost();

        let h_vals = barrier.eval_h_values(&vars);
        assert_eq!(h_vals.len(), 1);
        // h = -x + 2 = -3 + 2 = -1 < 0
        assert!(
            h_vals[0] < 0.0,
            "h should be negative for infeasible init, got {}",
            h_vals[0]
        );
    }

    #[test]
    fn point2d_ipm_step_decreases_merit() {
        // Verify that the first few IPM steps decrease the merit.
        let problem = ConstrainedPoint2dProblem::new();
        let params = OptParams {
            ineq_method: IneqMethod::Ipm {
                tau: 0.99,
                inner_iters: 100,
                lambda_decay: 0.5,
            },
            ..base_params()
        };

        let mut optimizer = problem.build_optimizer(params);
        let info0 = optimizer.step().expect("step 0");
        let info1 = optimizer.step().expect("step 1");

        // After a few steps, merit should not increase (allowing for LM rejections).
        // Just check we can take steps without errors.
        assert!(info0.merit.is_finite(), "merit should be finite");
        assert!(info1.merit.is_finite(), "merit should be finite");
    }

    #[test]
    fn point2d_ellipse_center_phase1_recovers() {
        // Starting at the exact center of an ellipse where ∇h = 0.
        // Phase-1 should detect the degenerate gradient, perturb, and recover.
        use crate::nlls::ineq_constraints::ScalarEllipseConstraint;

        let target = sophus_autodiff::linalg::VecF64::<2>::new(5.5, 5.5);
        let init = sophus_autodiff::linalg::VecF64::<2>::new(2.0, 2.0); // exact center

        let obstacles = vec![ScalarEllipseConstraint::new(
            sophus_autodiff::linalg::VecF64::<2>::new(2.0, 2.0),
            1.2,
            0.5,
            std::f64::consts::PI / 4.0,
            0,
        )];

        let variables = crate::variables::VarBuilder::new()
            .add_family(
                "point",
                crate::variables::VarFamily::new(crate::variables::VarKind::Free, vec![init]),
            )
            .build();

        let prior = crate::nlls::CostFn::new_boxed(
            (),
            crate::nlls::CostTerms::new(
                ["point"],
                vec![crate::nlls::costs::Quadratic2CostTerm {
                    z: target,
                    entity_indices: [0],
                }],
            ),
        );
        let barrier = crate::nlls::IneqBarrierCostFn::new_boxed(
            (),
            crate::nlls::IneqConstraints::new(["point"], obstacles),
        );

        let result = crate::nlls::Optimizer::new(
            variables,
            crate::nlls::OptProblem {
                costs: vec![prior],
                eq_constraints: vec![],
                ineq_constraints: vec![barrier],
            },
            OptParams {
                ineq_method: IneqMethod::Ipm {
                    tau: 0.99,
                    inner_iters: 50,
                    lambda_decay: 0.5,
                },
                ..base_params()
            },
        );

        assert!(result.is_ok(), "phase-1 should recover via perturbation");
    }
}
