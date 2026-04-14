//! Circle + obstacle constrained point optimization.
//!
//! A 2D point is pulled toward a target by a quadratic prior, while constrained
//! to lie ON a circle (equality: `‖p − c_circle‖ = r_circle`) and OUTSIDE an
//! obstacle disk (inequality: `‖p − c_obs‖² − r_obs² ≥ 0`).
//!
//! The optimal solution is the point on the constraint circle closest to the
//! target that is also outside the obstacle.

use sophus_autodiff::linalg::VecF64;

use crate::{
    nlls::{
        CostFn,
        CostTerms,
        EqConstraintFn,
        EqConstraints,
        IneqBarrierCostFn,
        IneqConstraints,
        OptParams,
        OptProblem,
        Optimizer,
        costs::Quadratic2CostTerm,
        eq_constraints::ScalarCircleEqConstraint,
        ineq_constraints::ScalarCircleConstraint,
    },
    variables::{
        VarBuilder,
        VarFamilies,
        VarFamily,
        VarKind,
    },
};

extern crate alloc;

/// Variable family name.
pub const POINT: &str = "point";

/// Circle + obstacle constrained point problem.
pub struct CircleObstacleProblem {
    /// Target position.
    pub target: VecF64<2>,
    /// Initial guess.
    pub init: VecF64<2>,
    /// Constraint circle center.
    pub circle_center: VecF64<2>,
    /// Constraint circle radius.
    pub circle_radius: f64,
    /// Obstacle center.
    pub obstacle_center: VecF64<2>,
    /// Obstacle radius.
    pub obstacle_radius: f64,
}

impl CircleObstacleProblem {
    /// Default problem:
    /// - Target at (4, 3)
    /// - Constraint circle: center (2, 2), radius 2
    /// - Obstacle: center (3.5, 3.5), radius 0.5
    /// - Init on the circle at (2, 4) (top of circle, outside obstacle)
    pub fn new() -> Self {
        Self {
            target: VecF64::<2>::new(4.0, 3.0),
            init: VecF64::<2>::new(2.0, 4.0),
            circle_center: VecF64::<2>::new(2.0, 2.0),
            circle_radius: 2.0,
            obstacle_center: VecF64::<2>::new(3.5, 3.5),
            obstacle_radius: 0.5,
        }
    }

    /// Build variable families.
    pub fn build_variables(&self) -> VarFamilies {
        VarBuilder::new()
            .add_family(POINT, VarFamily::new(VarKind::Free, vec![self.init]))
            .build()
    }

    /// Build the quadratic prior cost.
    pub fn build_prior_cost(&self) -> alloc::boxed::Box<dyn crate::nlls::IsCostFn> {
        CostFn::new_boxed(
            (),
            CostTerms::new(
                [POINT],
                vec![Quadratic2CostTerm {
                    z: self.target,
                    entity_indices: [0],
                }],
            ),
        )
    }

    /// Build the circle equality constraint.
    pub fn build_eq_constraint(&self) -> alloc::boxed::Box<dyn crate::nlls::IsEqConstraintsFn> {
        EqConstraintFn::new_boxed(
            (),
            EqConstraints::new(
                [POINT],
                vec![ScalarCircleEqConstraint {
                    center: self.circle_center,
                    radius: self.circle_radius,
                    entity_indices: [0],
                }],
            ),
        )
    }

    /// Build barrier cost for the obstacle avoidance inequality constraint.
    pub fn build_obstacle_barrier(&self) -> alloc::boxed::Box<dyn crate::nlls::IsIneqBarrierFn> {
        IneqBarrierCostFn::new_boxed(
            (),
            IneqConstraints::new(
                [POINT],
                vec![ScalarCircleConstraint {
                    center: self.obstacle_center,
                    radius: self.obstacle_radius,
                    entity_indices: [0],
                }],
            ),
        )
    }

    /// Build an optimizer with equality (circle) + inequality (obstacle) constraints.
    pub fn build_optimizer(&self, params: OptParams) -> Optimizer {
        Optimizer::new(
            self.build_variables(),
            OptProblem {
                costs: vec![self.build_prior_cost()],
                eq_constraints: vec![self.build_eq_constraint()],
                ineq_constraints: vec![self.build_obstacle_barrier()],
            },
            params,
        )
        .expect("Optimizer::new failed")
    }

    /// Run to completion.
    pub fn optimize(&self, params: OptParams) -> VarFamilies {
        let mut optimizer = self.build_optimizer(params);
        while let Ok(info) = optimizer.step() {
            if info.termination.is_some() {
                break;
            }
        }
        optimizer.into_variables()
    }

    /// Get the optimized point.
    pub fn get_point(&self, vars: &VarFamilies) -> VecF64<2> {
        vars.get_members::<VecF64<2>>(POINT)[0]
    }

    /// Check circle equality constraint: `| ‖p − c‖ − r | < tol`.
    pub fn circle_satisfied(&self, vars: &VarFamilies, tol: f64) -> bool {
        let p = self.get_point(vars);
        let dist = (p - self.circle_center).norm();
        (dist - self.circle_radius).abs() < tol
    }

    /// Check obstacle inequality constraint: `‖p − c_obs‖² − r_obs² ≥ -tol`.
    pub fn obstacle_satisfied(&self, vars: &VarFamilies, tol: f64) -> bool {
        let p = self.get_point(vars);
        let d = p - self.obstacle_center;
        d.dot(&d) - self.obstacle_radius * self.obstacle_radius >= -tol
    }
}

impl Default for CircleObstacleProblem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use sophus_solver::LinearSolverEnum;

    use super::*;
    use crate::nlls::{
        IneqMethod,
        OptParams,
    };

    fn base_params() -> OptParams {
        OptParams {
            num_iterations: 50000,
            initial_lm_damping: 1.0,
            solver: LinearSolverEnum::DenseLdlt(sophus_solver::ldlt::DenseLdlt::default()),
            parallelize: false,
            ..Default::default()
        }
    }

    fn dense_ldlt_params(ineq_method: IneqMethod) -> OptParams {
        OptParams {
            num_iterations: 50000,
            initial_lm_damping: 1.0,
            solver: LinearSolverEnum::DenseLdlt(sophus_solver::ldlt::DenseLdlt::default()),
            parallelize: false,
            ineq_method,
            ..Default::default()
        }
    }

    fn block_sparse_ldlt_params(ineq_method: IneqMethod) -> OptParams {
        OptParams {
            num_iterations: 50000,
            initial_lm_damping: 1.0,
            solver: LinearSolverEnum::BlockSparseLdlt(
                sophus_solver::ldlt::BlockSparseLdlt::default(),
            ),
            parallelize: false,
            ineq_method,
            ..Default::default()
        }
    }

    #[test]
    fn circle_obstacle_ipm_converges() {
        let problem = CircleObstacleProblem::new();
        let params = OptParams {
            ineq_method: IneqMethod::Ipm {
                tau: 0.99,
                inner_iters: 100,
                lambda_decay: 0.5,
            },
            ..base_params()
        };

        let vars = problem.optimize(params);
        let p = problem.get_point(&vars);
        println!("IPM result: ({:.4}, {:.4})", p[0], p[1]);

        assert!(
            problem.circle_satisfied(&vars, 0.01),
            "circle constraint violated: dist={:.4}, want={:.4}",
            (p - problem.circle_center).norm(),
            problem.circle_radius,
        );
        assert!(
            problem.obstacle_satisfied(&vars, 0.01),
            "obstacle constraint violated",
        );
    }

    #[test]
    fn circle_obstacle_sqp_converges() {
        let problem = CircleObstacleProblem::new();
        let params = OptParams {
            ineq_method: IneqMethod::Sqp {
                tau: 0.99,
                inner_iters: 100,
                mu_decay: 0.5,
            },
            ..base_params()
        };

        let vars = problem.optimize(params);
        let p = problem.get_point(&vars);
        println!("SQP result: ({:.4}, {:.4})", p[0], p[1]);

        assert!(
            problem.circle_satisfied(&vars, 0.01),
            "circle constraint violated: dist={:.4}, want={:.4}",
            (p - problem.circle_center).norm(),
            problem.circle_radius,
        );
        assert!(
            problem.obstacle_satisfied(&vars, 0.01),
            "obstacle constraint violated",
        );
    }

    #[test]
    fn circle_obstacle_dense_ldlt_ipm_converges() {
        let problem = CircleObstacleProblem::new();
        let params = dense_ldlt_params(IneqMethod::Ipm {
            tau: 0.99,
            inner_iters: 100,
            lambda_decay: 0.5,
        });

        let vars = problem.optimize(params);
        let p = problem.get_point(&vars);
        println!("Dense LDLᵀ IPM result: ({:.4}, {:.4})", p[0], p[1]);

        assert!(
            problem.circle_satisfied(&vars, 0.01),
            "circle constraint violated: dist={:.4}, want={:.4}",
            (p - problem.circle_center).norm(),
            problem.circle_radius,
        );
        assert!(
            problem.obstacle_satisfied(&vars, 0.01),
            "obstacle constraint violated",
        );
    }

    #[test]
    fn circle_obstacle_dense_ldlt_sqp_converges() {
        let problem = CircleObstacleProblem::new();
        let params = dense_ldlt_params(IneqMethod::Sqp {
            tau: 0.99,
            inner_iters: 100,
            mu_decay: 0.5,
        });

        let vars = problem.optimize(params);
        let p = problem.get_point(&vars);
        println!("Dense LDLᵀ SQP result: ({:.4}, {:.4})", p[0], p[1]);

        assert!(
            problem.circle_satisfied(&vars, 0.01),
            "circle constraint violated: dist={:.4}, want={:.4}",
            (p - problem.circle_center).norm(),
            problem.circle_radius,
        );
        assert!(
            problem.obstacle_satisfied(&vars, 0.01),
            "obstacle constraint violated",
        );
    }

    #[test]
    fn circle_obstacle_block_sparse_ldlt_ipm_converges() {
        let problem = CircleObstacleProblem::new();
        let params = block_sparse_ldlt_params(IneqMethod::Ipm {
            tau: 0.99,
            inner_iters: 100,
            lambda_decay: 0.5,
        });

        let vars = problem.optimize(params);
        let p = problem.get_point(&vars);
        println!(
            "Block-sparse Dense LDLᵀ IPM result: ({:.4}, {:.4})",
            p[0], p[1]
        );

        assert!(
            problem.circle_satisfied(&vars, 0.01),
            "circle constraint violated: dist={:.4}, want={:.4}",
            (p - problem.circle_center).norm(),
            problem.circle_radius,
        );
        assert!(
            problem.obstacle_satisfied(&vars, 0.01),
            "obstacle constraint violated",
        );
    }

    #[test]
    fn circle_obstacle_block_sparse_ldlt_sqp_converges() {
        let problem = CircleObstacleProblem::new();
        let params = block_sparse_ldlt_params(IneqMethod::Sqp {
            tau: 0.99,
            inner_iters: 100,
            mu_decay: 0.5,
        });

        let vars = problem.optimize(params);
        let p = problem.get_point(&vars);
        println!(
            "Block-sparse Dense LDLᵀ SQP result: ({:.4}, {:.4})",
            p[0], p[1]
        );

        assert!(
            problem.circle_satisfied(&vars, 0.01),
            "circle constraint violated: dist={:.4}, want={:.4}",
            (p - problem.circle_center).norm(),
            problem.circle_radius,
        );
        assert!(
            problem.obstacle_satisfied(&vars, 0.01),
            "obstacle constraint violated",
        );
    }
}
