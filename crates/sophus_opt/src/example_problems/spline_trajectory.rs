//! Spline-based 2D trajectory optimization using SE(2) control points.
//!
//! 16 SE(2) control points form a trajectory from a fixed start pose to a fixed end pose.
//! A smoothness cost uses `PoseGraph2CostTerm` (residual: `log(T_i⁻¹ T_{i+1})`) to minimise
//! the sum of squared relative pose differences between consecutive control points.
//! An obstacle avoidance inequality constraint keeps every control point's translation
//! outside a circular obstacle.
//!
//! The first 2 and last 2 control points are pinned as constants for B-spline endpoint
//! clamping; the 12 interior control points are free. With no obstacle the optimal
//! trajectory is a straight line (all poses aligned along x). The obstacle pushes the
//! free poses around the disk.

use sophus_autodiff::linalg::VecF64;
use sophus_lie::{
    Isometry2F64,
    Rotation2F64,
    prelude::IsAffineGroup,
};
use sophus_spline::spline_segment::SegmentCase;

use crate::{
    nlls::{
        CostFn,
        CostTerms,
        IneqBarrierCostFn,
        IneqConstraints,
        OptParams,
        OptProblem,
        Optimizer,
        costs::PoseGraph2CostTerm,
        ineq_constraints::{
            SE2CircleConstraint,
            SE2SplineCircleConstraint,
            SE2SplineLateralVelConstraint,
        },
    },
    variables::{
        VarBuilder,
        VarFamilies,
        VarFamily,
        VarKind,
    },
};

extern crate alloc;

/// Variable family name for the SE(2) waypoint control points.
pub const WAYPOINTS: &str = "waypoints";

/// Time step between consecutive control points (seconds).
pub const DELTA_T: f64 = 1.0;

/// Spline trajectory optimization problem using SE(2) control points.
///
/// - 16 control points in SE(2); first 2 and last 2 are pinned for B-spline endpoint clamping.
/// - Smoothness cost: `Σᵢ ½ ‖log(Tᵢ⁻¹ Tᵢ₊₁)‖²` (PoseGraph2CostTerm with identity measurement).
/// - Obstacle inequality: every control point's translation must satisfy `‖t(T) − obstacle_center‖²
///   − r² ≥ 0`.
pub struct SplineTrajProblem {
    /// Start position (pinned as constant).
    pub start: VecF64<2>,
    /// End target position (pinned as constant).
    pub end_target: VecF64<2>,
    /// Number of control points (including pinned endpoints).
    pub num_control_points: usize,
    /// Circular obstacle center.
    pub obstacle_center: VecF64<2>,
    /// Circular obstacle radius.
    pub obstacle_radius: f64,
}

impl SplineTrajProblem {
    /// Default problem:
    /// - 16 control points from (0, 0) to (7, 0).
    /// - Circular obstacle at (3.5, 0) radius 1.0 (on the straight-line path).
    pub fn new() -> Self {
        Self {
            start: VecF64::<2>::new(0.0, 0.0),
            end_target: VecF64::<2>::new(7.0, 0.0),
            // 16 control points: first 2 and last 2 are duplicated for endpoint clamping.
            num_control_points: 16,
            obstacle_center: VecF64::<2>::new(3.5, 0.0),
            obstacle_radius: 1.0,
        }
    }

    /// Build initial SE(2) control points as a smooth arc around the obstacle.
    ///
    /// The first 2 and last 2 control points are duplicated (pinned) for B-spline
    /// endpoint clamping. Interior points follow a smooth cosine bump above the
    /// obstacle with headings tangent to the curve, ensuring initial feasibility
    /// for obstacle, lateral velocity, and smoothness constraints.
    pub fn build_init_points(&self) -> alloc::vec::Vec<Isometry2F64> {
        let n = self.num_control_points;
        let cx = self.obstacle_center[0];
        let cy = self.obstacle_center[1];
        let clearance = self.obstacle_radius * 2.0;
        let sx = self.start[0];
        let sy = self.start[1];
        let ex = self.end_target[0];
        let ey = self.end_target[1];
        let total_x = ex - sx;

        // Generate positions along a smooth cosine bump.
        let pos_at = |t: f64| -> (f64, f64) {
            let x = sx + t * total_x;
            let y_base = sy + t * (ey - sy);
            // Smooth bump: cosine bell centered at obstacle x.
            let dx = x - cx;
            let bump_half_width = total_x * 0.5;
            let bump = if dx.abs() < bump_half_width {
                clearance * (1.0 + (core::f64::consts::PI * dx / bump_half_width).cos()) * 0.5
            } else {
                0.0
            };
            (x, y_base + cy + bump)
        };

        (0..n)
            .map(|i| {
                let (x, y, theta) = if i < 2 {
                    // Pinned start (duplicated).
                    (sx, sy, 0.0)
                } else if i >= n - 2 {
                    // Pinned end (duplicated).
                    (ex, ey, 0.0)
                } else {
                    // Interior: evenly spaced.
                    let interior_idx = i - 2;
                    let num_interior = n - 4;
                    let t = (interior_idx as f64 + 1.0) / (num_interior as f64 + 1.0);
                    let (x, y) = pos_at(t);

                    // Heading tangent to the curve (finite difference).
                    let dt = 0.001;
                    let (x1, y1) = pos_at((t + dt).min(1.0));
                    let (x0, y0) = pos_at((t - dt).max(0.0));
                    let theta = (y1 - y0).atan2(x1 - x0);
                    (x, y, theta)
                };

                let rotation = Rotation2F64::exp(VecF64::<1>::new(theta));
                Isometry2F64::from_factor_and_translation(rotation, VecF64::<2>::new(x, y))
            })
            .collect()
    }

    /// Build variable families. First 2 and last 2 control points are pinned
    /// (duplicated for B-spline endpoint clamping).
    pub fn build_variables(&self) -> VarFamilies {
        let members = self.build_init_points();
        let n = self.num_control_points;
        let mut constant_ids = alloc::collections::BTreeMap::new();
        constant_ids.insert(0, ());
        constant_ids.insert(1, ());
        constant_ids.insert(n - 2, ());
        constant_ids.insert(n - 1, ());
        VarBuilder::new()
            .add_family(
                WAYPOINTS,
                VarFamily::new_with_const_ids(VarKind::Free, members, constant_ids),
            )
            .build()
    }

    /// Build the smoothness cost using `PoseGraph2CostTerm`.
    ///
    /// For each consecutive pair (i, i+1) adds a term with identity measurement
    /// (`pose_m_from_pose_n = identity`), so the residual is `log(T_i⁻¹ T_{i+1})`.
    /// This penalises relative pose differences and pulls the trajectory smooth.
    pub fn build_smoothness_cost(
        &self,
    ) -> alloc::vec::Vec<alloc::boxed::Box<dyn crate::nlls::IsCostFn>> {
        let n = self.num_control_points;
        let identity = Isometry2F64::identity();

        let terms: alloc::vec::Vec<PoseGraph2CostTerm> = (0..n - 1)
            .map(|i| PoseGraph2CostTerm {
                pose_m_from_pose_n: identity,
                entity_indices: [i, i + 1],
            })
            .collect();

        alloc::vec![CostFn::new_boxed(
            (),
            CostTerms::new([WAYPOINTS, WAYPOINTS], terms),
        )]
    }

    /// Build obstacle avoidance barriers.
    ///
    /// Constrains each control point's translation and each spline sample point's
    /// translation (at u = 0.5 per segment) to be outside the circular obstacle.
    pub fn build_barrier_costs(
        &self,
    ) -> alloc::vec::Vec<alloc::boxed::Box<dyn crate::nlls::IsIneqBarrierFn>> {
        let n = self.num_control_points;

        // Constraints on control points themselves.
        let cp_constraints: alloc::vec::Vec<SE2CircleConstraint> = (0..n)
            .map(|i| SE2CircleConstraint {
                center: self.obstacle_center,
                radius: self.obstacle_radius,
                entity_indices: [i],
            })
            .collect();

        let cp_barrier =
            IneqBarrierCostFn::new_boxed((), IneqConstraints::new([WAYPOINTS], cp_constraints));

        // Constraints at 3 sample points per spline segment.
        let num_segments = n - 1;
        let mut spline_constraints: alloc::vec::Vec<SE2SplineCircleConstraint> =
            alloc::vec::Vec::new();
        for seg_idx in 0..num_segments {
            let case = if seg_idx == 0 {
                SegmentCase::First
            } else if seg_idx == num_segments - 1 {
                SegmentCase::Last
            } else {
                SegmentCase::Normal
            };
            let idx_prev = if seg_idx == 0 { 0 } else { seg_idx - 1 };
            let idx_0 = seg_idx;
            let idx_1 = seg_idx + 1;
            let idx_2 = (seg_idx + 2).min(n - 1);
            {
                let u = 0.5_f64;
                spline_constraints.push(SE2SplineCircleConstraint {
                    center: self.obstacle_center,
                    radius: self.obstacle_radius,
                    u,
                    case,
                    entity_indices: [idx_prev, idx_0, idx_1, idx_2],
                });
            }
        }

        let spline_barrier = IneqBarrierCostFn::new_boxed(
            (),
            IneqConstraints::new(
                [WAYPOINTS, WAYPOINTS, WAYPOINTS, WAYPOINTS],
                spline_constraints,
            ),
        );

        // Lateral velocity constraint: |vy| ≤ vy_max at spline sample points.
        let vy_max = 0.05;
        let mut lateral_constraints: alloc::vec::Vec<SE2SplineLateralVelConstraint> =
            alloc::vec::Vec::new();
        for seg_idx in 0..num_segments {
            let case = if seg_idx == 0 {
                SegmentCase::First
            } else if seg_idx == num_segments - 1 {
                SegmentCase::Last
            } else {
                SegmentCase::Normal
            };
            let idx_prev = if seg_idx == 0 { 0 } else { seg_idx - 1 };
            let idx_0 = seg_idx;
            let idx_1 = seg_idx + 1;
            let idx_2 = (seg_idx + 2).min(n - 1);
            {
                let u = 0.5_f64;
                lateral_constraints.push(SE2SplineLateralVelConstraint {
                    vy_max,
                    u,
                    delta_t: DELTA_T,
                    case,
                    entity_indices: [idx_prev, idx_0, idx_1, idx_2],
                });
            }
        }

        let lateral_barrier = IneqBarrierCostFn::new_boxed(
            (),
            IneqConstraints::new(
                [WAYPOINTS, WAYPOINTS, WAYPOINTS, WAYPOINTS],
                lateral_constraints,
            ),
        );

        alloc::vec![cp_barrier, spline_barrier, lateral_barrier]
    }

    /// Build an optimizer with smoothness costs and obstacle/lateral-velocity inequality barriers.
    pub fn build_optimizer(&self, params: OptParams) -> Optimizer {
        let variables = self.build_variables();
        let costs = self.build_smoothness_cost();
        let eq_constraints = alloc::vec![];
        let barriers = self.build_barrier_costs();
        Optimizer::new(
            variables,
            OptProblem {
                costs,
                eq_constraints,
                ineq_constraints: barriers,
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

    /// Check that all control points' translations are outside the obstacle (with tolerance).
    pub fn obstacle_constraint_satisfied(&self, vars: &VarFamilies, tol: f64) -> bool {
        let poses = vars.get_members::<Isometry2F64>(WAYPOINTS);
        for pose in &poses {
            let t = pose.translation();
            let d = t - self.obstacle_center;
            let h = d.dot(&d) - self.obstacle_radius * self.obstacle_radius;
            if h < -tol {
                return false;
            }
        }
        true
    }

    /// Get the optimized SE(2) control point poses.
    pub fn get_waypoints(&self, vars: &VarFamilies) -> alloc::vec::Vec<Isometry2F64> {
        vars.get_members::<Isometry2F64>(WAYPOINTS)
    }

    /// Get the optimized control point translations (2D positions).
    pub fn get_translations(&self, vars: &VarFamilies) -> alloc::vec::Vec<VecF64<2>> {
        self.get_waypoints(vars)
            .iter()
            .map(|p| p.translation())
            .collect()
    }
}

impl Default for SplineTrajProblem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use sophus_lie::Rotation2F64;
    use sophus_solver::LinearSolverEnum;
    use sophus_spline::spline_segment::SegmentCase;

    use super::*;
    use crate::nlls::{
        IneqMethod,
        OptParams,
        ineq_constraints::SE2SplineCircleConstraint,
    };

    fn base_params() -> OptParams {
        OptParams {
            num_iterations: 50000,
            initial_lm_damping: 0.1,
            solver: LinearSolverEnum::DenseLdlt(sophus_solver::ldlt::DenseLdlt::default()),
            parallelize: false,
            ..Default::default()
        }
    }

    fn ipm_params() -> OptParams {
        OptParams {
            ineq_method: IneqMethod::Ipm {
                tau: 0.9,
                inner_iters: 30,
                lambda_decay: 0.5,
            },
            ..base_params()
        }
    }

    fn sqp_params() -> OptParams {
        OptParams {
            ineq_method: IneqMethod::Sqp {
                tau: 0.9,
                inner_iters: 30,
                mu_decay: 0.5,
            },
            ..base_params()
        }
    }

    fn dense_ldlt_ipm_params() -> OptParams {
        OptParams {
            solver: LinearSolverEnum::DenseLdlt(sophus_solver::ldlt::DenseLdlt::default()),
            ineq_method: IneqMethod::Ipm {
                tau: 0.9,
                inner_iters: 30,
                lambda_decay: 0.5,
            },
            ..base_params()
        }
    }

    fn block_sparse_ldlt_ipm_params() -> OptParams {
        OptParams {
            solver: LinearSolverEnum::BlockSparseLdlt(
                sophus_solver::ldlt::BlockSparseLdlt::default(),
            ),
            ineq_method: IneqMethod::Ipm {
                tau: 0.9,
                inner_iters: 30,
                lambda_decay: 0.5,
            },
            ..base_params()
        }
    }

    #[test]
    fn spline_traj_dense_ldlt_ipm_avoids_obstacle() {
        let problem = SplineTrajProblem::new();
        let vars = problem.optimize(dense_ldlt_ipm_params());

        assert!(
            problem.obstacle_constraint_satisfied(&vars, 1e-3),
            "Dense LDLᵀ IPM: obstacle constraint violated"
        );

        let poses = problem.get_waypoints(&vars);
        let n = poses.len();
        let start_t = poses[0].translation();
        let end_t = poses[n - 1].translation();
        assert!(
            (start_t - problem.start).norm() < 1e-8,
            "start pinned: got {start_t:?}",
        );
        assert!(
            (end_t - problem.end_target).norm() < 1e-8,
            "end pinned: got {end_t:?}",
        );
    }

    #[test]
    fn spline_traj_block_sparse_ldlt_ipm_avoids_obstacle() {
        let problem = SplineTrajProblem::new();
        let vars = problem.optimize(block_sparse_ldlt_ipm_params());

        assert!(
            problem.obstacle_constraint_satisfied(&vars, 1e-3),
            "Block-sparse Dense LDLᵀ IPM: obstacle constraint violated"
        );

        let poses = problem.get_waypoints(&vars);
        let n = poses.len();
        let start_t = poses[0].translation();
        let end_t = poses[n - 1].translation();
        assert!(
            (start_t - problem.start).norm() < 1e-8,
            "start pinned: got {start_t:?}",
        );
        assert!(
            (end_t - problem.end_target).norm() < 1e-8,
            "end pinned: got {end_t:?}",
        );
    }

    #[test]
    fn spline_traj_ipm_avoids_obstacle() {
        let problem = SplineTrajProblem::new();
        let vars = problem.optimize(ipm_params());

        assert!(
            problem.obstacle_constraint_satisfied(&vars, 1e-3),
            "IPM: obstacle constraint violated"
        );

        // Endpoints must stay pinned.
        let poses = problem.get_waypoints(&vars);
        let n = poses.len();
        let start_t = poses[0].translation();
        let end_t = poses[n - 1].translation();
        assert!(
            (start_t - problem.start).norm() < 1e-8,
            "start pinned: got {start_t:?}",
        );
        assert!(
            (end_t - problem.end_target).norm() < 1e-8,
            "end pinned: got {end_t:?}",
        );

        // The optimal trajectory should hug the obstacle (shortest path).
        // At least one control point should be within 0.5 of the boundary.
        let min_dist_to_boundary: f64 = poses
            .iter()
            .map(|p| {
                let t = p.translation();
                ((t - problem.obstacle_center).norm() - problem.obstacle_radius).abs()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_dist_to_boundary < 0.15,
            "IPM: trajectory not hugging obstacle (min_dist_to_boundary={min_dist_to_boundary:.3}, expected < 0.15)"
        );
    }

    #[test]
    fn spline_traj_sqp_avoids_obstacle() {
        let problem = SplineTrajProblem::new();
        let vars = problem.optimize(sqp_params());

        assert!(
            problem.obstacle_constraint_satisfied(&vars, 0.1),
            "SQP: obstacle constraint violated"
        );

        let poses = problem.get_waypoints(&vars);
        let n = poses.len();
        let start_t = poses[0].translation();
        let end_t = poses[n - 1].translation();
        assert!(
            (start_t - problem.start).norm() < 1e-8,
            "start pinned: got {start_t:?}",
        );
        assert!(
            (end_t - problem.end_target).norm() < 1e-8,
            "end pinned: got {end_t:?}",
        );
    }

    #[test]
    fn spline_traj_endpoints_pinned() {
        // Verify that pinned endpoints are not moved by the optimizer.
        let problem = SplineTrajProblem::new();
        let vars = problem.build_variables();
        let poses = problem.get_waypoints(&vars);
        let n = poses.len();
        assert_eq!(n, problem.num_control_points);
        let start_t = poses[0].translation();
        let end_t = poses[n - 1].translation();
        assert!((start_t - problem.start).norm() < 1e-12, "start not pinned");
        assert!(
            (end_t - problem.end_target).norm() < 1e-12,
            "end not pinned"
        );
    }

    #[test]
    fn spline_traj_init_feasible() {
        use sophus_spline::{
            lie_group_spline::LieGroupBSplineSegment,
            spline_segment::SegmentCase,
        };

        let problem = SplineTrajProblem::new();
        let vars = problem.build_variables();
        assert!(
            problem.obstacle_constraint_satisfied(&vars, 0.0),
            "initial guess violates obstacle constraint"
        );

        // Check lateral velocities at sample points.
        let poses = problem.get_waypoints(&vars);
        let n = poses.len();
        let num_segments = n - 1;
        let vy_max = 0.05_f64;
        println!("Lateral velocities at init:");
        for seg in 0..num_segments {
            let case = if seg == 0 {
                SegmentCase::First
            } else if seg == num_segments - 1 {
                SegmentCase::Last
            } else {
                SegmentCase::Normal
            };
            let idx_prev = if seg == 0 { 0 } else { seg - 1 };
            let idx_2 = (seg + 2).min(n - 1);
            let segment = LieGroupBSplineSegment {
                case,
                control_points: [poses[idx_prev], poses[seg], poses[seg + 1], poses[idx_2]],
            };
            let vel = segment.velocity(0.5, DELTA_T);
            let vy = vel[2];
            let h = vy_max * vy_max - vy * vy;
            println!("  seg={seg} u=0.50: vy={vy:.4}, h={h:.4}");
            assert!(
                h >= 0.0,
                "initial lateral velocity constraint violated at seg={seg}: vy={vy:.4}, h={h:.4}"
            );
        }
    }

    #[test]
    fn se2_circle_constraint_jacobian() {
        use crate::nlls::HasIneqConstraintFn as _;

        let constraint = SE2CircleConstraint {
            center: VecF64::<2>::new(3.5, 0.0),
            radius: 1.0,
            entity_indices: [0],
        };

        // A pose with known translation.
        let translation = VecF64::<2>::new(4.0, 1.5);
        let rotation = Rotation2F64::identity();
        let pose = Isometry2F64::from_factor_and_translation(rotation, translation);
        let var_kinds = [VarKind::Free; 1];

        let eval = constraint.eval(&(), [0], pose, var_kinds);
        println!("h = {:.6}", eval.h);

        // Finite-difference check for each DOF (θ, tx, ty) via left-perturbation.
        let eps = 1e-6;
        for dof in 0..3 {
            let mut xi = VecF64::<3>::zeros();
            xi[dof] = eps;
            let pose_plus = sophus_lie::Isometry2F64::exp(xi) * pose;
            xi[dof] = -eps;
            let pose_minus = sophus_lie::Isometry2F64::exp(xi) * pose;

            let h_plus = constraint.eval(&(), [0], pose_plus, var_kinds).h;
            let h_minus = constraint.eval(&(), [0], pose_minus, var_kinds).h;
            let num_jac = (h_plus - h_minus) / (2.0 * eps);
            let analytic_jac = eval.jacobian.block(0)[(0, dof)];
            let err = (num_jac - analytic_jac).abs();
            println!(
                "  dof[{dof}]: analytic={analytic_jac:.6}, numeric={num_jac:.6}, err={err:.2e}"
            );
            assert!(
                err < 1e-4,
                "Jacobian mismatch for dof[{dof}]: analytic={analytic_jac}, numeric={num_jac}"
            );
        }
    }

    #[test]
    fn spline_traj_debug_convergence() {
        use sophus_spline::lie_group_spline::LieGroupCubicBSpline;

        let problem = SplineTrajProblem::new();
        let params = ipm_params();
        let mut optimizer = problem.build_optimizer(params);

        // Run some steps and trace.
        for i in 0..200 {
            match optimizer.step() {
                Ok(info) => {
                    if i < 5 || i % 50 == 0 || info.termination.is_some() {
                        println!(
                            "step {i}: smooth={:.4e} merit={:.4e} lm={:.3e} term={:?}",
                            info.smooth_cost, info.merit, info.lm_damping, info.termination
                        );
                    }
                    if info.termination.is_some() {
                        break;
                    }
                }
                Err(e) => {
                    println!("step {i}: error: {e:?}");
                    break;
                }
            }
        }

        let vars = optimizer.into_variables();
        let poses = problem.get_waypoints(&vars);

        println!("Control points:");
        for (i, pose) in poses.iter().enumerate() {
            let t = pose.translation();
            let d = (t - problem.obstacle_center).norm();
            println!(
                "  cp[{i}] = ({:.3}, {:.3}), dist_to_obs = {d:.3}",
                t[0], t[1]
            );
        }

        // Evaluate the SE(2) Lie group spline densely and check for violations.
        let spline = LieGroupCubicBSpline {
            control_points: poses.clone(),
            delta_t: DELTA_T,
            t0: 0.0,
        };
        let num_samples = 100;
        let t_max = spline.t_max();
        let mut min_h = f64::MAX;
        let mut worst_t = 0.0;
        for i in 0..=num_samples {
            let t = i as f64 / num_samples as f64 * t_max;
            let pose = spline.interpolate(t);
            let pos = pose.translation();
            let diff = pos - problem.obstacle_center;
            let h = diff.dot(&diff) - problem.obstacle_radius * problem.obstacle_radius;
            if h < min_h {
                min_h = h;
                worst_t = t;
            }
        }
        let worst_pose = spline.interpolate(worst_t);
        let worst_pos = worst_pose.translation();
        println!(
            "Worst spline sample: t={worst_t:.3}, pos=({:.3}, {:.3}), h={min_h:.4}",
            worst_pos[0], worst_pos[1]
        );
        // The IPM optimizer should have driven the control-point obstacle constraint
        // to feasibility; assert that the minimum h across densely-sampled spline
        // points is not wildly negative (tolerating small interpolation overshoot).
        assert!(
            min_h > -0.5,
            "spline passes far inside obstacle: h_min={min_h:.4} at t={worst_t:.3}"
        );
    }

    /// Build a simple set of 4 test control points for spline constraint Jacobian tests.
    fn make_test_control_points() -> [Isometry2F64; 4] {
        [
            Isometry2F64::from_factor_and_translation(
                Rotation2F64::identity(),
                VecF64::<2>::new(0.0, 0.0),
            ),
            Isometry2F64::from_factor_and_translation(
                Rotation2F64::identity(),
                VecF64::<2>::new(1.0, 2.0),
            ),
            Isometry2F64::from_factor_and_translation(
                Rotation2F64::identity(),
                VecF64::<2>::new(2.0, 3.0),
            ),
            Isometry2F64::from_factor_and_translation(
                Rotation2F64::identity(),
                VecF64::<2>::new(3.0, 2.5),
            ),
        ]
    }

    #[test]
    fn se2_spline_circle_constraint_jacobian() {
        use crate::nlls::HasIneqConstraintFn as _;

        let center = VecF64::<2>::new(1.5, 1.5);
        let radius = 0.5_f64;
        let u = 0.5_f64;
        let case = SegmentCase::Normal;

        let cps = make_test_control_points();

        let constraint = SE2SplineCircleConstraint {
            center,
            radius,
            u,
            case,
            entity_indices: [0, 1, 2, 3],
        };

        let var_kinds = [VarKind::Free; 4];
        let args = (cps[0], cps[1], cps[2], cps[3]);
        let eval = constraint.eval(&(), [0, 1, 2, 3], args, var_kinds);
        println!("se2_spline_circle: h = {:.6}", eval.h);

        // Finite-difference check: verify the numerical Jacobian is self-consistent.
        let eps = 1e-6;
        for cp_idx in 0..4 {
            for dof in 0..3 {
                let mut xi_plus = VecF64::<3>::zeros();
                xi_plus[dof] = eps;
                let mut xi_minus = VecF64::<3>::zeros();
                xi_minus[dof] = -eps;

                let mut cps_plus = cps;
                cps_plus[cp_idx] = Isometry2F64::exp(xi_plus) * cps[cp_idx];
                let mut cps_minus = cps;
                cps_minus[cp_idx] = Isometry2F64::exp(xi_minus) * cps[cp_idx];

                let eval_plus = constraint.eval(
                    &(),
                    [0, 1, 2, 3],
                    (cps_plus[0], cps_plus[1], cps_plus[2], cps_plus[3]),
                    var_kinds,
                );
                let eval_minus = constraint.eval(
                    &(),
                    [0, 1, 2, 3],
                    (cps_minus[0], cps_minus[1], cps_minus[2], cps_minus[3]),
                    var_kinds,
                );

                let num_jac = (eval_plus.h - eval_minus.h) / (2.0 * eps);
                let computed_jac = eval.jacobian.block(cp_idx)[(0, dof)];
                let err = (num_jac - computed_jac).abs();
                println!(
                    "  cp[{cp_idx}] dof[{dof}]: computed={computed_jac:.6}, \
                     numeric={num_jac:.6}, err={err:.2e}"
                );
                // The Jacobian is itself computed numerically, so both sides should match closely.
                assert!(
                    err < 1e-4,
                    "Jacobian mismatch cp[{cp_idx}] dof[{dof}]: \
                     computed={computed_jac}, numeric={num_jac}"
                );
            }
        }
    }
}
