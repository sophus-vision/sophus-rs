//! Corridor navigation via constrained optimization.
//!
//! 30 SE(3) poses must navigate from start to goal. A corridor constraint forces
//! the middle section of the trajectory (poses 8..22) to stay below y = -2
//! (north wall). A prior pulls all poses toward y = 0, creating tension: the
//! optimizer must dip the trajectory into the corridor region to satisfy the
//! constraint while staying as close to y = 0 as possible.

use sophus_autodiff::linalg::VecF64;
use sophus_lie::{
    IsAffineGroup,
    Isometry3F64,
};

use crate::{
    nlls::{
        CostFn,
        CostTerms,
        IneqBarrierCostFn,
        IneqConstraints,
        OptParams,
        OptProblem,
        Optimizer,
        ineq_constraints::HalfPlaneConstraint,
    },
    variables::{
        VarBuilder,
        VarFamilies,
        VarFamily,
        VarKind,
    },
};

extern crate alloc;

/// Name of the pose variable family.
pub const POSES: &str = "poses";

/// A half-plane wall: nᵀ · t + d >= 0.
#[derive(Clone, Debug)]
pub struct Wall {
    /// Inward-pointing normal.
    pub normal: VecF64<3>,
    /// Offset.
    pub offset: f64,
    /// Human-readable label (e.g. "north wall").
    pub label: &'static str,
}

/// Corridor navigation problem using half-plane inequality constraints.
///
/// 30 SE(3) poses navigate from (0,0,0) to (29,0,0). A half-plane constraint
/// forces poses 8..22 to satisfy y <= -2 (the corridor). The cost is **path
/// length** (sum of squared translation differences between consecutive poses),
/// so the optimal trajectory is the shortest path that respects the corridor
/// constraint — it dips down to y = -2 (the wall), rides along it, and comes
/// back up.
pub struct CorridorNavigationProblem {
    /// Prior mean positions (straight line at y = 0, used only for visualization).
    pub prior_poses: alloc::vec::Vec<Isometry3F64>,
    /// Initial guess (dipping into the corridor).
    pub init_poses: alloc::vec::Vec<Isometry3F64>,
    /// Wall half-planes defining the corridor constraints.
    pub walls: alloc::vec::Vec<Wall>,
    /// Pose indices that are constrained by the corridor walls.
    pub constrained_pose_indices: alloc::vec::Vec<usize>,
}

impl CorridorNavigationProblem {
    /// Create a new corridor navigation problem.
    ///
    /// - 30 poses along x = 0..29.
    /// - Prior: straight line at y = 0.
    /// - Corridor constraint: poses 8..22 must have y <= -2 and y >= -6.
    /// - Endpoints are free (not pinned).
    /// - Initial guess: smooth dip from y = 0 down to y = -3 in the corridor section.
    pub fn new() -> Self {
        let num_poses = 30;
        let corridor_start = 8;
        let corridor_end = 22;

        // Prior: straight line at y = 0.
        let prior_poses: alloc::vec::Vec<Isometry3F64> = (0..num_poses)
            .map(|i| Isometry3F64::from_translation(VecF64::<3>::new(i as f64, 0.0, 0.0)))
            .collect();

        // Initial guess: just inside the wall (y=-2.5) for constrained poses.
        // The path-length gradient is zero in flat sections, so the init must
        // be near the optimal y=-2 for the optimizer to converge there.
        let corridor_center = -4.0; // center of corridor y ∈ [-6, -2]
        let init_poses: alloc::vec::Vec<Isometry3F64> = (0..num_poses)
            .map(|i| {
                let y = if i >= corridor_start && i <= corridor_end {
                    corridor_center
                } else if i >= corridor_start - 3 && i < corridor_start {
                    let t = (corridor_start - i) as f64 / 3.0;
                    corridor_center * (1.0 - t)
                } else if i > corridor_end && i <= corridor_end + 3 {
                    let t = (i - corridor_end) as f64 / 3.0;
                    corridor_center * (1.0 - t)
                } else {
                    0.0
                };
                Isometry3F64::from_translation(VecF64::<3>::new(i as f64, y, 0.0))
            })
            .collect();

        // Corridor walls only apply to constrained poses.
        let walls = alloc::vec![
            Wall {
                // North wall: y <= -2 => -y - 2 >= 0
                normal: VecF64::<3>::new(0.0, -1.0, 0.0),
                offset: -2.0,
                label: "north wall (y <= -2)",
            },
            Wall {
                // South wall: y >= -6 => y + 6 >= 0
                normal: VecF64::<3>::new(0.0, 1.0, 0.0),
                offset: 6.0,
                label: "south wall (y >= -6)",
            },
        ];

        let constrained_pose_indices: alloc::vec::Vec<usize> =
            (corridor_start..=corridor_end).collect();

        Self {
            prior_poses,
            init_poses,
            walls,
            constrained_pose_indices,
        }
    }

    /// Build variable families. Endpoints are pinned at (0,0,0) and (29,0,0).
    pub fn build_variables(&self) -> VarFamilies {
        let n = self.init_poses.len();
        let members = self.init_poses.clone();
        let mut constant_ids = alloc::collections::BTreeMap::new();
        constant_ids.insert(0, ());
        constant_ids.insert(n - 1, ());
        VarBuilder::new()
            .add_family(
                POSES,
                VarFamily::new_with_const_ids(VarKind::Free, members, constant_ids),
            )
            .build()
    }

    /// Build the cost functions.
    ///
    /// - Smoothness: `Σ ‖log(T_i⁻¹ T_{i+1})‖²` — penalizes rotation and translation differences
    ///   between consecutive poses.
    /// - Prior: pulls each pose toward `(x_i, 0, 0)` with translation precision 100 and rotation
    ///   precision 10.
    pub fn build_cost(&self) -> alloc::vec::Vec<alloc::boxed::Box<dyn crate::nlls::IsCostFn>> {
        use sophus_autodiff::linalg::MatF64;

        use crate::nlls::costs::PoseGraph3CostTerm;

        let n = self.init_poses.len();

        let smoothness_terms: alloc::vec::Vec<PoseGraph3CostTerm> = (0..n - 1)
            .map(|i| PoseGraph3CostTerm {
                pose_m_from_pose_n: Isometry3F64::identity(),
                entity_indices: [i, i + 1],
            })
            .collect();
        let smoothness_cost =
            CostFn::new_boxed((), CostTerms::new([POSES, POSES], smoothness_terms));

        let mut precision = MatF64::<6, 6>::zeros();
        for i in 0..3 {
            precision[(i, i)] = 100.0;
        }
        for i in 3..6 {
            precision[(i, i)] = 10.0;
        }

        let prior_terms: alloc::vec::Vec<crate::nlls::costs::Isometry3PriorCostTerm> =
            (0..self.prior_poses.len())
                .map(|i| crate::nlls::costs::Isometry3PriorCostTerm {
                    isometry_prior_mean: self.prior_poses[i],
                    isometry_prior_precision: precision,
                    entity_indices: [i],
                })
                .collect();
        let prior_cost = CostFn::new_boxed((), CostTerms::new([POSES], prior_terms));

        alloc::vec![smoothness_cost, prior_cost]
    }

    /// Build inequality barrier cost — only for constrained poses.
    pub fn build_barrier_cost(&self) -> alloc::boxed::Box<dyn crate::nlls::IsIneqBarrierFn> {
        let mut constraints: alloc::vec::Vec<HalfPlaneConstraint> = alloc::vec![];
        for &pose_i in &self.constrained_pose_indices {
            for wall in &self.walls {
                constraints.push(HalfPlaneConstraint {
                    normal: wall.normal,
                    offset: wall.offset,
                    entity_indices: [pose_i],
                });
            }
        }
        IneqBarrierCostFn::new_boxed((), IneqConstraints::new([POSES], constraints))
    }

    /// Build an optimizer for this problem.
    pub fn build_optimizer(&self, params: OptParams) -> Optimizer {
        let variables = self.build_variables();
        let costs = self.build_cost();
        let barrier = self.build_barrier_cost();
        Optimizer::new(
            variables,
            OptProblem {
                costs,
                eq_constraints: alloc::vec![],
                ineq_constraints: alloc::vec![barrier],
            },
            params,
        )
        .expect("Optimizer::new failed")
    }

    /// Run IPM optimization. Inner/outer loop parameters are in `params.ineq_method`.
    pub fn optimize_ipm(&self, params: OptParams) -> VarFamilies {
        let mut opt = self.build_optimizer(params);
        while let Ok(info) = opt.step() {
            if info.termination.is_some() {
                break;
            }
        }
        opt.into_variables()
    }

    /// Run SQP optimization. Inner/outer loop parameters are in `params.ineq_method`.
    pub fn optimize_sqp(&self, params: OptParams) -> VarFamilies {
        let mut opt = self.build_optimizer(params);
        while let Ok(info) = opt.step() {
            if info.termination.is_some() {
                break;
            }
        }
        opt.into_variables()
    }

    /// Check whether all corridor constraints are satisfied.
    pub fn constraint_satisfied(&self, vars: &VarFamilies) -> bool {
        let poses = vars.get_members::<Isometry3F64>(POSES);
        for &i in &self.constrained_pose_indices {
            let t = poses[i].translation();
            for wall in &self.walls {
                let h = wall.normal.dot(&t) + wall.offset;
                if h < -1e-4 {
                    return false;
                }
            }
        }
        true
    }
}

impl Default for CorridorNavigationProblem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use sophus_lie::IsAffineGroup;

    use super::*;
    use crate::nlls::IneqMethod;

    fn check_result(problem: &CorridorNavigationProblem, vars: &VarFamilies, method: &str) {
        assert!(
            problem.constraint_satisfied(vars),
            "[{method}] constraints violated"
        );

        let poses = vars.get_members::<Isometry3F64>(POSES);

        // Constrained poses should wall-hug at y ≈ -2 (the north wall).
        // The optimal trajectory rides the north wall (closest to the prior at y=0).
        for &i in &problem.constrained_pose_indices {
            let y = poses[i].translation()[1];
            assert!(
                (-2.10..=-1.99).contains(&y),
                "[{method}] pose {i}: y={y:.4} not wall-hugging (expected y ≈ -2.0)"
            );
        }
    }

    fn ipm_params() -> OptParams {
        OptParams {
            num_iterations: 50000,
            initial_lm_damping: 1.0,
            ineq_method: IneqMethod::Ipm {
                tau: 0.99,
                inner_iters: 500,
                lambda_decay: 0.5,
            },
            ..Default::default()
        }
    }

    fn sqp_params() -> OptParams {
        OptParams {
            num_iterations: 50000,
            initial_lm_damping: 1.0,
            ineq_method: IneqMethod::Sqp {
                tau: 0.99,
                inner_iters: 500,
                mu_decay: 0.5,
            },
            ..Default::default()
        }
    }

    #[test]
    fn corridor_ipm_hugs_wall() {
        let problem = CorridorNavigationProblem::new();
        let vars = problem.optimize_ipm(ipm_params());
        check_result(&problem, &vars, "IPM");
    }

    #[test]
    fn corridor_sqp_feasible() {
        let problem = CorridorNavigationProblem::new();
        let vars = problem.optimize_sqp(sqp_params());
        check_result(&problem, &vars, "SQP");
    }

    #[test]
    fn corridor_ipm_infeasible_start() {
        // All poses start at y=0 — violates north wall (y <= -2).
        // Elastic feasibility must find a feasible point, then IPM optimizes.
        let mut problem = CorridorNavigationProblem::new();
        let n = problem.init_poses.len();
        problem.init_poses = (0..n)
            .map(|i| Isometry3F64::from_translation(VecF64::<3>::new(i as f64, 0.0, 0.0)))
            .collect();
        let vars = problem.optimize_ipm(ipm_params());
        check_result(&problem, &vars, "IPM-infeasible");
    }

    #[test]
    fn corridor_sqp_infeasible_start() {
        let mut problem = CorridorNavigationProblem::new();
        let n = problem.init_poses.len();
        problem.init_poses = (0..n)
            .map(|i| Isometry3F64::from_translation(VecF64::<3>::new(i as f64, 0.0, 0.0)))
            .collect();
        let vars = problem.optimize_sqp(sqp_params());
        check_result(&problem, &vars, "SQP-infeasible");
    }
}
