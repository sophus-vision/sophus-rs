use sophus_autodiff::linalg::{
    MatF64,
    VecF64,
};
use sophus_lie::{
    Isometry2F64,
    Isometry3F64,
};

use crate::{
    nlls::{
        functor_library::costs::{
            isometry2_prior::Isometry2PriorCostTerm,
            isometry3_prior::Isometry3PriorCostTerm,
        },
        optimize,
        quadratic_cost::{
            cost_fn::CostFn,
            cost_term::CostTerms,
        },
        LinearSolverType,
        OptParams,
    },
    prelude::*,
    variables::{
        var_builder::VarBuilder,
        var_family::VarFamily,
        VarKind,
    },
};

extern crate alloc;

/// Simple 2D isometry prior problem
pub struct SimpleIso2PriorProblem {
    /// True world from robot isometry
    pub true_world_from_robot: Isometry2F64,
    /// Estimated world from robot isometry
    pub est_world_from_robot: Isometry2F64,
}

impl Default for SimpleIso2PriorProblem {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleIso2PriorProblem {
    fn new() -> Self {
        let true_world_from_robot =
            Isometry2F64::exp(VecF64::<3>::from_real_array([0.2, 1.0, 0.2]));
        Self {
            true_world_from_robot,
            est_world_from_robot: Isometry2F64::identity(),
        }
    }

    /// Test the simple 3D isometry prior problem
    pub fn test(&self, solver: LinearSolverType) {
        use sophus_autodiff::linalg::EPS_F64;

        const POSE: &str = "poses";
        let obs_pose_a_from_pose_b_poses = CostTerms::new(
            [POSE],
            alloc::vec![Isometry2PriorCostTerm {
                isometry_prior_mean: self.true_world_from_robot,
                isometry_prior_precision: MatF64::<3, 3>::identity(),
                entity_indices: [0],
            }],
        );
        let variables = VarBuilder::new()
            .add_family(
                POSE,
                VarFamily::new(VarKind::Free, alloc::vec![self.est_world_from_robot]),
            )
            .build();
        let solution = optimize(
            variables,
            alloc::vec![CostFn::new_box((), obs_pose_a_from_pose_b_poses.clone(),)],
            OptParams {
                num_iterations: 1,
                initial_lm_damping: EPS_F64, // if lm prior param is tiny
                parallelize: true,
                solver,
            },
        )
        .unwrap();

        let refined_world_from_robot = solution.variables.get_members::<Isometry2F64>(POSE)[0];

        approx::assert_abs_diff_eq!(
            self.true_world_from_robot.compact(),
            refined_world_from_robot.compact(),
            epsilon = EPS_F64
        );
    }
}

/// Simple 3D isometry prior problem
pub struct SimpleIso3PriorProblem {
    /// True world from robot isometry
    pub true_world_from_robot: Isometry3F64,
    /// Estimated world from robot isometry
    pub est_world_from_robot: Isometry3F64,
}

impl Default for SimpleIso3PriorProblem {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleIso3PriorProblem {
    fn new() -> Self {
        let true_world_from_robot =
            Isometry3F64::exp(VecF64::<6>::from_real_array([0.2, 0.0, 1.0, 0.2, 0.0, 1.0]));
        Self {
            true_world_from_robot,
            est_world_from_robot: Isometry3F64::identity(),
        }
    }

    /// Test the simple 3D isometry prior problem
    pub fn test(&self, solver: LinearSolverType) {
        use sophus_autodiff::linalg::EPS_F64;

        const POSE: &str = "poses";
        let obs_pose_a_from_pose_b_poses = CostTerms::new(
            [POSE],
            alloc::vec![Isometry3PriorCostTerm {
                isometry_prior_mean: self.true_world_from_robot,
                isometry_prior_precision: MatF64::<6, 6>::identity(),
                entity_indices: [0],
            }],
        );
        let variables = VarBuilder::new()
            .add_family(
                POSE,
                VarFamily::new(VarKind::Free, alloc::vec![self.est_world_from_robot]),
            )
            .build();
        let solution = optimize(
            variables,
            alloc::vec![CostFn::new_box((), obs_pose_a_from_pose_b_poses.clone(),)],
            OptParams {
                num_iterations: 1,
                initial_lm_damping: EPS_F64, // if lm prior param is tiny
                parallelize: true,
                solver,
            },
        )
        .unwrap();

        let refined_world_from_robot = solution.variables.get_members::<Isometry3F64>(POSE)[0];

        approx::assert_abs_diff_eq!(
            self.true_world_from_robot.compact(),
            refined_world_from_robot.compact(),
            epsilon = EPS_F64
        );
    }
}

#[test]
fn simple_prior_opt_tests() {
    for solver in LinearSolverType::all_solvers() {
        //SimpleIso2PriorProblem::new().test(solver);
        SimpleIso3PriorProblem::new().test(solver);
    }
}
