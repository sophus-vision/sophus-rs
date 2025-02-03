use sophus_autodiff::linalg::VecF64;
use sophus_lie::{
    Isometry2,
    Isometry2F64,
};

use crate::{
    nlls::{
        functor_library::costs::pose_graph::PoseGraphCostTerm,
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

/// Pose graph example problem
#[derive(Debug, Clone)]
pub struct PoseCircleProblem {
    /// true poses
    pub true_world_from_robot: alloc::vec::Vec<Isometry2F64>,
    /// estimated poses
    pub est_world_from_robot: alloc::vec::Vec<Isometry2F64>,
    /// pose-pose constraints
    pub obs_pose_a_from_pose_b_poses:
        CostTerms<12, 2, (), (Isometry2F64, Isometry2F64), PoseGraphCostTerm>,
}

impl Default for PoseCircleProblem {
    fn default() -> Self {
        Self::new(25)
    }
}

impl PoseCircleProblem {
    /// Create a new pose graph problem
    pub fn new(len: usize) -> Self {
        let mut true_world_from_robot_poses = alloc::vec![];
        let mut est_world_from_robot_poses = alloc::vec![];
        let mut obs_pose_a_from_pose_b_poses = CostTerms::new(["poses", "poses"], alloc::vec![]);

        let radius = 10.0;

        for i in 0..len {
            let frac = i as f64 / len as f64;
            let angle = frac * core::f64::consts::TAU;
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            let p = VecF64::<3>::from_real_array([x, y, 0.1 * angle]);
            true_world_from_robot_poses.push(Isometry2::exp(p));
        }

        for i in 0..len {
            let a_idx = i;
            let b_idx = (i + 1) % len;
            let true_world_from_pose_a = true_world_from_robot_poses[a_idx];
            let true_world_from_pose_b = true_world_from_robot_poses[b_idx];

            let p = VecF64::<3>::from_real_array([0.001, 0.001, 0.0001]);
            let pose_a_from_pose_b =
                Isometry2::exp(p) * true_world_from_pose_a.inverse() * true_world_from_pose_b;

            obs_pose_a_from_pose_b_poses
                .collection
                .push(PoseGraphCostTerm {
                    pose_a_from_pose_b,
                    entity_indices: [a_idx, b_idx],
                });
        }

        est_world_from_robot_poses.push(true_world_from_robot_poses[0]);

        for i in 1..len {
            let a_idx = i - 1;
            let b_idx = i;
            let obs = obs_pose_a_from_pose_b_poses.collection[a_idx].clone();
            assert_eq!(obs.entity_indices[0], a_idx);
            assert_eq!(obs.entity_indices[1], b_idx);

            let world_from_pose_a = est_world_from_robot_poses[a_idx];
            let pose_a_from_pose_b = obs.pose_a_from_pose_b;
            let p = VecF64::<3>::from_real_array([0.1, 0.1, 0.1]);
            let world_from_pose_b = Isometry2::exp(p) * world_from_pose_a * pose_a_from_pose_b;

            est_world_from_robot_poses.push(world_from_pose_b);
        }

        assert_eq!(
            true_world_from_robot_poses.len(),
            est_world_from_robot_poses.len()
        );

        Self {
            true_world_from_robot: true_world_from_robot_poses,
            est_world_from_robot: est_world_from_robot_poses,
            obs_pose_a_from_pose_b_poses,
        }
    }

    /// Calculate the error of the current estimate
    pub fn calc_error(&self, est_world_from_robot: &[Isometry2F64]) -> f64 {
        let mut res_err = 0.0;
        for obs in self.obs_pose_a_from_pose_b_poses.collection.clone() {
            let residual = PoseGraphCostTerm::residual(
                est_world_from_robot[obs.entity_indices[0]],
                est_world_from_robot[obs.entity_indices[1]],
                obs.pose_a_from_pose_b,
            );
            res_err += residual.dot(&residual);
        }
        res_err /= self.obs_pose_a_from_pose_b_poses.collection.len() as f64;
        res_err
    }

    /// Optimize the problem
    pub fn optimize(&self, solver: LinearSolverType) -> Vec<Isometry2F64> {
        let mut constants = alloc::collections::BTreeMap::new();
        constants.insert(0, ());

        const POSES: &str = "poses";
        let variables = VarBuilder::new()
            .add_family(
                POSES,
                VarFamily::new_with_const_ids(
                    VarKind::Free,
                    self.est_world_from_robot.clone(),
                    constants,
                ),
            )
            .build();
        let solution = optimize(
            variables,
            alloc::vec![CostFn::new_box(
                (),
                self.obs_pose_a_from_pose_b_poses.clone(),
            )],
            OptParams {
                num_iterations: 16,
                initial_lm_damping: 1.0,
                parallelize: true,
                solver,
            },
        )
        .unwrap();
        solution.variables.get_members::<Isometry2F64>(POSES)
    }
}

#[test]
fn pose_circle_opt_tests() {
    for solver in LinearSolverType::sparse_solvers() {
        let pose_graph = PoseCircleProblem::new(2500);

        let res_err = pose_graph.calc_error(&pose_graph.est_world_from_robot);
        assert!(res_err > 1.0, "{} > thr?", res_err);

        let refined_world_from_robot = pose_graph.optimize(solver);

        let res_err = pose_graph.calc_error(&refined_world_from_robot);
        assert!(res_err < 0.05, "{} < thr?", res_err);
    }
}
