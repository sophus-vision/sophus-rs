use super::cost_fn::pose_graph::PoseGraphCostFn;
use crate::cost_fn::CostFn;
use crate::cost_fn::CostSignature;
use crate::example_problems::cost_fn::pose_graph::PoseGraphCostTermSignature;
use crate::nlls::optimize;
use crate::nlls::OptParams;
use crate::prelude::*;
use crate::variables::VarFamily;
use crate::variables::VarKind;
use crate::variables::VarPool;
use crate::variables::VarPoolBuilder;
use sophus_core::linalg::VecF64;
use sophus_lie::Isometry2;
use std::collections::HashMap;

/// Pose graph example problem
#[derive(Debug, Clone)]
pub struct PoseCircleProblem {
    /// true poses
    pub true_world_from_robot: Vec<Isometry2<f64, 1>>,
    /// estimated poses
    pub est_world_from_robot: Vec<Isometry2<f64, 1>>,
    /// pose-pose constraints
    pub obs_pose_a_from_pose_b_poses:
        CostSignature<2, Isometry2<f64, 1>, PoseGraphCostTermSignature>,
}

impl Default for PoseCircleProblem {
    fn default() -> Self {
        Self::new(25)
    }
}

impl PoseCircleProblem {
    /// Create a new pose graph problem
    pub fn new(len: usize) -> Self {
        let mut true_world_from_robot_poses = vec![];
        let mut est_world_from_robot_poses = vec![];
        let mut obs_pose_a_from_pose_b_poses =
            CostSignature::<2, Isometry2<f64, 1>, PoseGraphCostTermSignature> {
                family_names: ["poses".into(), "poses".into()],
                terms: vec![],
            };

        let radius = 10.0;

        for i in 0..len {
            let frac = i as f64 / len as f64;
            let angle = frac * std::f64::consts::TAU;
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            let p = VecF64::<3>::from_real_array([x, y, 0.1 * angle]);
            true_world_from_robot_poses.push(Isometry2::exp(&p));
        }

        for i in 0..len {
            let a_idx = i;
            let b_idx = (i + 1) % len;
            let true_world_from_pose_a = true_world_from_robot_poses[a_idx];
            let true_world_from_pose_b = true_world_from_robot_poses[b_idx];

            let p = VecF64::<3>::from_real_array([0.001, 0.001, 0.0001]);
            let pose_a_from_pose_b = Isometry2::exp(&p).group_mul(
                &true_world_from_pose_a
                    .inverse()
                    .group_mul(&true_world_from_pose_b),
            );

            obs_pose_a_from_pose_b_poses
                .terms
                .push(PoseGraphCostTermSignature {
                    pose_a_from_pose_b,
                    entity_indices: [a_idx, b_idx],
                });
        }

        est_world_from_robot_poses.push(true_world_from_robot_poses[0]);

        for i in 1..len {
            let a_idx = i - 1;
            let b_idx = i;
            let obs = obs_pose_a_from_pose_b_poses.terms[a_idx].clone();
            assert_eq!(obs.entity_indices[0], a_idx);
            assert_eq!(obs.entity_indices[1], b_idx);

            let world_from_pose_a = est_world_from_robot_poses[a_idx];
            let pose_a_from_pose_b = obs.pose_a_from_pose_b;
            let p = VecF64::<3>::from_real_array([0.1, 0.1, 0.1]);
            let world_from_pose_b =
                Isometry2::exp(&p).group_mul(&world_from_pose_a.group_mul(&pose_a_from_pose_b));

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
    pub fn calc_error(&self, est_world_from_robot: &[Isometry2<f64, 1>]) -> f64 {
        let mut res_err = 0.0;
        for obs in self.obs_pose_a_from_pose_b_poses.terms.clone() {
            let residual = super::cost_fn::pose_graph::res_fn(
                est_world_from_robot[obs.entity_indices[0]],
                est_world_from_robot[obs.entity_indices[1]],
                obs.pose_a_from_pose_b,
            );
            res_err += residual.dot(residual);
        }
        res_err /= self.obs_pose_a_from_pose_b_poses.terms.len() as f64;
        res_err
    }

    /// Optimize the problem
    pub fn optimize(&self) -> VarPool {
        let mut constants = HashMap::new();
        constants.insert(0, ());

        let family: VarFamily<Isometry2<f64, 1>> = VarFamily::new_with_const_ids(
            VarKind::Free,
            self.est_world_from_robot.clone(),
            constants,
        );

        let var_pool = VarPoolBuilder::new().add_family("poses", family).build();

        optimize(
            var_pool,
            vec![CostFn::new_box(
                self.obs_pose_a_from_pose_b_poses.clone(),
                PoseGraphCostFn {},
            )],
            OptParams {
                num_iter: 5,
                initial_lm_nu: 1.0,
            },
        )
    }
}

#[test]
fn pose_circle_opt_tests() {
    let pose_graph = PoseCircleProblem::new(2500);

    let res_err = pose_graph.calc_error(&pose_graph.est_world_from_robot);
    assert!(res_err > 1.0, "{} > thr?", res_err);

    let up_var_pool = pose_graph.optimize();
    let refined_world_from_robot = up_var_pool.get_members::<Isometry2<f64, 1>>("poses".into());

    let res_err = pose_graph.calc_error(&refined_world_from_robot);
    assert!(res_err < 0.05, "{} < thr?", res_err);
}
