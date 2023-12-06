use crate::calculus::dual::dual_scalar::Dual;
use crate::calculus::dual::dual_vector::DualV;
use crate::calculus::maps::vector_valued_maps::VectorValuedMapFromVector;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::IsVector;
use crate::calculus::types::V;
use crate::lie::rotation2::Isometry2;
use crate::opt::nlls::*;
use std::collections::HashMap;

fn res_fn<S: IsScalar>(
    world_from_pose_a: Isometry2<S>,
    world_from_pose_b: Isometry2<S>,
    pose_a_from_pose_b: Isometry2<S>,
) -> S::Vector<3> {
    (world_from_pose_a.inverse()
        .group_mul(&world_from_pose_b.group_mul(&pose_a_from_pose_b.inverse())))
        .log()
}

#[derive(Copy, Clone)]
struct PoseGraph {}

impl IsResidualFn<12, 2, (Isometry2<f64>, Isometry2<f64>), Isometry2<f64>> for PoseGraph {
    fn eval(
        &self,
        world_from_pose_x: (Isometry2<f64>, Isometry2<f64>),
        derivatives: [VarKind; 2],
        obs: &Isometry2<f64>,
    ) -> EvaluatedTerm<12, 2> {
        let world_from_pose_a = world_from_pose_x.0;
        let world_from_pose_b = world_from_pose_x.1;

        let residual = res_fn(world_from_pose_a, world_from_pose_b, *obs);

        let mut maybe_dx0 = None;
        let mut maybe_dx1 = None;

        let dx_res_fn_a = |x: DualV<3>| -> DualV<3> {
            let world_from_pose_a_bar: Isometry2<Dual> =
                Isometry2::<Dual>::exp(&x).group_mul(&world_from_pose_a.to_dual_c());
            res_fn(
                world_from_pose_a_bar,
                world_from_pose_b.to_dual_c(),
                obs.to_dual_c(),
            )
        };
        let dx_res_fn_b = |x: DualV<3>| -> DualV<3> {
            let world_from_pose_b_bar: Isometry2<Dual> =
                Isometry2::<Dual>::exp(&x).group_mul(&world_from_pose_b.to_dual_c());
            res_fn(
                world_from_pose_a.to_dual_c(),
                world_from_pose_b_bar,
                obs.to_dual_c(),
            )
        };
        let zeros: V<3> = V::<3>::zeros();

        if derivatives[0] != VarKind::Conditioned {
            let dx_res_a = VectorValuedMapFromVector::static_fw_autodiff(dx_res_fn_a, zeros);
            maybe_dx0 = Some(dx_res_a);
        }

        if derivatives[1] != VarKind::Conditioned {
            let dx_res_b = VectorValuedMapFromVector::static_fw_autodiff(dx_res_fn_b, zeros);
            maybe_dx1 = Some(dx_res_b);
        }
        EvaluatedTerm::new2(maybe_dx0, maybe_dx1, residual)
    }
}

#[derive(Clone)]
struct PoseGraphCostTermSignature {
    pose_a_from_pose_b: Isometry2<f64>,
    entity_indices: [usize; 2],
}

impl IsTermSignature<2> for PoseGraphCostTermSignature {
    type Constants = Isometry2<f64>;

    fn c_ref(&self) -> &Self::Constants {
        &self.pose_a_from_pose_b
    }

    fn idx_ref(&self) -> &[usize; 2] {
        &self.entity_indices
    }

    const DOF_TUPLE: [i64; 2] = [3, 3];
}

pub struct PoseCircleProblem {
    pub true_world_from_robot: Vec<Isometry2<f64>>,
    est_world_from_robot: Vec<Isometry2<f64>>,
    obs_pose_a_from_pose_b_poses: CostSignature<2, Isometry2<f64>, PoseGraphCostTermSignature>,
}

impl PoseCircleProblem {
    pub fn new(len: usize) -> Self {
        let mut true_world_from_robot_poses = vec![];
        let mut est_world_from_robot_poses = vec![];
        let mut obs_pose_a_from_pose_b_poses =
            CostSignature::<2, Isometry2<f64>, PoseGraphCostTermSignature> {
                family_names: ["poses".into(), "poses".into()],
                terms: vec![],
            };

        let radius = 10.0;

        for i in 0..len {
            let frac = i as f64 / len as f64;
            let angle = frac * std::f64::consts::TAU;
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            let p = V::<3>::from_c_array([x, y, 0.1 * angle]);
            true_world_from_robot_poses.push(Isometry2::exp(&p));
        }

        for i in 0..len {
            let a_idx = i;
            let b_idx = (i + 1) % len;
            let true_world_from_pose_a = true_world_from_robot_poses[a_idx];
            let true_world_from_pose_b = true_world_from_robot_poses[b_idx];

            let p = V::<3>::from_c_array([0.001, 0.001, 0.0001]);
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
            let p = V::<3>::from_c_array([0.1, 0.1, 0.1]);
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

    pub fn test(&self) {
        let mut res_err = 0.0;
        for obs in self.obs_pose_a_from_pose_b_poses.terms.clone() {
            let residual = res_fn(
                self.est_world_from_robot[obs.entity_indices[0]],
                self.est_world_from_robot[obs.entity_indices[1]],
                obs.pose_a_from_pose_b,
            );
            res_err += residual.dot(residual);
        }
        res_err /= self.obs_pose_a_from_pose_b_poses.terms.len() as f64;
        assert!(res_err > 1.0, "{} > thr?", res_err);

        let mut constants = HashMap::new();
        constants.insert(0, ());

        let family: VarFamily<Isometry2<f64>> =
            VarFamily::new(VarKind::Free, self.est_world_from_robot.clone(), constants);

        let var_pool = VarPoolBuilder::new().add_family("poses", family).build();

        let up_var_pool = optimize_one_cost(
            var_pool,
            Cost::new(self.obs_pose_a_from_pose_b_poses.clone(), PoseGraph {}),
            OptParams {
                num_iter: 5,
                initial_lm_nu: 1.0,
            },
        );

        let refined_world_from_robot = up_var_pool.get_members::<Isometry2<f64>>("poses".into());

        let mut res_err = 0.0;

        for obs in self.obs_pose_a_from_pose_b_poses.terms.clone() {
            let residual = res_fn(
                refined_world_from_robot[obs.entity_indices[0]],
                refined_world_from_robot[obs.entity_indices[1]],
                obs.pose_a_from_pose_b,
            );
            res_err += residual.dot(residual);
        }
        res_err /= self.obs_pose_a_from_pose_b_poses.terms.len() as f64;
        assert!(res_err < 0.05, "{} < thr?", res_err);
    }
}

mod tests {

    #[test]
    fn simple_prior_opt_tests() {
        use super::PoseCircleProblem;

        PoseCircleProblem::new(2500).test();
    }
}
