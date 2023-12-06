use std::collections::HashMap;

use crate::calculus::dual::dual_scalar::Dual;
use crate::calculus::dual::dual_vector::DualV;
use crate::calculus::maps::vector_valued_maps::VectorValuedMapFromVector;
use crate::calculus::types::params::HasParams;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::IsVector;
use crate::calculus::types::V;
use crate::lie::rotation2::Isometry2;
use crate::opt::nlls::optimize_one_cost;
use crate::opt::nlls::Cost;
use crate::opt::nlls::CostSignature;
use crate::opt::nlls::EvaluatedTerm;
use crate::opt::nlls::IsResidualFn;
use crate::opt::nlls::IsTermSignature;
use crate::opt::nlls::OptParams;
use crate::opt::nlls::VarFamily;
use crate::opt::nlls::VarKind;
use crate::opt::nlls::VarPoolBuilder;

#[derive(Clone)]
struct SimplePriorCostTermSignature {
    c: Isometry2<f64>,
    entity_indices: [usize; 1],
}

impl IsTermSignature<1> for SimplePriorCostTermSignature {
    type Constants = Isometry2<f64>;

    fn c_ref(&self) -> &Self::Constants {
        &self.c
    }

    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    const DOF_TUPLE: [i64; 1] = [3];
}

fn res_fn<Scalar: IsScalar>(
    world_from_robot_pose: Isometry2<Scalar>,
    obs: Isometry2<Scalar>,
) -> Scalar::Vector<3> {
    Isometry2::<Scalar>::group_mul(&world_from_robot_pose, &obs.inverse()).log()
}

#[derive(Copy, Clone)]
struct SimplePrior {}

impl IsResidualFn<3, 1, Isometry2<f64>, Isometry2<f64>> for SimplePrior {
    fn eval(
        &self,
        args: Isometry2<f64>,
        derivatives: [VarKind; 1],
        obs: &Isometry2<f64>,
    ) -> EvaluatedTerm<3, 1> {
        let world_from_robot_pose: Isometry2<f64> = args;

        let residual = res_fn(world_from_robot_pose, *obs);
        let dx_res_fn = |x: DualV<3>| -> DualV<3> {
            let pp = Isometry2::<Dual>::exp(&x).group_mul(&world_from_robot_pose.to_dual_c());
            res_fn(pp, Isometry2::from_params(&DualV::c(*obs.params())))
        };

        let zeros: V<3> = V::<3>::zeros();
        let mut maybe_dx = None;
        if derivatives[0] != VarKind::Conditioned {
            let dx_res = VectorValuedMapFromVector::static_fw_autodiff(dx_res_fn, zeros);
            maybe_dx = Some(dx_res);
        }

        EvaluatedTerm::new1(maybe_dx, residual)
    }
}

pub struct SimplePriorProblem {
    true_world_from_robot: Isometry2<f64>,
    est_world_from_robot: Isometry2<f64>,
}

impl Default for SimplePriorProblem {
    fn default() -> Self {
        Self::new()
    }
}

impl SimplePriorProblem {
    pub fn new() -> Self {
        let p = V::<3>::from_c_array([0.2, 0.0, 1.0]);
        let true_world_from_robot = Isometry2::<f64>::exp(&p);
        Self {
            true_world_from_robot,
            est_world_from_robot: Isometry2::<f64>::identity(),
        }
    }

    pub fn test(&self) {
        let cost_signature = vec![SimplePriorCostTermSignature {
            c: self.true_world_from_robot,
            entity_indices: [0],
        }];

        let obs_pose_a_from_pose_b_poses =
            CostSignature::<1, Isometry2<f64>, SimplePriorCostTermSignature> {
                family_names: ["poses".into()],
                terms: cost_signature,
            };

        let family: VarFamily<Isometry2<f64>> = VarFamily::new(
            VarKind::Free,
            vec![self.est_world_from_robot],
            HashMap::new(),
        );

        let families = VarPoolBuilder::new().add_family("poses", family).build();

        approx::assert_abs_diff_ne!(
            self.true_world_from_robot.compact(),
            self.est_world_from_robot.compact(),
            epsilon = 1e-6
        );

        let up_families = optimize_one_cost(
            families,
            Cost::new(obs_pose_a_from_pose_b_poses.clone(), SimplePrior {}),
            OptParams {
                num_iter: 1,         // should converge in single iteration
                initial_lm_nu: 1e-6, // if lm prior param is tiny
            },
        );
        let refined_world_from_robot = up_families.get_members::<Isometry2<f64>>("poses".into());

        approx::assert_abs_diff_eq!(
            self.true_world_from_robot.compact(),
            refined_world_from_robot[0].compact(),
            epsilon = 1e-6
        );
    }
}

mod tests {

    #[test]
    fn simple_prior_opt_tests() {
        use super::SimplePriorProblem;

        SimplePriorProblem::new().test();
    }
}
