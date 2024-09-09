use crate::cost_fn::IsResidualFn;
use crate::cost_fn::IsTermSignature;
use crate::prelude::*;
use crate::robust_kernel;
use crate::term::MakeTerm;
use crate::term::Term;
use crate::variables::VarKind;
use sophus_lie::Isometry2;
use sophus_lie::Isometry2F64;

/// residual function for a pose-pose constraint
pub fn res_fn<S: IsSingleScalar>(
    world_from_pose_a: Isometry2<S, 1>,
    world_from_pose_b: Isometry2<S, 1>,
    pose_a_from_pose_b: Isometry2<S, 1>,
) -> S::Vector<3> {
    (world_from_pose_a
        .inverse()
        .group_mul(&world_from_pose_b.group_mul(&pose_a_from_pose_b.inverse())))
    .log()
}

/// Cost function for 2d pose graph
#[derive(Copy, Clone, Debug)]
pub struct PoseGraphCostFn {}

impl IsResidualFn<12, 2, (Isometry2F64, Isometry2F64), Isometry2F64> for PoseGraphCostFn {
    fn eval(
        &self,
        world_from_pose_x: (Isometry2F64, Isometry2F64),
        var_kinds: [VarKind; 2],
        robust_kernel: Option<robust_kernel::RobustKernel>,
        obs: &Isometry2F64,
    ) -> Term<12, 2> {
        let world_from_pose_a = world_from_pose_x.0;
        let world_from_pose_b = world_from_pose_x.1;

        let residual = res_fn(world_from_pose_a, world_from_pose_b, *obs);

        (
            || {
                -Isometry2::dx_log_a_exp_x_b_at_0(
                    &world_from_pose_a.inverse(),
                    &world_from_pose_b.group_mul(&obs.inverse()),
                )
            },
            || {
                Isometry2::dx_log_a_exp_x_b_at_0(
                    &world_from_pose_a.inverse(),
                    &world_from_pose_b.group_mul(&obs.inverse()),
                )
            },
        )
            .make_term(var_kinds, residual, robust_kernel, None)
    }
}

/// Pose graph term signature
#[derive(Debug, Clone)]
pub struct PoseGraphCostTermSignature {
    /// 2d relative pose constraint
    pub pose_a_from_pose_b: Isometry2F64,
    /// ids of the two poses
    pub entity_indices: [usize; 2],
}

impl IsTermSignature<2> for PoseGraphCostTermSignature {
    type Constants = Isometry2F64;

    fn c_ref(&self) -> &Self::Constants {
        &self.pose_a_from_pose_b
    }

    fn idx_ref(&self) -> &[usize; 2] {
        &self.entity_indices
    }

    const DOF_TUPLE: [i64; 2] = [3, 3];
}
