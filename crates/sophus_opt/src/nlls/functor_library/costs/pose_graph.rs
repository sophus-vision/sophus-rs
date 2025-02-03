use sophus_lie::{
    Isometry2,
    Isometry2F64,
};

use crate::{
    nlls::quadratic_cost::evaluated_term::EvaluatedCostTerm,
    prelude::*,
    robust_kernel,
    variables::VarKind,
};

/// Pose graph term
#[derive(Debug, Clone)]
pub struct PoseGraphCostTerm {
    /// 2d relative pose constraint
    pub pose_a_from_pose_b: Isometry2F64,
    /// ids of the two poses
    pub entity_indices: [usize; 2],
}

impl PoseGraphCostTerm {
    /// Compute the residual of the pose graph term
    pub fn residual<S: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        world_from_pose_a: Isometry2<S, 1, DM, DN>,
        world_from_pose_b: Isometry2<S, 1, DM, DN>,
        pose_a_from_pose_b: Isometry2<S, 1, DM, DN>,
    ) -> S::Vector<3> {
        (world_from_pose_a.inverse() * world_from_pose_b * pose_a_from_pose_b.inverse()).log()
    }
}

impl IsCostTerm<12, 2, (), (Isometry2F64, Isometry2F64)> for PoseGraphCostTerm {
    fn idx_ref(&self) -> &[usize; 2] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 2],
        world_from_pose_x: (Isometry2F64, Isometry2F64),
        var_kinds: [VarKind; 2],
        robust_kernel: Option<robust_kernel::RobustKernel>,
    ) -> EvaluatedCostTerm<12, 2> {
        let world_from_pose_a = world_from_pose_x.0;
        let world_from_pose_b = world_from_pose_x.1;

        let residual = Self::residual(
            world_from_pose_a,
            world_from_pose_b,
            self.pose_a_from_pose_b,
        );

        (
            || {
                -Isometry2::dx_log_a_exp_x_b_at_0(
                    world_from_pose_a.inverse(),
                    world_from_pose_b * self.pose_a_from_pose_b.inverse(),
                )
            },
            || {
                Isometry2::dx_log_a_exp_x_b_at_0(
                    world_from_pose_a.inverse(),
                    world_from_pose_b * self.pose_a_from_pose_b.inverse(),
                )
            },
        )
            .make(idx, var_kinds, residual, robust_kernel, None)
    }
}
