use crate::prelude::*;
use crate::quadratic_cost::evaluated_term::EvaluatedCostTerm;
use crate::quadratic_cost::evaluated_term::MakeEvaluatedCostTerm;
use crate::quadratic_cost::residual_fn::IsResidualFn;
use crate::quadratic_cost::term::IsTerm;
use crate::robust_kernel;
use crate::variables::VarKind;
use sophus_lie::Isometry2;
use sophus_lie::Isometry2F64;

/// residual function for a pose-pose constraint
pub fn res_fn<S: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
    world_from_pose_a: Isometry2<S, 1, DM, DN>,
    world_from_pose_b: Isometry2<S, 1, DM, DN>,
    pose_a_from_pose_b: Isometry2<S, 1, DM, DN>,
) -> S::Vector<3> {
    (world_from_pose_a.inverse() * world_from_pose_b * pose_a_from_pose_b.inverse()).log()
}

/// Cost function for 2d pose graph
#[derive(Copy, Clone, Debug)]
pub struct PoseGraphCostFn {}

impl IsResidualFn<12, 2, (), (Isometry2F64, Isometry2F64), Isometry2F64> for PoseGraphCostFn {
    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 2],
        world_from_pose_x: (Isometry2F64, Isometry2F64),
        var_kinds: [VarKind; 2],
        robust_kernel: Option<robust_kernel::RobustKernel>,
        obs: &Isometry2F64,
    ) -> EvaluatedCostTerm<12, 2> {
        let world_from_pose_a = world_from_pose_x.0;
        let world_from_pose_b = world_from_pose_x.1;

        let residual = res_fn(world_from_pose_a, world_from_pose_b, *obs);

        (
            || {
                -Isometry2::dx_log_a_exp_x_b_at_0(
                    world_from_pose_a.inverse(),
                    world_from_pose_b * obs.inverse(),
                )
            },
            || {
                Isometry2::dx_log_a_exp_x_b_at_0(
                    world_from_pose_a.inverse(),
                    world_from_pose_b * obs.inverse(),
                )
            },
        )
            .make(idx, var_kinds, residual, robust_kernel, None)
    }
}

/// Pose graph term
#[derive(Debug, Clone)]
pub struct PoseGraphCostTerm {
    /// 2d relative pose constraint
    pub pose_a_from_pose_b: Isometry2F64,
    /// ids of the two poses
    pub entity_indices: [usize; 2],
}

impl IsTerm<2> for PoseGraphCostTerm {
    type Constants = Isometry2F64;

    fn c_ref(&self) -> &Self::Constants {
        &self.pose_a_from_pose_b
    }

    fn idx_ref(&self) -> &[usize; 2] {
        &self.entity_indices
    }

    const DOF_TUPLE: [i64; 2] = [3, 3];
}
