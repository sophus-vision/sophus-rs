use sophus_lie::{
    Isometry3,
    Isometry3F64,
};

use crate::{
    nlls::cost::evaluated_term::EvaluatedCostTerm,
    prelude::*,
    robust_kernel,
    variables::VarKind,
};

/// SE(3) pose graph relative-pose residual cost term.
///
/// `g(ʷTₘ, ʷTₙ) = log[ (ʷTₘ)⁻¹ ∙ ʷTₙ ∙ (ᵐTₙ)⁻¹ ]`
///
/// where `ʷTₘ ∈ SE(3)` is a pose of `m` in the world frame, `ʷTₙ ∈ SE(3)` is a pose
/// of `n` in the world frame, and `ᵐTₙ ∈ SE(3)` is the measured relative pose.
#[derive(Debug, Clone)]
pub struct PoseGraph3CostTerm {
    /// Measured relative pose: `ᵐTₙ` of type [Isometry3F64].
    pub pose_m_from_pose_n: Isometry3F64,
    /// Entity indices:
    /// - 0: `m` pose in the world frame `ʷTₘ`
    /// - 1: `n` pose in the world frame `ʷTₙ`
    pub entity_indices: [usize; 2],
}

impl PoseGraph3CostTerm {
    /// Compute the residual of the SE(3) pose graph term.
    ///
    /// `g(ʷTₘ, ʷTₙ) = log[ (ʷTₘ)⁻¹ ∙ ʷTₙ ∙ (ᵐTₙ)⁻¹ ]`
    pub fn residual<S: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        world_from_pose_m: Isometry3<S, 1, DM, DN>,
        world_from_pose_n: Isometry3<S, 1, DM, DN>,
        pose_m_from_pose_n: Isometry3<S, 1, DM, DN>,
    ) -> S::Vector<6> {
        (world_from_pose_m.inverse() * world_from_pose_n * pose_m_from_pose_n.inverse()).log()
    }
}

impl HasResidualFn<12, 2, (), (Isometry3F64, Isometry3F64)> for PoseGraph3CostTerm {
    fn idx_ref(&self) -> &[usize; 2] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 2],
        world_from_pose_x: (Isometry3F64, Isometry3F64),
        var_kinds: [VarKind; 2],
        robust_kernel: Option<robust_kernel::RobustKernel>,
    ) -> EvaluatedCostTerm<12, 2> {
        let world_from_pose_m = world_from_pose_x.0;
        let world_from_pose_n = world_from_pose_x.1;

        let residual = Self::residual(
            world_from_pose_m,
            world_from_pose_n,
            self.pose_m_from_pose_n,
        );

        (
            || {
                -Isometry3::dx_log_a_exp_x_b_at_0(
                    world_from_pose_m.inverse(),
                    world_from_pose_n * self.pose_m_from_pose_n.inverse(),
                )
            },
            || {
                Isometry3::dx_log_a_exp_x_b_at_0(
                    world_from_pose_m.inverse(),
                    world_from_pose_n * self.pose_m_from_pose_n.inverse(),
                )
            },
        )
            .make(idx, var_kinds, residual, robust_kernel, None)
    }
}
