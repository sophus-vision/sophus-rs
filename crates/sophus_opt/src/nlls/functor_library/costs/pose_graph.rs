use sophus_lie::{
    Isometry2,
    Isometry2F64,
};

use crate::{
    nlls::cost::evaluated_term::EvaluatedCostTerm,
    prelude::*,
    robust_kernel,
    variables::VarKind,
};

/// 2d pose graph prior residual cost term.
///
/// `g(ʷTₘ, ʷTₙ) = log[ (ʷTₘ)⁻¹ ∙ ʷTₙ ∙ (ᵐTₙ)⁻¹ ]`
///
/// where `ʷTₘ ∈ SE(2)` is a pose of `m` in the world coordinate frame, `ʷTₙ ∈ SE(2)` is a pose
/// of `n` in the world coordinate frame, and `ᵐTₙ ∈ SE(2)` is the relative pose of `n` in the
/// `m` coordinate frame.
#[derive(Debug, Clone)]
pub struct PoseGraph2CostTerm {
    /// 2d relative pose constraint: `ᵐTₙ` of type [Isometry2F64].
    pub pose_m_from_pose_n: Isometry2F64,
    /// Entity indices:
    /// - 0: `m` pose in the world coordinate frame `ʷTₘ`
    /// - 1: `n` pose in the world coordinate frame `ʷTₙ`
    pub entity_indices: [usize; 2],
}

impl PoseGraph2CostTerm {
    /// Compute the residual of the pose graph term
    ///
    /// `g(ʷTₘ, ʷTₙ) = log[ (ʷTₘ)⁻¹ ∙ ʷTₙ ∙ (ᵐTₙ)⁻¹ ]`
    ///
    /// Note that `ʷTₘ:= world_from_pose_m`, `ʷTₙ:= world_from_pose_n` and
    /// `ᵐTₙ:= pose_m_from_pose_n` are of type `Isometry2F64`.
    pub fn residual<S: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        world_from_pose_m: Isometry2<S, 1, DM, DN>,
        world_from_pose_n: Isometry2<S, 1, DM, DN>,
        pose_m_from_pose_n: Isometry2<S, 1, DM, DN>,
    ) -> S::Vector<3> {
        (world_from_pose_m.inverse() * world_from_pose_n * pose_m_from_pose_n.inverse()).log()
    }
}

impl HasResidualFn<12, 2, (), (Isometry2F64, Isometry2F64)> for PoseGraph2CostTerm {
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
        let world_from_pose_m = world_from_pose_x.0;
        let world_from_pose_n = world_from_pose_x.1;

        let residual = Self::residual(
            world_from_pose_m,
            world_from_pose_n,
            self.pose_m_from_pose_n,
        );

        (
            || {
                -Isometry2::dx_log_a_exp_x_b_at_0(
                    world_from_pose_m.inverse(),
                    world_from_pose_n * self.pose_m_from_pose_n.inverse(),
                )
            },
            || {
                Isometry2::dx_log_a_exp_x_b_at_0(
                    world_from_pose_m.inverse(),
                    world_from_pose_n * self.pose_m_from_pose_n.inverse(),
                )
            },
        )
            .make(idx, var_kinds, residual, robust_kernel, None)
    }
}
