use sophus_lie::Isometry3F64;

use crate::{
    nlls::{
        cost::evaluated_term::EvaluatedCostTerm,
        functor_library::ineq_constraints::translation_jac,
    },
    prelude::*,
    robust_kernel,
    variables::VarKind,
};

/// Translation difference cost term between two SE(3) poses.
///
/// Residual: `r(Tₘ, Tₙ) = tₙ - tₘ`  (3D world-frame displacement).
///
/// Minimizing `½ Σ ‖tᵢ₊₁ - tᵢ‖²` over a chain of poses with fixed endpoints
/// yields the shortest path (straight line in the unconstrained case).
///
/// Jacobians use SE(3) left-perturbation: `∂t(exp(ξ)·T)/∂ξ = [−hat(t), I]` (3×6).
#[derive(Debug, Clone)]
pub struct TranslationDiffCostTerm {
    /// Entity indices: [pose_m, pose_n].
    pub entity_indices: [usize; 2],
}

impl HasResidualFn<12, 2, (), (Isometry3F64, Isometry3F64)> for TranslationDiffCostTerm {
    fn idx_ref(&self) -> &[usize; 2] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 2],
        args: (Isometry3F64, Isometry3F64),
        var_kinds: [VarKind; 2],
        robust_kernel: Option<robust_kernel::RobustKernel>,
    ) -> EvaluatedCostTerm<12, 2> {
        let t_m = args.0.translation();
        let t_n = args.1.translation();
        let residual = t_n - t_m;

        // dr/dtₘ = −I,  dr/dtₙ = I
        // dr/dξₘ = −I · translation_jac(tₘ) = −translation_jac(tₘ)
        // dr/dξₙ =  I · translation_jac(tₙ) =  translation_jac(tₙ)
        (|| -translation_jac(t_m), || translation_jac(t_n)).make(
            idx,
            var_kinds,
            residual,
            robust_kernel,
            None,
        )
    }
}
