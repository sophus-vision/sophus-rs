use sophus_autodiff::linalg::MatF64;
use sophus_lie::{
    IsAffineGroup,
    Isometry3F64,
};

use crate::{
    nlls::constraint::ineq_constraint::{
        EvaluatedIneqConstraint,
        HasIneqConstraintFn,
        MakeEvaluatedIneqConstraint,
    },
    variables::VarKind,
};

/// Maximum velocity inequality constraint between two consecutive SE(3) poses.
///
/// The constraint ensures that the distance between translations of two poses
/// does not exceed `v_max`:
///
/// `h(T_a, T_b) = v_max² − ‖t(T_b) − t(T_a)‖² >= 0`
///
/// Jacobians are computed analytically using SE(3) left-perturbation:
/// `∂t(exp(ξ)·T)/∂ξ = [−hat(t), I]` (3×6).
#[derive(Clone, Debug)]
pub struct MaxVelocityConstraint {
    /// Maximum velocity squared (v_max²).
    pub v_max_sq: f64,
    /// Entity indices: [pose_a index, pose_b index].
    pub entity_indices: [usize; 2],
}

use super::translation_jac;

impl HasIneqConstraintFn<12, 2, (), (Isometry3F64, Isometry3F64)> for MaxVelocityConstraint {
    fn idx_ref(&self) -> &[usize; 2] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 2],
        args: (Isometry3F64, Isometry3F64),
        var_kinds: [VarKind; 2],
    ) -> EvaluatedIneqConstraint<12, 2> {
        let t_a = args.0.translation();
        let t_b = args.1.translation();
        let d = t_b - t_a;
        let h = self.v_max_sq - d.dot(&d);

        // dh/dd = -2 dᵀ  (1×3 row)
        // dd/dt_a = -I,  dd/dt_b = I
        // dh/dt_a =  2 dᵀ
        // dh/dt_b = -2 dᵀ
        let two_d_t: MatF64<1, 3> = 2.0 * d.transpose();
        let j_a = two_d_t * translation_jac(t_a);
        let j_b = (-two_d_t) * translation_jac(t_b);

        (|| j_a, || j_b).make_ineq(idx, var_kinds, h)
    }
}
