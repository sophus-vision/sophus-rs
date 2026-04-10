use sophus_autodiff::linalg::{
    MatF64,
    VecF64,
};
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

/// Half-plane inequality constraint for an SE(3) pose.
///
/// The constraint ensures that the translation component of an SE(3) pose lies
/// on the positive side of a half-plane:
///
/// `h(T) = nᵀ · t(T) + d >= 0`
///
/// where `n` is the inward-pointing normal of the half-plane and `d` is the offset.
///
/// For example, `n = (0, 0, 1), d = 1` gives `h = z + 1 >= 0`, i.e. `z >= -1` (floor).
///
/// Jacobians are computed analytically using SE(3) left-perturbation:
/// `∂t(exp(ξ)·T)/∂ξ = [−hat(t), I]` (3×6).
#[derive(Clone, Debug)]
pub struct HalfPlaneConstraint {
    /// Inward-pointing normal of the half-plane.
    pub normal: VecF64<3>,
    /// Offset: the constraint is `n^T * t + d >= 0`.
    pub offset: f64,
    /// Entity index for the SE(3) pose `T`.
    pub entity_indices: [usize; 1],
}

use super::translation_jac;

impl HasIneqConstraintFn<6, 1, (), Isometry3F64> for HalfPlaneConstraint {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        pose: Isometry3F64,
        var_kinds: [VarKind; 1],
    ) -> EvaluatedIneqConstraint<6, 1> {
        let t = pose.translation();
        let h = self.normal.dot(&t) + self.offset;

        // dh/dξ = nᵀ · [−hat(t), I]  (1×6)
        let n_t: MatF64<1, 3> = self.normal.transpose();
        let j = n_t * translation_jac(t);

        (|| j,).make_ineq(idx, var_kinds, h)
    }
}
