use sophus_autodiff::linalg::{
    MatF64,
    VecF64,
};

use crate::{
    nlls::constraint::ineq_constraint::{
        EvaluatedIneqConstraint,
        HasIneqConstraintFn,
        MakeEvaluatedIneqConstraint,
    },
    variables::VarKind,
};

/// Half-plane inequality constraint for a 2D point variable.
///
/// The constraint ensures that a 2D point lies on the positive side of a
/// half-plane:
///
/// `h(p) = nᵀ · p + d >= 0`
///
/// where `n` is the inward-pointing normal and `d` is the offset.
///
/// For example, `n = (-1, 0), d = 2` gives `h = -x + 2 >= 0`, i.e. `x <= 2`.
#[derive(Clone, Debug)]
pub struct ScalarHalfPlaneConstraint {
    /// Inward-pointing normal of the half-plane.
    pub normal: VecF64<2>,
    /// Offset: the constraint is `n^T * p + d >= 0`.
    pub offset: f64,
    /// Entity index for the 2D point.
    pub entity_indices: [usize; 1],
}

impl HasIneqConstraintFn<2, 1, (), VecF64<2>> for ScalarHalfPlaneConstraint {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        point: VecF64<2>,
        var_kinds: [VarKind; 1],
    ) -> EvaluatedIneqConstraint<2, 1> {
        let h = self.normal.dot(&point) + self.offset;

        // dh/dp = nᵀ (1×2 row)
        let j: MatF64<1, 2> = self.normal.transpose();

        (|| j,).make_ineq(idx, var_kinds, h)
    }
}
