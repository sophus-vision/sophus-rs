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

/// Circle obstacle inequality constraint for a 2D point variable.
///
/// The constraint ensures that a 2D point stays outside a circle:
///
/// `h(p) = ‖p − center‖² − radius² >= 0`
///
/// Positive means the point is outside the circle (feasible).
#[derive(Clone, Debug)]
pub struct ScalarCircleConstraint {
    /// Center of the circle obstacle.
    pub center: VecF64<2>,
    /// Radius of the circle obstacle.
    pub radius: f64,
    /// Entity index for the 2D point.
    pub entity_indices: [usize; 1],
}

impl HasIneqConstraintFn<2, 1, (), VecF64<2>> for ScalarCircleConstraint {
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
        let diff = point - self.center;
        let h = diff.dot(&diff) - self.radius * self.radius;

        // dh/dp = 2 * (p - center)ᵀ  (1×2 row)
        let j: MatF64<1, 2> = 2.0 * diff.transpose();

        (|| j,).make_ineq(idx, var_kinds, h)
    }
}
