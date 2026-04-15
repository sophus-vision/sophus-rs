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

/// Ellipse obstacle inequality constraint for a 2D point variable.
///
/// The constraint ensures that a 2D point stays outside an axis-aligned ellipse
/// that has been rotated by `angle`:
///
/// `h(p) = uᵀ A u − 1 >= 0`
///
/// where `u = p − center` and `A = Rᵀ diag(1/a², 1/b²) R` with R being
/// a 2D rotation by `angle`. Semi-axes `a` (along rotated x) and `b` (along
/// rotated y).
///
/// Positive means the point is outside the ellipse (feasible).
#[derive(Clone, Debug)]
pub struct ScalarEllipseConstraint {
    /// Center of the ellipse.
    pub center: VecF64<2>,
    /// Inverse shape matrix A = Rᵀ diag(1/a², 1/b²) R.
    /// Pre-computed for efficiency.
    pub inv_shape: MatF64<2, 2>,
    /// Entity index for the 2D point.
    pub entity_indices: [usize; 1],
}

impl ScalarEllipseConstraint {
    /// Create from center, semi-axes (a, b), and rotation angle in radians.
    pub fn new(center: VecF64<2>, a: f64, b: f64, angle: f64, entity_index: usize) -> Self {
        let c = angle.cos();
        let s = angle.sin();
        // R = [[c, -s], [s, c]]
        // A = Rᵀ diag(1/a², 1/b²) R
        let ia2 = 1.0 / (a * a);
        let ib2 = 1.0 / (b * b);
        let mut inv_shape = MatF64::<2, 2>::zeros();
        inv_shape[(0, 0)] = c * c * ia2 + s * s * ib2;
        inv_shape[(0, 1)] = c * s * (ia2 - ib2);
        inv_shape[(1, 0)] = c * s * (ia2 - ib2);
        inv_shape[(1, 1)] = s * s * ia2 + c * c * ib2;

        Self {
            center,
            inv_shape,
            entity_indices: [entity_index],
        }
    }
}

impl HasIneqConstraintFn<2, 1, (), VecF64<2>> for ScalarEllipseConstraint {
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
        let u = point - self.center;
        let au = self.inv_shape * u;
        let h = u.dot(&au) - 1.0;

        // dh/dp = 2 uᵀ A = 2 (Au)ᵀ  (since A is symmetric)
        let j: MatF64<1, 2> = 2.0 * au.transpose();

        (|| j,).make_ineq(idx, var_kinds, h)
    }
}
