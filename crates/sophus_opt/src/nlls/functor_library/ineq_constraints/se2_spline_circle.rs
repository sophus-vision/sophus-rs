//! Circle obstacle constraint evaluated at a Lie group SE(2) B-spline sample point.
//!
//! Takes 4 `Isometry2F64` control points, evaluates the SE(2) Lie group cubic B-spline
//! at parameter `u`, extracts the translation, and checks the circle obstacle constraint:
//!
//! `h = ‖translation(spline(u)) − center‖² − radius² ≥ 0`
//!
//! Jacobians are computed via **numerical central differences** using left-perturbation:
//!
//! For each control point `i` (0..4), for each DOF `d` (0..3):
//! - Perturb: `T_i_plus = exp(eps * e_d) * T_i`, rebuild segment, evaluate `h`
//! - `J[i][0, d] = (h_plus - h_minus) / (2 * eps)`

use sophus_autodiff::linalg::{
    MatF64,
    VecF64,
};
use sophus_lie::{
    Isometry2F64,
    prelude::*,
};
use sophus_spline::{
    lie_group_spline::LieGroupBSplineSegment,
    spline_segment::SegmentCase,
};

use crate::{
    nlls::constraint::ineq_constraint::{
        EvaluatedIneqConstraint,
        HasIneqConstraintFn,
        MakeEvaluatedIneqConstraint,
    },
    variables::VarKind,
};

/// Circle obstacle inequality constraint evaluated at an SE(2) Lie group B-spline sample point.
///
/// Given 4 SE(2) control points and a spline parameter `u ∈ [0, 1)`, evaluates
/// the Lie group cubic B-spline at `u`, extracts the translation, and ensures
/// that translation lies outside the circular obstacle:
///
/// `h = ‖translation(spline(u)) − center‖² − radius² ≥ 0`
///
/// Positive means outside (feasible). Jacobians are numerical (central differences,
/// left-perturbation on SE(2)).
#[derive(Clone, Debug)]
pub struct SE2SplineCircleConstraint {
    /// Center of the circular obstacle.
    pub center: VecF64<2>,
    /// Radius of the circular obstacle.
    pub radius: f64,
    /// Local parameter within the segment, `u ∈ [0, 1)`.
    pub u: f64,
    /// Segment boundary case.
    pub case: SegmentCase,
    /// Entity indices for the 4 control points `[prev, 0, 1, 2]`.
    pub entity_indices: [usize; 4],
}

impl SE2SplineCircleConstraint {
    /// Evaluate h = ‖translation(spline(u)) − center‖² − radius² for the given control points.
    fn eval_h(&self, control_points: [Isometry2F64; 4]) -> f64 {
        let seg = LieGroupBSplineSegment {
            case: self.case,
            control_points,
        };
        let pose = seg.interpolate(self.u);
        let t = pose.translation();
        let diff = t - self.center;
        diff.dot(&diff) - self.radius * self.radius
    }
}

impl HasIneqConstraintFn<12, 4, (), (Isometry2F64, Isometry2F64, Isometry2F64, Isometry2F64)>
    for SE2SplineCircleConstraint
{
    fn idx_ref(&self) -> &[usize; 4] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 4],
        args: (Isometry2F64, Isometry2F64, Isometry2F64, Isometry2F64),
        var_kinds: [VarKind; 4],
    ) -> EvaluatedIneqConstraint<12, 4> {
        let (t0, t1, t2, t3) = args;
        let control_points = [t0, t1, t2, t3];

        let h = self.eval_h(control_points);

        // Numerical Jacobians via central differences with left-perturbation.
        let eps = 1e-6;

        let make_jac_i = |cp_idx: usize| -> MatF64<1, 3> {
            let mut j = MatF64::<1, 3>::zeros();
            for d in 0..3 {
                let mut xi_plus = VecF64::<3>::zeros();
                xi_plus[d] = eps;
                let mut xi_minus = VecF64::<3>::zeros();
                xi_minus[d] = -eps;

                let mut pts_plus = control_points;
                pts_plus[cp_idx] = Isometry2F64::exp(xi_plus) * control_points[cp_idx];
                let mut pts_minus = control_points;
                pts_minus[cp_idx] = Isometry2F64::exp(xi_minus) * control_points[cp_idx];

                let h_plus = self.eval_h(pts_plus);
                let h_minus = self.eval_h(pts_minus);
                j[(0, d)] = (h_plus - h_minus) / (2.0 * eps);
            }
            j
        };

        (
            || make_jac_i(0),
            || make_jac_i(1),
            || make_jac_i(2),
            || make_jac_i(3),
        )
            .make_ineq(idx, var_kinds, h)
    }
}
