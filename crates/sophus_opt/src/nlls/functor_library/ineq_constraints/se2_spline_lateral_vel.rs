//! Lateral velocity bound for an SE(2) Lie group B-spline.
//!
//! Constrains `vy_max² − vy(u)² ≥ 0` where `vy` is the lateral (body-frame)
//! velocity component of the SE(2) spline at parameter `u`.
//! Jacobians are numerical via left-perturbation.

use sophus_autodiff::linalg::{
    MatF64,
    VecF64,
};
use sophus_lie::Isometry2F64;
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

/// Lateral velocity inequality constraint for an SE(2) Lie group B-spline.
///
/// `h = vy_max² − vy(u)² ≥ 0`
///
/// where `vy(u)` is the lateral component (index 2) of the body-frame velocity.
#[derive(Clone, Debug)]
pub struct SE2SplineLateralVelConstraint {
    /// Maximum lateral velocity.
    pub vy_max: f64,
    /// Local parameter within the segment.
    pub u: f64,
    /// Time interval between control points.
    pub delta_t: f64,
    /// Segment boundary case.
    pub case: SegmentCase,
    /// Entity indices for the 4 control points.
    pub entity_indices: [usize; 4],
}

impl SE2SplineLateralVelConstraint {
    fn eval_h(&self, control_points: [Isometry2F64; 4]) -> f64 {
        let seg = LieGroupBSplineSegment {
            case: self.case,
            control_points,
        };
        let vel = seg.velocity(self.u, self.delta_t);
        let vy = vel[2];
        self.vy_max * self.vy_max - vy * vy
    }
}

impl HasIneqConstraintFn<12, 4, (), (Isometry2F64, Isometry2F64, Isometry2F64, Isometry2F64)>
    for SE2SplineLateralVelConstraint
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
        let control_points = [args.0, args.1, args.2, args.3];
        let h = self.eval_h(control_points);

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
