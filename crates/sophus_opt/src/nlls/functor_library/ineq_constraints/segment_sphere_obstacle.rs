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

/// Sphere obstacle constraint for the **line segment** between two consecutive SE(3) poses.
///
/// Ensures that the full segment from `t_a` to `t_b` stays outside the sphere:
///
/// `h(T_a, T_b) = min_{t ∈ [0,1]} ‖(1−t)·t_a + t·t_b − center‖² − radius²  ≥  0`
///
/// Three branches handle the three cases for the closest-point parameter `t*`:
/// - `t* ≤ 0`: closest point is `t_a` — gradient only from pose A.
/// - `t* ≥ 1`: closest point is `t_b` — gradient only from pose B.
/// - `0 < t* < 1`: interior case — both poses contribute via analytic Jacobians.
///
/// Jacobians use SE(3) left-perturbation: `∂t(exp(ξ)·T)/∂ξ = [−hat(t), I]` (3×6).
#[derive(Clone, Debug)]
pub struct SegmentSphereObstacleConstraint {
    /// Center of the sphere obstacle in world coordinates.
    pub center: VecF64<3>,
    /// Radius of the sphere obstacle.
    pub radius: f64,
    /// Entity indices: [pose_a index, pose_b index].
    pub entity_indices: [usize; 2],
}

/// Build the 3×6 translation Jacobian for SE(3) left-perturbation: `[−hat(t), I]`.
fn translation_jac(t: VecF64<3>) -> MatF64<3, 6> {
    let mut j = MatF64::<3, 6>::zeros();
    // -hat(t) block (columns 0-2)
    j[(0, 1)] = t[2];
    j[(0, 2)] = -t[1];
    j[(1, 0)] = -t[2];
    j[(1, 2)] = t[0];
    j[(2, 0)] = t[1];
    j[(2, 1)] = -t[0];
    // I block (columns 3-5)
    j[(0, 3)] = 1.0;
    j[(1, 4)] = 1.0;
    j[(2, 5)] = 1.0;
    j
}

impl HasIneqConstraintFn<12, 2, (), (Isometry3F64, Isometry3F64)>
    for SegmentSphereObstacleConstraint
{
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
        let c = self.center;
        let r_sq = self.radius * self.radius;

        let u = t_b - t_a; // B − A
        let v = c - t_a; // C − A
        let s = u.dot(&u); // ‖u‖²

        // Degenerate: poses coincide — treat as point constraint at A.
        if s < 1e-12 {
            let d = t_a - c;
            let h = d.dot(&d) - r_sq;
            let dh_dta: MatF64<1, 3> = 2.0 * d.transpose();
            let j_a = dh_dta * translation_jac(t_a);
            let j_b = MatF64::<1, 6>::zeros();
            return (|| j_a, || j_b).make_ineq(idx, var_kinds, h);
        }

        let t_raw = v.dot(&u) / s;
        let t_star = t_raw.clamp(0.0, 1.0);
        let closest = t_a + t_star * u;
        let d = closest - c; // closest point − center
        let h = d.dot(&d) - r_sq;

        let (j_a, j_b) = if t_raw <= 0.0 {
            // Clamped to A: only pose A contributes.
            let dh_dta: MatF64<1, 3> = 2.0 * d.transpose();
            (dh_dta * translation_jac(t_a), MatF64::<1, 6>::zeros())
        } else if t_raw >= 1.0 {
            // Clamped to B: only pose B contributes.
            let dh_dtb: MatF64<1, 3> = 2.0 * d.transpose();
            (MatF64::<1, 6>::zeros(), dh_dtb * translation_jac(t_b))
        } else {
            // Interior: both poses contribute.
            //
            // alpha = t* = (v·u) / s
            // P = A + alpha * u
            // d = P − C
            //
            // ∂alpha/∂A = [−(u+v) + 2·alpha·u] / s  (column 3-vector)
            // ∂alpha/∂B = [v − 2·alpha·u] / s
            //
            // ∂d/∂A = (1−alpha)·I + u·(∂alpha/∂A)ᵀ
            // ∂d/∂B =    alpha·I + u·(∂alpha/∂B)ᵀ
            //
            // ∂h/∂t_A = 2·dᵀ·∂d/∂A = 2·[(1−alpha)·d + (d·u)·(∂alpha/∂A)]ᵀ
            // ∂h/∂t_B = 2·dᵀ·∂d/∂B = 2·[alpha·d    + (d·u)·(∂alpha/∂B)]ᵀ
            let alpha = t_raw;
            let du = d.dot(&u);
            let q = (-(u + v) + 2.0 * alpha * u) / s; // ∂alpha/∂A
            let p = (v - 2.0 * alpha * u) / s; // ∂alpha/∂B

            let dh_dta: MatF64<1, 3> = 2.0 * ((1.0 - alpha) * d + du * q).transpose();
            let dh_dtb: MatF64<1, 3> = 2.0 * (alpha * d + du * p).transpose();

            (dh_dta * translation_jac(t_a), dh_dtb * translation_jac(t_b))
        };

        (|| j_a, || j_b).make_ineq(idx, var_kinds, h)
    }
}
