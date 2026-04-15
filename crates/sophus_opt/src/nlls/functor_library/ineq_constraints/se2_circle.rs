//! Circle obstacle constraint for an SE(2) pose.
//!
//! Checks that the **translation** component of an SE(2) pose lies outside a
//! circular obstacle:
//!
//! `h(T) = ‖t(T) − center‖² − radius² >= 0`
//!
//! The Jacobian is derived analytically using left-perturbation:
//!
//! For `T(ξ) = exp(ξ) · T`, differentiating translation w.r.t. `ξ = (θ, vx, vy)` at `ξ = 0`:
//!
//! `dt/dξ = [−t_y, 1, 0; t_x, 0, 1]`  (2×3)
//!
//! so `dh/dξ = 2·(t − center)ᵀ · dt/dξ`  (1×3).

use sophus_autodiff::linalg::{
    MatF64,
    VecF64,
};
use sophus_lie::{
    Isometry2F64,
    prelude::IsAffineGroup,
};

use crate::{
    nlls::constraint::ineq_constraint::{
        EvaluatedIneqConstraint,
        HasIneqConstraintFn,
        MakeEvaluatedIneqConstraint,
    },
    variables::VarKind,
};

/// Circle obstacle inequality constraint for an SE(2) pose.
///
/// The constraint ensures that the translation component of an SE(2) pose stays
/// outside a circle:
///
/// `h(T) = ‖t(T) − center‖² − radius² >= 0`
///
/// Positive means outside the circle (feasible).
#[derive(Clone, Debug)]
pub struct SE2CircleConstraint {
    /// Center of the circular obstacle.
    pub center: VecF64<2>,
    /// Radius of the circular obstacle.
    pub radius: f64,
    /// Entity index for the SE(2) pose.
    pub entity_indices: [usize; 1],
}

impl HasIneqConstraintFn<3, 1, (), Isometry2F64> for SE2CircleConstraint {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        pose: Isometry2F64,
        var_kinds: [VarKind; 1],
    ) -> EvaluatedIneqConstraint<3, 1> {
        let t = pose.translation();
        let diff = t - self.center;
        let h = diff.dot(&diff) - self.radius * self.radius;

        // Translation Jacobian for SE(2) left-perturbation.
        // For exp(ξ)·T, differentiating t(T(ξ)) at ξ = 0:
        //
        //   dt/dξ =  [ -t_y,  1,  0  ]   (2×3)
        //            [  t_x,  0,  1  ]
        //
        // where ξ = (θ, vx, vy).
        let tx = t[0];
        let ty = t[1];

        let mut dt_dxi = MatF64::<2, 3>::zeros();
        dt_dxi[(0, 0)] = -ty;
        dt_dxi[(0, 1)] = 1.0;
        dt_dxi[(1, 0)] = tx;
        dt_dxi[(1, 2)] = 1.0;

        // dh/dξ = 2·(t − center)ᵀ · dt/dξ  (1×3)
        let dh_dt: MatF64<1, 2> = 2.0 * diff.transpose();
        let j: MatF64<1, 3> = dh_dt * dt_dxi;

        (|| j,).make_ineq(idx, var_kinds, h)
    }
}
