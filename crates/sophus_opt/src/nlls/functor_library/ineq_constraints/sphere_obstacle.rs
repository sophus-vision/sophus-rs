use sophus_autodiff::{
    dual::{
        DualScalar,
        DualVector,
    },
    linalg::VecF64,
};
use sophus_lie::{
    Isometry3,
    Isometry3F64,
};

use crate::{
    nlls::constraint::ineq_constraint::HasIneqConstraintFn,
    prelude::*,
};

/// Sphere obstacle inequality constraint for an SE(3) pose.
///
/// The constraint ensures that the translation component of an SE(3) pose stays
/// outside a sphere defined by a center point and radius:
///
/// `h(T) = ‖t(T) − center‖² − radius² >= 0`
///
/// A positive value means the pose is outside the sphere (feasible region).
#[derive(Clone, Debug)]
pub struct SphereObstacleConstraint {
    /// The center of the sphere obstacle in world coordinates.
    pub center: VecF64<3>,
    /// The radius of the sphere obstacle.
    pub radius: f64,
    /// Entity index for the SE(3) pose `T`.
    pub entity_indices: [usize; 1],
}

impl SphereObstacleConstraint {
    /// Compute `h(T) = ‖t(T) − center‖² − radius²` as a 1-vector for autodiff.
    pub fn h_as_vec<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        pose: Isometry3<Scalar, 1, DM, DN>,
        center: Scalar::Vector<3>,
        radius_sq: Scalar,
    ) -> Scalar::Vector<1> {
        let t = pose.translation();
        let diff = t - center;
        let h = diff.squared_norm() - radius_sq;
        Scalar::Vector::<1>::from_array([h])
    }
}

impl HasIneqConstraintFn<6, 1, (), Isometry3F64> for SphereObstacleConstraint {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        pose: Isometry3F64,
        var_kinds: [crate::variables::VarKind; 1],
    ) -> crate::nlls::constraint::ineq_constraint::EvaluatedIneqConstraint<6, 1> {
        let radius_sq = self.radius * self.radius;

        // Compute the scalar h value at the current estimate.
        let h = {
            let t = pose.translation();
            let diff = t - self.center;
            diff.dot(&diff) - radius_sq
        };

        // Jacobian via left-perturbation: T(x) = exp(x) * T, differentiated at x = 0.
        let cx = self.center[0];
        let cy = self.center[1];
        let cz = self.center[2];
        let rsq = radius_sq;

        let dx_h_fn = |x: DualVector<f64, 6, 6, 1>| -> DualVector<f64, 1, 6, 1> {
            let center_f64 = VecF64::<3>::new(cx, cy, cz);
            let center_dual = DualVector::<f64, 3, 6, 1>::from_real_vector(center_f64);
            let radius_sq_dual = DualScalar::from_f64(rsq);
            Self::h_as_vec::<DualScalar<f64, 6, 1>, 6, 1>(
                Isometry3::exp(x) * pose.to_dual_c(),
                center_dual,
                radius_sq_dual,
            )
        };

        (|| dx_h_fn(DualVector::var(VecF64::<6>::zeros())).jacobian(),).make_ineq(idx, var_kinds, h)
    }
}
