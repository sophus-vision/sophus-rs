use sophus_autodiff::{
    dual::{
        DualScalar,
        DualVector,
    },
    linalg::VecF64,
};

use crate::{
    nlls::constraint::eq_constraint::HasEqConstraintResidualFn,
    prelude::*,
};

/// Circle equality constraint for a 2D point.
///
/// `‖p − center‖ = radius`.
///
/// The constraint residual is `c(p) = ‖p − center‖ − radius`.
#[derive(Clone, Debug)]
pub struct ScalarCircleEqConstraint {
    /// Circle center.
    pub center: VecF64<2>,
    /// Circle radius.
    pub radius: f64,
    /// Entity index for the point variable.
    pub entity_indices: [usize; 1],
}

impl ScalarCircleEqConstraint {
    fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        point: Scalar::Vector<2>,
        center: Scalar::Vector<2>,
        radius: Scalar,
    ) -> Scalar::Vector<1> {
        let diff = point - center;
        let norm = diff.norm();
        Scalar::Vector::<1>::from_array([norm - radius])
    }
}

impl HasEqConstraintResidualFn<1, 2, 1, (), VecF64<2>> for ScalarCircleEqConstraint {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        point: VecF64<2>,
        var_kinds: [crate::variables::VarKind; 1],
    ) -> crate::nlls::constraint::evaluated_eq_constraint::EvaluatedEqConstraint<1, 2, 1> {
        let residual = Self::residual(point, self.center, self.radius);
        let center = self.center;
        let radius = self.radius;
        let dx_res_fn = |x: DualVector<f64, 2, 2, 1>| -> DualVector<f64, 1, 2, 1> {
            let center_dual = DualVector::from_real_vector(center);
            let radius_dual = DualScalar::from_f64(radius);
            Self::residual::<DualScalar<f64, 2, 1>, 2, 1>(x, center_dual, radius_dual)
        };

        (|| dx_res_fn(DualVector::var(point)).jacobian(),).make_eq(idx, var_kinds, residual)
    }
}
