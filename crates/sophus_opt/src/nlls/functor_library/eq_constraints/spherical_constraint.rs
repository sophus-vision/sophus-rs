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

/// Spherical equality constraint.
///
/// `|x| = r`.
///
/// where `x` is a 3D vector and `r` is the sphere radius.
///
/// The corresponding constraint residual is:
///
/// `c(x) = |x| - r`.
#[derive(Clone, Debug)]
pub struct SphericalConstraint {
    /// Sphere radius `r`.
    pub radius: f64,
    /// Entity indices for `x`.
    pub entity_indices: [usize; 1],
}

impl SphericalConstraint {
    /// Compute the constraint residual
    ///
    /// `c(x) = |x| - r`.
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        vec: Scalar::Vector<3>,
        radius: Scalar,
    ) -> Scalar::Vector<1> {
        let norm = vec.norm();
        Scalar::Vector::<1>::from_array([norm - radius])
    }
}

impl HasEqConstraintResidualFn<1, 3, 1, (), VecF64<3>> for SphericalConstraint {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        vec3: VecF64<3>,
        var_kinds: [crate::variables::VarKind; 1],
    ) -> crate::nlls::constraint::evaluated_eq_constraint::EvaluatedEqConstraint<1, 3, 1> {
        let residual = Self::residual(vec3, self.radius);
        let dx_res_fn = |x: DualVector<3, 3, 1>| -> DualVector<1, 3, 1> {
            let radius_dual = DualScalar::from_f64(self.radius);
            Self::residual::<DualScalar<3, 1>, 3, 1>(x, radius_dual)
        };

        (|| dx_res_fn(DualVector::var(vec3)).jacobian(),).make_eq(idx, var_kinds, residual)
    }
}
