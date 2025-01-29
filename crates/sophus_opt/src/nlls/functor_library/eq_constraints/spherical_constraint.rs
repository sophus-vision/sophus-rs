use crate::nlls::constraint::eq_constraint::IsEqConstraint;
use crate::prelude::*;
use sophus_autodiff::dual::DualScalar;
use sophus_autodiff::dual::DualVector;
use sophus_autodiff::linalg::VecF64;
use sophus_autodiff::maps::VectorValuedVectorMap;

/// spherical equality constraint
#[derive(Clone, Debug)]
pub struct SphericalConstraint {
    /// sphere radius
    pub radius: f64,
    /// entity index
    pub entity_indices: [usize; 1],
}

impl SphericalConstraint {
    /// Compute the residual
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        vec: Scalar::Vector<3>,
        radius: Scalar,
    ) -> Scalar::Vector<1> {
        let norm = vec.norm();
        Scalar::Vector::<1>::from_array([norm - radius])
    }
}

impl IsEqConstraint<1, 3, 1, (), VecF64<3>, f64> for SphericalConstraint {
    type Constants = f64;

    fn c_ref(&self) -> &Self::Constants {
        &self.radius
    }

    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        vec3: VecF64<3>,
        var_kinds: [crate::variables::VarKind; 1],
        constants: &f64,
    ) -> crate::nlls::constraint::evaluated_constraint::EvaluatedConstraint<1, 3, 1> {
        let residual = Self::residual(vec3, *constants);
        let dx_res_fn = |x: DualVector<3, 3, 1>| -> DualVector<1, 3, 1> {
            let radius_dual = DualScalar::from_f64(*constants);
            Self::residual::<DualScalar<3, 1>, 3, 1>(x, radius_dual)
        };

        (|| {
            VectorValuedVectorMap::<DualScalar<3, 1>, 1, 3, 1>::fw_autodiff_jacobian(
                dx_res_fn, vec3,
            )
        },)
            .make_eq(idx, var_kinds, residual)
    }
}
