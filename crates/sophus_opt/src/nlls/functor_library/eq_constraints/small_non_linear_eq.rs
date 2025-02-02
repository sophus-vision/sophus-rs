use sophus_autodiff::{
    dual::{
        DualScalar,
        DualVector,
    },
    linalg::VecF64,
};

use crate::{
    nlls::constraint::eq_constraint::IsEqConstraint,
    prelude::*,
};

/// spherical equality constraint
#[derive(Clone, Debug)]
pub struct SmallNonLinearEqConstraint {
    /// sphere radius
    pub lhs: f64,
    /// entity index
    pub entity_indices: [usize; 1],
}

impl SmallNonLinearEqConstraint {
    /// Compute the residual
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        x: Scalar::Vector<2>,
        lhs: Scalar,
    ) -> Scalar::Vector<1> {
        let r = x.get_elem(0) * x.get_elem(0) + x.get_elem(1) * x.get_elem(1) - lhs;
        Scalar::Vector::<1>::from_array([r])
    }
}

impl IsEqConstraint<1, 2, 1, (), VecF64<2>, f64> for SmallNonLinearEqConstraint {
    type Constants = f64;

    fn c_ref(&self) -> &Self::Constants {
        &self.lhs
    }

    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        x: VecF64<2>,
        var_kinds: [crate::variables::VarKind; 1],
        constants: &f64,
    ) -> crate::nlls::constraint::evaluated_constraint::EvaluatedConstraint<1, 2, 1> {
        let residual = Self::residual(x, *constants);
        let dx0_res_fn = |x: DualVector<2, 2, 1>| -> DualVector<1, 2, 1> {
            let radius_dual = DualScalar::from_f64(*constants);
            Self::residual::<DualScalar<2, 1>, 2, 1>(x, radius_dual)
        };

        (|| dx0_res_fn(DualVector::var(x)).jacobian(),).make_eq(idx, var_kinds, residual)
    }
}
