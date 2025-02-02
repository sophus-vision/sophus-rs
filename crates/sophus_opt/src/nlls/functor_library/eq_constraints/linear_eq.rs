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
pub struct LinearEqConstraint1 {
    /// sphere radius
    pub lhs: f64,
    /// entity index
    pub entity_indices: [usize; 2],
}

impl LinearEqConstraint1 {
    /// Compute the residual
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        x0: Scalar::Vector<1>,
        x1: Scalar::Vector<1>,
        lhs: Scalar,
    ) -> Scalar::Vector<1> {
        Scalar::Vector::<1>::from_array([(x0.get_elem(0) + x1.get_elem(0)) - lhs])
    }
}

impl IsEqConstraint<1, 2, 2, (), (VecF64<1>, VecF64<1>), f64> for LinearEqConstraint1 {
    type Constants = f64;

    fn c_ref(&self) -> &Self::Constants {
        &self.lhs
    }

    fn idx_ref(&self) -> &[usize; 2] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 2],
        args: (VecF64<1>, VecF64<1>),
        var_kinds: [crate::variables::VarKind; 2],
        constants: &f64,
    ) -> crate::nlls::constraint::evaluated_constraint::EvaluatedConstraint<1, 2, 2> {
        let (x0, x1) = args;
        let residual: VecF64<1> = Self::residual(x0, x1, *constants);
        let dx0_res_fn = |x: DualVector<1, 1, 1>| -> DualVector<1, 1, 1> {
            let radius_dual = DualScalar::from_f64(*constants);
            Self::residual::<DualScalar<1, 1>, 1, 1>(
                x,
                DualVector::from_real_vector(x1),
                radius_dual,
            )
        };
        let dx1_res_fn = |x: DualVector<1, 1, 1>| -> DualVector<1, 1, 1> {
            let radius_dual = DualScalar::from_f64(*constants);
            Self::residual::<DualScalar<1, 1>, 1, 1>(
                DualVector::from_real_vector(x0),
                x,
                radius_dual,
            )
        };

        (
            || dx0_res_fn(DualVector::var(x0)).jacobian(),
            || dx1_res_fn(DualVector::var(x1)).jacobian(),
        )
            .make_eq(idx, var_kinds, residual)
    }
}
