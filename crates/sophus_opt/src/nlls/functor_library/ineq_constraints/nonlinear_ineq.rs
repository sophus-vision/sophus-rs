use sophus_autodiff::{
    dual::{
        DualScalar,
        DualVector,
    },
    linalg::VecF64,
};
use sophus_geo::region::box_region::NonEmptyBoxRegion;

use crate::{
    nlls::constraint::{
        evaluated_ineq_constraint::{
            EvaluatedIneqConstraint,
            MakeEvaluatedIneqConstraint,
        },
        ineq_constraint::IsIneqConstraint,
    },
    prelude::*,
    variables::VarKind,
};

/// linear equality constraint
#[derive(Clone, Debug)]
pub struct NonLinearIneqConstraint {
     /// lower and upper bounds
     pub bounds:  NonEmptyBoxRegion<1>,
    /// entity index
    pub entity_indices: [usize; 1],
}

// 0 < x2 - ln(x1) < infinity
// 0 < x1 < infinity

impl NonLinearIneqConstraint {
    /// Compute the residual
    pub fn constraint<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        x: Scalar::Vector<2>,
    ) -> Scalar::Vector<1> {
        Scalar::Vector::<1>::from_array([x.elem(1) - x.elem(0).sin()])
    }
}

impl IsIneqConstraint<1, 2, 1, (), VecF64<2>> for NonLinearIneqConstraint {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn bounds(&self) -> NonEmptyBoxRegion<1> {
        self.bounds
    }
   

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        args: VecF64<2>,
        derivatives: [VarKind; 1],
    ) -> EvaluatedIneqConstraint<1, 2, 1> {
        let constraint_value: VecF64<1> = Self::constraint::<f64, 0, 0>(args);

        (|| Self::constraint::<DualScalar<2, 1>, 2, 1>(DualVector::var(args)).jacobian(),)
            .make_ineq(idx, derivatives, constraint_value, self.bounds)
    }
}
