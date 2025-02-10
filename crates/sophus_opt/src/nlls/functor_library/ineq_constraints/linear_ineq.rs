use sophus_autodiff::linalg::{
    MatF64,
    VecF64,
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
pub struct LinearIneqConstraint {
    /// linear inequality constraint matrix
    pub mat_a: MatF64<3, 2>,
    /// lower and upper bounds
    pub bounds:  NonEmptyBoxRegion<3>,
    /// entity index
    pub entity_indices: [usize; 1],
}

impl LinearIneqConstraint {
    /// Compute the residual
    pub fn constraint<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        mat_a: Scalar::Matrix<3, 2>,
        x: Scalar::Vector<2>,
    ) -> Scalar::Vector<3> {
        mat_a * x
    }
}

impl IsIneqConstraint<3, 2, 1, (), VecF64<2>> for LinearIneqConstraint {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn bounds(&self) -> NonEmptyBoxRegion<3> {
        self.bounds
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        args: VecF64<2>,
        derivatives: [VarKind; 1],
    ) -> EvaluatedIneqConstraint<3, 2, 1> {
        let constraint_value: VecF64<3> = Self::constraint::<f64, 0, 0>(self.mat_a, args);

        (|| self.mat_a,).make_ineq(idx, derivatives, constraint_value, self.bounds)
    }
}
