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

/// Linear 1d equality constraint.
///
/// `x + y = lhs`.
///
/// The corresponding constraint residual is:
///
/// `c(x,y) = x + y - lhs`.
#[derive(Clone, Debug)]
pub struct ExampleLinearEqConstraint {
    /// Left-hand side of the equality constraint.
    pub lhs: f64,
    /// Entity indices:
    /// - 0: index for `x`.
    /// - 1: index for `y`.
    pub entity_indices: [usize; 2],
}

impl ExampleLinearEqConstraint {
    /// Compute the residual
    ///
    /// `c(x,y) = x + y - lhs`.
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        x: Scalar::Vector<1>,
        y: Scalar::Vector<1>,
        lhs: Scalar,
    ) -> Scalar::Vector<1> {
        Scalar::Vector::<1>::from_array([(x.elem(0) + y.elem(0)) - lhs])
    }
}

impl HasEqConstraintResidualFn<1, 2, 2, (), (VecF64<1>, VecF64<1>)> for ExampleLinearEqConstraint {
    fn idx_ref(&self) -> &[usize; 2] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 2],
        args: (VecF64<1>, VecF64<1>),
        var_kinds: [crate::variables::VarKind; 2],
    ) -> crate::nlls::constraint::evaluated_eq_constraint::EvaluatedEqConstraint<1, 2, 2> {
        let (x, y) = args;
        let residual: VecF64<1> = Self::residual(x, y, self.lhs);
        let dx0_res_fn = |a: DualVector<1, 1, 1>| -> DualVector<1, 1, 1> {
            let lhs_dual = DualScalar::from_f64(self.lhs);
            Self::residual::<DualScalar<1, 1>, 1, 1>(a, DualVector::from_real_vector(y), lhs_dual)
        };
        let dx1_res_fn = |a: DualVector<1, 1, 1>| -> DualVector<1, 1, 1> {
            let lhs_dual = DualScalar::from_f64(self.lhs);
            Self::residual::<DualScalar<1, 1>, 1, 1>(DualVector::from_real_vector(x), a, lhs_dual)
        };

        (
            || dx0_res_fn(DualVector::var(x)).jacobian(),
            || dx1_res_fn(DualVector::var(y)).jacobian(),
        )
            .make_eq(idx, var_kinds, residual)
    }
}
