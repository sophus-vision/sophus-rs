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
/// `x₀² + x₁² = lhs`.
///
/// The corresponding constraint residual is:
///
/// `c(x) = x₀² + x₁² - lhs`.
#[derive(Clone, Debug)]
pub struct ExampleNonLinearEqConstraint {
    /// Left-hand side of the equality constraint.
    pub lhs: f64,
    /// Entity index for `x`.
    pub entity_indices: [usize; 1],
}

impl ExampleNonLinearEqConstraint {
    /// Compute the constraint residual
    ///
    /// `c(x) = x₀² + x₁² - lhs`.
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        x: Scalar::Vector<2>,
        lhs: Scalar,
    ) -> Scalar::Vector<1> {
        let r = x.elem(0) * x.elem(0) + x.elem(1) * x.elem(1) - lhs;
        Scalar::Vector::<1>::from_array([r])
    }
}

impl HasEqConstraintResidualFn<1, 2, 1, (), VecF64<2>> for ExampleNonLinearEqConstraint {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        x: VecF64<2>,
        var_kinds: [crate::variables::VarKind; 1],
    ) -> crate::nlls::constraint::evaluated_eq_constraint::EvaluatedEqConstraint<1, 2, 1> {
        let residual = Self::residual(x, self.lhs);
        let dx0_res_fn = |x: DualVector<2, 2, 1>| -> DualVector<1, 2, 1> {
            let lhs_dual = DualScalar::from_f64(self.lhs);
            Self::residual::<DualScalar<2, 1>, 2, 1>(x, lhs_dual)
        };

        (|| dx0_res_fn(DualVector::var(x)).jacobian(),).make_eq(idx, var_kinds, residual)
    }
}
