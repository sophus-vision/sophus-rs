use sophus_autodiff::{
    dual::{
        DualScalar,
        DualVector,
    },
    linalg::VecF64,
};

use crate::{
    nlls::cost::evaluated_term::EvaluatedCostTerm,
    prelude::*,
    robust_kernel::RobustKernel,
    variables::VarKind,
};

/// Example non-linear residual cost term.
///
/// ```ascii
///        [[x₀² + x₁]]
/// g(x) = [[        ]] - z
///        [[x₁² - x₀]]
/// ```
#[derive(Clone, Debug)]
pub struct ExampleNonLinearCostTerm {
    /// 2d measurement.
    pub z: VecF64<2>,
    /// Entity index for `x`.
    pub entity_indices: [usize; 1],
}

impl ExampleNonLinearCostTerm {
    /// Compute the residual
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        x: Scalar::Vector<2>,
        z: Scalar::Vector<2>,
    ) -> Scalar::Vector<2> {
        Scalar::Vector::<2>::from_array([
            x.elem(0) * x.elem(0) + x.elem(1),
            x.elem(1) * x.elem(1) - x.elem(0),
        ]) - z
    }
}

impl HasResidualFn<2, 1, (), VecF64<2>> for ExampleNonLinearCostTerm {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        args: VecF64<2>,
        var_kinds: [VarKind; 1],
        robust_kernel: Option<RobustKernel>,
    ) -> EvaluatedCostTerm<2, 1> {
        let residual = Self::residual::<f64, 0, 0>(args, self.z);
        let dx_res_fn =
            |x| Self::residual::<DualScalar<2, 1>, 2, 1>(x, DualVector::from_real_vector(self.z));

        (|| dx_res_fn(DualVector::var(args)).jacobian(),).make(
            idx,
            var_kinds,
            residual,
            robust_kernel,
            None,
        )
    }
}
