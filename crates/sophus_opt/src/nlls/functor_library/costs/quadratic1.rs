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

/// Linear 1d quadratic residual cost term.
///
/// `g(x) = x - z`.
///
/// The quadratic cost function is defined as:
///
/// `f(x) = 0.5 * (g(x)ᵀ * g(x)).`
#[derive(Clone, Debug)]
pub struct Quadratic1CostTerm {
    /// Measurement
    pub z: VecF64<1>,
    /// entity index
    pub entity_indices: [usize; 1],
}

impl Quadratic1CostTerm {
    /// Compute the residual
    ///
    /// `g(x) = x - z`.
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        x: Scalar::Vector<1>,
        z: Scalar::Vector<1>,
    ) -> Scalar::Vector<1> {
        x - z
    }
}

impl HasResidualFn<1, 1, (), VecF64<1>> for Quadratic1CostTerm {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        args: VecF64<1>,
        var_kinds: [VarKind; 1],
        robust_kernel: Option<RobustKernel>,
    ) -> EvaluatedCostTerm<1, 1> {
        let residual = Self::residual::<f64, 0, 0>(args, self.z);
        let dx_res_fn = |x: DualVector<1, 1, 1>| -> DualVector<1, 1, 1> {
            Self::residual::<DualScalar<1, 1>, 1, 1>(x, DualVector::from_real_vector(self.z))
        };

        (|| dx_res_fn(DualVector::var(args)).jacobian(),).make(
            idx,
            var_kinds,
            residual,
            robust_kernel,
            None,
        )
    }
}
