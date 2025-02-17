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

/// Linear 2d quadratic residual cost term.
///
/// `g(x) = x - z`.
///
/// The quadratic cost function is defined as:
///
/// `f(x) = 0.5 * (g(x)ᵀ * g(x)).`
#[derive(Clone, Debug)]
pub struct Quadratic2CostTerm {
    /// Measurement
    pub z: VecF64<2>,
    /// entity index
    pub entity_indices: [usize; 1],
}

impl Quadratic2CostTerm {
    /// Compute the residual
    ///
    /// `g(x) = x - z`.
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        x: Scalar::Vector<2>,
        z: Scalar::Vector<2>,
    ) -> Scalar::Vector<2> {
        x - z
    }
}

impl HasResidualFn<2, 1, (), VecF64<2>> for Quadratic2CostTerm {
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
        let dx_res_fn = |x: DualVector<2, 2, 1>| -> DualVector<2, 2, 1> {
            Self::residual::<DualScalar<2, 1>, 2, 1>(
                x,
                DualVector::<2, 2, 1>::from_real_vector(self.z),
            )
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
