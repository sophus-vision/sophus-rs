use sophus_autodiff::{
    dual::DualVector,
    linalg::{
        MatF64,
        VecF64,
    },
};
use sophus_lie::{
    Isometry3,
    Isometry3F64,
};

use crate::{
    nlls::cost::evaluated_term::EvaluatedCostTerm,
    prelude::*,
    robust_kernel,
    variables::VarKind,
};

/// Isometry3 prior residual cost term.
///
///
/// `g(T) = log[T * E(T)⁻¹]`
///
/// where `T ∈ SE(3)` is the current isometry and `E(T) ∈ SE(3)` is the
/// prior mean isometry. The quadratic cost function is defined as:
///
///
/// `f(T) = 0.5 * (g(T)ᵀ * W * g(T)).`
///
/// where `W` is the prior `6x6` precision matrix, which is the inverse of the prior
/// covariance matrix.
#[derive(Clone, Debug)]
pub struct Isometry3PriorCostTerm {
    /// Prior mean, `E(T)` of type [Isometry3F64].
    pub isometry_prior_mean: Isometry3F64,
    /// Prior precision, `W`.
    pub isometry_prior_precision: MatF64<6, 6>,
    /// Entity index for `T`.
    pub entity_indices: [usize; 1],
}

impl Isometry3PriorCostTerm {
    /// Compute the residual.
    ///
    /// `g(T) = log[T * E(T)⁻¹]`
    ///
    /// Note that `T:= isometry` and `E(T):= isometry_prior_mean` are of type [Isometry3F64].
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        isometry: Isometry3<Scalar, 1, DM, DN>,
        isometry_prior_mean: Isometry3<Scalar, 1, DM, DN>,
    ) -> Scalar::Vector<6> {
        (isometry * isometry_prior_mean.inverse()).log()
    }
}

impl HasResidualFn<6, 1, (), Isometry3F64> for Isometry3PriorCostTerm {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        args: Isometry3F64,
        var_kinds: [VarKind; 1],
        robust_kernel: Option<robust_kernel::RobustKernel>,
    ) -> EvaluatedCostTerm<6, 1> {
        let isometry: Isometry3F64 = args;

        let residual = Self::residual(isometry, self.isometry_prior_mean);
        let dx_res_fn = |x: DualVector<6, 6, 1>| -> DualVector<6, 6, 1> {
            Self::residual(
                Isometry3::exp(x) * isometry.to_dual_c(),
                self.isometry_prior_mean.to_dual_c(),
            )
        };

        (|| dx_res_fn(DualVector::var(VecF64::<6>::zeros())).jacobian(),).make(
            idx,
            var_kinds,
            residual,
            robust_kernel,
            Some(self.isometry_prior_precision),
        )
    }
}
