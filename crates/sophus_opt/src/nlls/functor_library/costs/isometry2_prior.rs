use sophus_autodiff::{
    dual::DualVector,
    linalg::{
        MatF64,
        VecF64,
    },
};
use sophus_lie::{
    Isometry2,
    Isometry2F64,
};

use crate::{
    nlls::cost::evaluated_term::EvaluatedCostTerm,
    prelude::*,
    robust_kernel,
    variables::VarKind,
};

/// Isometry2 prior residual cost term.
///
///
/// `g(T) = log[T * E(T)⁻¹]`
///
/// where `T ∈ SE(2)` is the current isometry and `E(T) ∈ SE(2)` is the
/// prior mean isometry. The quadratic cost function is defined as:
///
///
/// `f(T) = 0.5 * (g(T)ᵀ * W * g(T)).`
///
/// where `W` is the prior `3x3` precision matrix, which is the inverse of the prior
/// covariance matrix.
#[derive(Clone, Debug)]
pub struct Isometry2PriorCostTerm {
    /// Prior mean, `E(T)` of type [Isometry2F64].
    pub isometry_prior_mean: Isometry2F64,
    /// Prior precision, `W`.
    pub isometry_prior_precision: MatF64<3, 3>,
    /// Entity index for `T`.
    pub entity_indices: [usize; 1],
}

impl Isometry2PriorCostTerm {
    /// Compute the residual.
    ///
    /// `g(T) = log[T * E(T)⁻¹]`
    ///
    /// Note that `T:= isometry` and `E(T):= isometry_prior_mean` are of type [Isometry2F64].
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        isometry: Isometry2<Scalar, 1, DM, DN>,
        isometry_prior_mean: Isometry2<Scalar, 1, DM, DN>,
    ) -> Scalar::Vector<3> {
        (isometry * isometry_prior_mean.inverse()).log()
    }
}

impl HasResidualFn<3, 1, (), Isometry2F64> for Isometry2PriorCostTerm {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        args: Isometry2F64,
        var_kinds: [VarKind; 1],
        robust_kernel: Option<robust_kernel::RobustKernel>,
    ) -> EvaluatedCostTerm<3, 1> {
        let isometry: Isometry2F64 = args;

        let residual = Self::residual(isometry, self.isometry_prior_mean);
        let dx_res_fn = |x: DualVector<3, 3, 1>| -> DualVector<3, 3, 1> {
            Self::residual(
                Isometry2::exp(x) * isometry.to_dual_c(),
                self.isometry_prior_mean.to_dual_c(),
            )
        };

        (|| dx_res_fn(DualVector::var(VecF64::<3>::zeros())).jacobian(),).make(
            idx,
            var_kinds,
            residual,
            robust_kernel,
            Some(self.isometry_prior_precision),
        )
    }
}
