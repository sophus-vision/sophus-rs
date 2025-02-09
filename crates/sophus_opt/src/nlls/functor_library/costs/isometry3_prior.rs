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

/// Isometry3 prior cost term
#[derive(Clone, Debug)]
pub struct Isometry3PriorCostTerm {
    /// prior mean
    pub isometry_prior_mean: Isometry3F64,
    /// prior precision
    pub isometry_prior_precision: MatF64<6, 6>,
    /// entity index
    pub entity_indices: [usize; 1],
}

impl Isometry3PriorCostTerm {
    /// Compute the residual
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        isometry: Isometry3<Scalar, 1, DM, DN>,
        isometry_prior_mean: Isometry3<Scalar, 1, DM, DN>,
    ) -> Scalar::Vector<6> {
        (isometry * isometry_prior_mean.inverse()).log()
    }
}

impl IsCostTerm<6, 1, (), Isometry3F64> for Isometry3PriorCostTerm {
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

        let zeros: VecF64<6> = VecF64::<6>::zeros();

        (|| dx_res_fn(DualVector::var(zeros)).jacobian(),).make(
            idx,
            var_kinds,
            residual,
            robust_kernel,
            Some(self.isometry_prior_precision),
        )
    }
}
