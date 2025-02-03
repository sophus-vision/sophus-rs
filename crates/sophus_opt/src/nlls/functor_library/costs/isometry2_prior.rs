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
    nlls::quadratic_cost::evaluated_term::EvaluatedCostTerm,
    prelude::*,
    robust_kernel,
    variables::VarKind,
};

/// Isometry2 prior cost term
#[derive(Clone, Debug)]
pub struct Isometry2PriorCostTerm {
    /// prior mean
    pub isometry_prior_mean: Isometry2F64,
    /// prior precision
    pub isometry_prior_precision: MatF64<3, 3>,
    /// entity index
    pub entity_indices: [usize; 1],
}

impl Isometry2PriorCostTerm {
    /// Compute the residual
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        isometry: Isometry2<Scalar, 1, DM, DN>,
        isometry_prior_mean: Isometry2<Scalar, 1, DM, DN>,
    ) -> Scalar::Vector<3> {
        (isometry * isometry_prior_mean.inverse()).log()
    }
}

impl IsCostTerm<3, 1, (), Isometry2F64> for Isometry2PriorCostTerm {
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

        let zeros: VecF64<3> = VecF64::<3>::zeros();

        (|| dx_res_fn(DualVector::var(zeros)).jacobian(),).make(
            idx,
            var_kinds,
            residual,
            robust_kernel,
            Some(self.isometry_prior_precision),
        )
    }
}
