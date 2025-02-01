use sophus_autodiff::{
    dual::{
        DualScalar,
        DualVector,
    },
    linalg::{
        MatF64,
        VecF64,
    },
    maps::VectorValuedVectorMap,
};
use sophus_lie::{
    Isometry3,
    Isometry3F64,
};

use crate::{
    nlls::quadratic_cost::evaluated_term::EvaluatedCostTerm,
    prelude::*,
    robust_kernel,
    variables::VarKind,
};

/// Isometry3 prior cost term
#[derive(Clone, Debug)]
pub struct Isometry3PriorCostTerm {
    /// prior mean
    pub isometry_prior_mean: (Isometry3F64, MatF64<6, 6>),
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

impl IsCostTerm<6, 1, (), Isometry3F64, (Isometry3F64, MatF64<6, 6>)> for Isometry3PriorCostTerm {
    type Constants = (Isometry3F64, MatF64<6, 6>);

    fn c_ref(&self) -> &Self::Constants {
        &self.isometry_prior_mean
    }

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
        isometry_prior: &(Isometry3F64, MatF64<6, 6>),
    ) -> EvaluatedCostTerm<6, 1> {
        let isometry: Isometry3F64 = args;

        let residual = Self::residual(isometry, isometry_prior.0);
        let dx_res_fn = |x: DualVector<6, 6, 1>| -> DualVector<6, 6, 1> {
            let pp = Isometry3::<DualScalar<6, 1>, 1, 6, 1>::exp(x) * isometry.to_dual_c();
            Self::residual(
                pp,
                Isometry3::from_params(DualVector::from_real_vector(isometry_prior.0.params())),
            )
        };

        let zeros: VecF64<6> = VecF64::<6>::zeros();

        (|| {
            VectorValuedVectorMap::<DualScalar<6, 1>, 1, 6, 1>::fw_autodiff_jacobian(
                dx_res_fn, zeros,
            )
        },)
            .make(
                idx,
                var_kinds,
                residual,
                robust_kernel,
                Some(isometry_prior.1),
            )
    }
}
