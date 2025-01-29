use crate::nlls::quadratic_cost::evaluated_term::EvaluatedCostTerm;
use crate::prelude::*;
use crate::robust_kernel;
use crate::variables::VarKind;
use sophus_autodiff::dual::DualScalar;
use sophus_autodiff::dual::DualVector;
use sophus_autodiff::linalg::MatF64;
use sophus_autodiff::linalg::VecF64;
use sophus_autodiff::maps::VectorValuedVectorMap;
use sophus_lie::Isometry2;
use sophus_lie::Isometry2F64;

/// Isometry2 prior cost term
#[derive(Clone, Debug)]
pub struct Isometry2PriorCostTerm {
    /// prior mean
    pub isometry_prior_mean: (Isometry2F64, MatF64<3, 3>),

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

impl IsCostTerm<3, 1, (), Isometry2F64, (Isometry2F64, MatF64<3, 3>)> for Isometry2PriorCostTerm {
    type Constants = (Isometry2F64, MatF64<3, 3>);

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
        args: Isometry2F64,
        var_kinds: [VarKind; 1],
        robust_kernel: Option<robust_kernel::RobustKernel>,
        isometry_prior: &(Isometry2F64, MatF64<3, 3>),
    ) -> EvaluatedCostTerm<3, 1> {
        let isometry: Isometry2F64 = args;

        let residual = Self::residual(isometry, isometry_prior.0);
        let dx_res_fn = |x: DualVector<3, 3, 1>| -> DualVector<3, 3, 1> {
            let pp = Isometry2::<DualScalar<3, 1>, 1, 3, 1>::exp(x) * isometry.to_dual_c();
            Self::residual(
                pp,
                Isometry2::from_params(DualVector::from_real_vector(isometry_prior.0.params())),
            )
        };

        let zeros: VecF64<3> = VecF64::<3>::zeros();

        (|| {
            VectorValuedVectorMap::<DualScalar<3, 1>, 1, 3, 1>::fw_autodiff_jacobian(
                dx_res_fn, zeros,
            )
        },)
            .make(idx, var_kinds, residual, robust_kernel, None)
    }
}
