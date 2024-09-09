use crate::cost_fn::IsResidualFn;
use crate::cost_fn::IsTermSignature;
use crate::prelude::*;
use crate::robust_kernel;
use crate::term::MakeTerm;
use crate::term::Term;
use crate::variables::VarKind;
use sophus_core::calculus::dual::DualScalar;
use sophus_core::calculus::dual::DualVector;
use sophus_core::calculus::maps::VectorValuedMapFromVector;
use sophus_core::linalg::VecF64;
use sophus_lie::Isometry2;
use sophus_lie::Isometry2F64;

/// Cost function for a prior on an 2d isometry
#[derive(Copy, Clone)]
pub struct Isometry2PriorCostFn {}

/// Isometry2 prior term signature
#[derive(Clone)]
pub struct Isometry2PriorTermSignature {
    /// prior mean
    pub isometry_prior_mean: Isometry2F64,
    /// entity index
    pub entity_indices: [usize; 1],
}

impl IsTermSignature<1> for Isometry2PriorTermSignature {
    type Constants = Isometry2F64;

    fn c_ref(&self) -> &Self::Constants {
        &self.isometry_prior_mean
    }

    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    const DOF_TUPLE: [i64; 1] = [3];
}

fn res_fn<Scalar: IsSingleScalar + IsScalar<1>>(
    isometry: Isometry2<Scalar, 1>,
    isometry_prior_mean: Isometry2<Scalar, 1>,
) -> Scalar::Vector<3> {
    Isometry2::<Scalar, 1>::group_mul(&isometry, &isometry_prior_mean.inverse()).log()
}

impl IsResidualFn<3, 1, Isometry2F64, Isometry2F64> for Isometry2PriorCostFn {
    fn eval(
        &self,
        args: Isometry2F64,
        var_kinds: [VarKind; 1],
        robust_kernel: Option<robust_kernel::RobustKernel>,
        isometry_prior_mean: &Isometry2F64,
    ) -> Term<3, 1> {
        let isometry: Isometry2F64 = args;

        let residual = res_fn(isometry, *isometry_prior_mean);
        let dx_res_fn = |x: DualVector<3>| -> DualVector<3> {
            let pp = Isometry2::<DualScalar, 1>::exp(&x).group_mul(&isometry.to_dual_c());
            res_fn(
                pp,
                Isometry2::from_params(&DualVector::from_real_vector(
                    *isometry_prior_mean.params(),
                )),
            )
        };

        let zeros: VecF64<3> = VecF64::<3>::zeros();

        (|| VectorValuedMapFromVector::<DualScalar, 1>::static_fw_autodiff(dx_res_fn, zeros),)
            .make_term(var_kinds, residual, robust_kernel, None)
    }
}
