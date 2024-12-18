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
use sophus_core::linalg::MatF64;
use sophus_core::linalg::VecF64;
use sophus_lie::Isometry3;
use sophus_lie::Isometry3F64;

/// Cost function for a prior on an 3d isometry
#[derive(Copy, Clone)]
pub struct Isometry3PriorCostFn {}

/// Isometry3 prior term signature
#[derive(Clone)]
pub struct Isometry3PriorTermSignature {
    /// prior mean
    pub isometry_prior: (Isometry3F64, MatF64<6, 6>),
    /// entity index
    pub entity_indices: [usize; 1],
}

impl IsTermSignature<1> for Isometry3PriorTermSignature {
    type Constants = (Isometry3F64, MatF64<6, 6>);

    fn c_ref(&self) -> &Self::Constants {
        &self.isometry_prior
    }

    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    const DOF_TUPLE: [i64; 1] = [3];
}

fn res_fn<Scalar: IsScalar<1> + IsSingleScalar>(
    isometry: Isometry3<Scalar, 1>,
    isometry_prior_mean: Isometry3<Scalar, 1>,
) -> Scalar::Vector<6> {
    (isometry * isometry_prior_mean.inverse()).log()
}

impl IsResidualFn<6, 1, (), Isometry3F64, (Isometry3F64, MatF64<6, 6>)> for Isometry3PriorCostFn {
    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        args: Isometry3F64,
        var_kinds: [VarKind; 1],
        robust_kernel: Option<robust_kernel::RobustKernel>,
        isometry_prior: &(Isometry3F64, MatF64<6, 6>),
    ) -> Term<6, 1> {
        let isometry: Isometry3F64 = args;

        let residual = res_fn(isometry, isometry_prior.0);
        let dx_res_fn = |x: DualVector<6>| -> DualVector<6> {
            let pp = Isometry3::<DualScalar, 1>::exp(x) * isometry.to_dual_c();
            res_fn(
                pp,
                Isometry3::from_params(DualVector::from_real_vector(isometry_prior.0.params())),
            )
        };

        let zeros: VecF64<6> = VecF64::<6>::zeros();

        (|| VectorValuedMapFromVector::<DualScalar, 1>::static_fw_autodiff(dx_res_fn, zeros),)
            .make_term(
                idx,
                var_kinds,
                residual,
                robust_kernel,
                Some(isometry_prior.1),
            )
    }
}
