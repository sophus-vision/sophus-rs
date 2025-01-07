use crate::prelude::*;
use crate::quadratic_cost::evaluated_term::EvaluatedCostTerm;
use crate::quadratic_cost::evaluated_term::MakeEvaluatedCostTerm;
use crate::quadratic_cost::residual_fn::IsResidualFn;
use crate::quadratic_cost::term::IsTerm;
use crate::robust_kernel;
use crate::variables::VarKind;
use sophus_autodiff::dual::DualScalar;
use sophus_autodiff::dual::DualVector;
use sophus_autodiff::linalg::VecF64;
use sophus_autodiff::maps::VectorValuedVectorMap;
use sophus_lie::Isometry2;
use sophus_lie::Isometry2F64;

/// Cost function for a prior on an 2d isometry
#[derive(Copy, Clone)]
pub struct Isometry2PriorCostFn {}

/// Isometry2 prior term
#[derive(Clone)]
pub struct Isometry2PriorTerm {
    /// prior mean
    pub isometry_prior_mean: Isometry2F64,
    /// entity index
    pub entity_indices: [usize; 1],
}

impl IsTerm<1> for Isometry2PriorTerm {
    type Constants = Isometry2F64;

    fn c_ref(&self) -> &Self::Constants {
        &self.isometry_prior_mean
    }

    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    const DOF_TUPLE: [i64; 1] = [3];
}

fn res_fn<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
    isometry: Isometry2<Scalar, 1, DM, DN>,
    isometry_prior_mean: Isometry2<Scalar, 1, DM, DN>,
) -> Scalar::Vector<3> {
    (isometry * isometry_prior_mean.inverse()).log()
}

impl IsResidualFn<3, 1, (), Isometry2F64, Isometry2F64> for Isometry2PriorCostFn {
    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        args: Isometry2F64,
        var_kinds: [VarKind; 1],
        robust_kernel: Option<robust_kernel::RobustKernel>,
        isometry_prior_mean: &Isometry2F64,
    ) -> EvaluatedCostTerm<3, 1> {
        let isometry: Isometry2F64 = args;

        let residual = res_fn(isometry, *isometry_prior_mean);
        let dx_res_fn = |x: DualVector<3, 3, 1>| -> DualVector<3, 3, 1> {
            let pp = Isometry2::<DualScalar<3, 1>, 1, 3, 1>::exp(x) * isometry.to_dual_c();
            res_fn(
                pp,
                Isometry2::from_params(DualVector::from_real_vector(*isometry_prior_mean.params())),
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
