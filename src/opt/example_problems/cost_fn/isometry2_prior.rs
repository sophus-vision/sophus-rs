use crate::calculus::dual::dual_scalar::Dual;
use crate::calculus::dual::dual_vector::DualV;
use crate::calculus::maps::vector_valued_maps::VectorValuedMapFromVector;
use crate::calculus::types::params::HasParams;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::IsVector;
use crate::calculus::types::VecF64;
use crate::lie::isometry2::Isometry2;
use crate::opt::cost_fn::IsResidualFn;
use crate::opt::cost_fn::IsTermSignature;
use crate::opt::robust_kernel;
use crate::opt::term::MakeTerm;
use crate::opt::term::Term;
use crate::opt::variables::VarKind;

/// Cost function for a prior on an 2d isometry
#[derive(Copy, Clone)]
pub struct Isometry2PriorCostFn {}

/// Isometry2 prior term signature
#[derive(Clone)]
pub struct Isometry2PriorTermSignature {
    /// prior mean
    pub isometry_prior_mean: Isometry2<f64>,
    /// entity index
    pub entity_indices: [usize; 1],
}

impl IsTermSignature<1> for Isometry2PriorTermSignature {
    type Constants = Isometry2<f64>;

    fn c_ref(&self) -> &Self::Constants {
        &self.isometry_prior_mean
    }

    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    const DOF_TUPLE: [i64; 1] = [3];
}

fn res_fn<Scalar: IsScalar<1>>(
    isometry: Isometry2<Scalar>,
    isometry_prior_mean: Isometry2<Scalar>,
) -> Scalar::Vector<3> {
    Isometry2::<Scalar>::group_mul(&isometry, &isometry_prior_mean.inverse()).log()
}

impl IsResidualFn<3, 1, Isometry2<f64>, Isometry2<f64>> for Isometry2PriorCostFn {
    fn eval(
        &self,
        args: Isometry2<f64>,
        var_kinds: [VarKind; 1],
        robust_kernel: Option<robust_kernel::RobustKernel>,
        isometry_prior_mean: &Isometry2<f64>,
    ) -> Term<3, 1> {
        let isometry: Isometry2<f64> = args;

        let residual = res_fn(isometry, *isometry_prior_mean);
        let dx_res_fn = |x: DualV<3>| -> DualV<3> {
            let pp = Isometry2::<Dual>::exp(&x).group_mul(&isometry.to_dual_c());
            res_fn(
                pp,
                Isometry2::from_params(&DualV::c(*isometry_prior_mean.params())),
            )
        };

        let zeros: VecF64<3> = VecF64::<3>::zeros();

        (|| VectorValuedMapFromVector::static_fw_autodiff(dx_res_fn, zeros),).make_term(
            var_kinds,
            residual,
            robust_kernel,
            None,
        )
    }
}
