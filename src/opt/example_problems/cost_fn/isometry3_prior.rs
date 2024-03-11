use crate::calculus::dual::dual_scalar::Dual;
use crate::calculus::dual::dual_vector::DualV;
use crate::calculus::maps::vector_valued_maps::VectorValuedMapFromVector;
use crate::calculus::types::params::HasParams;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::IsVector;
use crate::calculus::types::MatF64;
use crate::calculus::types::VecF64;
use crate::lie::rotation3::Isometry3;
use crate::opt::cost_fn::IsResidualFn;
use crate::opt::cost_fn::IsTermSignature;
use crate::opt::robust_kernel;
use crate::opt::term::MakeTerm;
use crate::opt::term::Term;
use crate::opt::variables::VarKind;

/// Cost function for a prior on an 3d isometry
#[derive(Copy, Clone)]
pub struct Isometry3PriorCostFn {}

/// Isometry3 prior term signature
#[derive(Clone)]
pub struct Isometry3PriorTermSignature {
    /// prior mean
    pub isometry_prior: (Isometry3<f64>, MatF64<6, 6>),
    /// entity index
    pub entity_indices: [usize; 1],
}

impl IsTermSignature<1> for Isometry3PriorTermSignature {
    type Constants = (Isometry3<f64>, MatF64<6, 6>);

    fn c_ref(&self) -> &Self::Constants {
        &self.isometry_prior
    }

    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    const DOF_TUPLE: [i64; 1] = [3];
}

fn res_fn<Scalar: IsScalar>(
    isometry: Isometry3<Scalar>,
    isometry_prior_mean: Isometry3<Scalar>,
) -> Scalar::Vector<6> {
    Isometry3::<Scalar>::group_mul(&isometry, &isometry_prior_mean.inverse()).log()
}

impl IsResidualFn<6, 1, Isometry3<f64>, (Isometry3<f64>, MatF64<6, 6>)> for Isometry3PriorCostFn {
    fn eval(
        &self,
        args: Isometry3<f64>,
        var_kinds: [VarKind; 1],
        robust_kernel: Option<robust_kernel::RobustKernel>,
        isometry_prior: &(Isometry3<f64>, MatF64<6, 6>),
    ) -> Term<6, 1> {
        let isometry: Isometry3<f64> = args;

        let residual = res_fn(isometry, isometry_prior.0);
        let dx_res_fn = |x: DualV<6>| -> DualV<6> {
            let pp = Isometry3::<Dual>::exp(&x).group_mul(&isometry.to_dual_c());
            res_fn(
                pp,
                Isometry3::from_params(&DualV::c(*isometry_prior.0.params())),
            )
        };

        let zeros: VecF64<6> = VecF64::<6>::zeros();

        (|| VectorValuedMapFromVector::static_fw_autodiff(dx_res_fn, zeros),).make_term(
            var_kinds,
            residual,
            robust_kernel,
            Some(isometry_prior.1),
        )
    }
}
