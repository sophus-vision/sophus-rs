use sophus_autodiff::{
    dual::{
        DualScalar,
        DualVector,
    },
    linalg::VecF64,
};

use crate::{
    nlls::quadratic_cost::evaluated_term::EvaluatedCostTerm,
    prelude::*,
    robust_kernel::RobustKernel,
    variables::VarKind,
};

/// Quadratic cost functor
#[derive(Copy, Clone)]
pub struct QuadraticCostFunctor {}

/// Term of the quadratic cost function
#[derive(Clone, Debug)]
pub struct QuadraticCostTerm {
    /// Measurement
    pub z: VecF64<1>,
    /// entity index
    pub entity_indices: [usize; 1],
}

impl QuadraticCostTerm {
    /// Compute the residual
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        x: Scalar::Vector<1>,
        z: Scalar::Vector<1>,
    ) -> Scalar::Vector<1> {
        x - z
    }
}

impl IsCostTerm<1, 1, (), VecF64<1>, VecF64<1>> for QuadraticCostTerm {
    type Constants = VecF64<1>;

    fn c_ref(&self) -> &Self::Constants {
        &self.z
    }

    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        args: VecF64<1>,
        var_kinds: [VarKind; 1],
        robust_kernel: Option<RobustKernel>,
        z: &VecF64<1>,
    ) -> EvaluatedCostTerm<1, 1> {
        let residual = Self::residual::<f64, 0, 0>(args, *z);
        let dx_res_fn = |x: DualVector<1, 1, 1>| -> DualVector<1, 1, 1> {
            Self::residual::<DualScalar<1, 1>, 1, 1>(x, DualVector::from_real_vector(z))
        };

        (|| dx_res_fn(DualVector::var(args)).jacobian(),).make(
            idx,
            var_kinds,
            residual,
            robust_kernel,
            None,
        )
    }
}
