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

/// Non-linear quadratic cost term
#[derive(Clone, Debug)]
pub struct ExampleNonLinearQuadraticCost {
    /// 2d measurement
    pub z: VecF64<2>,
    /// entity index
    pub entity_indices: [usize; 1],
}

impl ExampleNonLinearQuadraticCost {
    /// Compute the residual
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        x: Scalar::Vector<2>,
        z: Scalar::Vector<2>,
    ) -> Scalar::Vector<2> {
        Scalar::Vector::<2>::from_array([
            x.elem(0) * x.elem(0) + x.elem(1),
            x.elem(1) * x.elem(1) - x.elem(0),
        ]) - z
    }
}

impl IsCostTerm<2, 1, (), VecF64<2>> for ExampleNonLinearQuadraticCost {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        args: VecF64<2>,
        var_kinds: [VarKind; 1],
        robust_kernel: Option<RobustKernel>,
    ) -> EvaluatedCostTerm<2, 1> {
        let residual = Self::residual::<f64, 0, 0>(args, self.z);
        let dx_res_fn =
            |x| Self::residual::<DualScalar<2, 1>, 2, 1>(x, DualVector::from_real_vector(self.z));

        (|| dx_res_fn(DualVector::var(args)).jacobian(),).make(
            idx,
            var_kinds,
            residual,
            robust_kernel,
            None,
        )
    }
}
