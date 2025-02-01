use crate::nlls::quadratic_cost::evaluated_term::EvaluatedCostTerm;
use crate::prelude::*;
use crate::robust_kernel::RobustKernel;
use crate::variables::VarKind;
use sophus_autodiff::dual::DualScalar;
use sophus_autodiff::dual::DualVector;
use sophus_autodiff::linalg::VecF64;
use sophus_autodiff::maps::VectorValuedVectorMap;

/// Quadratic cost functor
#[derive(Copy, Clone)]
pub struct Quadratic2CostFunctor {}

/// Term of the quadratic cost function
#[derive(Clone, Debug)]
pub struct Quadratic2CostTerm {
    /// Measurement
    pub z: VecF64<2>,
    /// entity index
    pub entity_indices: [usize; 1],
}

impl Quadratic2CostTerm {
    /// Compute the residual
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        x: Scalar::Vector<2>,
        z: Scalar::Vector<2>,
    ) -> Scalar::Vector<2> {
        x - z
    }
}

impl IsCostTerm<2, 1, (), VecF64<2>, VecF64<2>> for Quadratic2CostTerm {
    type Constants = VecF64<2>;

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
        args: VecF64<2>,
        var_kinds: [VarKind; 1],
        robust_kernel: Option<RobustKernel>,
        z: &VecF64<2>,
    ) -> EvaluatedCostTerm<2, 1> {
        let residual = Self::residual::<f64, 0, 0>(args, *z);
        let dx_res_fn = |x: DualVector<2, 2, 1>| -> DualVector<2, 2, 1> {
            Self::residual::<DualScalar<2, 1>, 2, 1>(x, DualVector::<2, 2, 1>::from_real_vector(z))
        };

        (|| {
            VectorValuedVectorMap::<DualScalar<2, 1>, 1, 2, 1>::fw_autodiff_jacobian(
                dx_res_fn, args,
            )
        },)
            .make(idx, var_kinds, residual, robust_kernel, None)
    }
}
