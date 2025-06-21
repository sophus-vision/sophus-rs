use std::sync::Arc;

use sophus_autodiff::{
    dual::DualScalar,
    linalg::VecF64,
    maps::VectorValuedVectorMap,
};
use sophus_geo::{
    LineF64,
    Ray,
    UnitVector,
};
use sophus_lie::prelude::IsSingleScalar;
use sophus_opt::{
    nlls::{
        EvaluatedCostTerm,
        HasResidualFn,
        MakeEvaluatedCostTerm,
    },
    robust_kernel::RobustKernel,
    variables::VarKind,
};

use crate::examples::optics_sim::convex_lens::BiConvexLens2F64;

/// Cost term for the chief ray.
#[derive(Debug, Clone)]
pub struct ChiefRayCost {
    /// The entity indices for the cost term.
    pub entity_indices: [usize; 1],
    /// The scene point where the ray originates.
    pub scene_point: VecF64<2>,
    /// The aperture stop of the lens.
    pub aperture: VecF64<2>,
}

impl ChiefRayCost {
    /// Residual function for the chief ray cost term.
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        angle_rad: VecF64<1>,
        scene_point: &VecF64<2>,
        lens: &Arc<BiConvexLens2F64>,
        aperture: &VecF64<2>,
    ) -> VecF64<1> {
        let ray = Ray {
            origin: *scene_point,
            dir: UnitVector::from_vector_and_normalize(&VecF64::<2>::new(
                angle_rad[0].cos(),
                angle_rad[0].sin(),
            )),
        };

        let out_ray = match lens.refract(ray.clone()) {
            Some(r) => r[1].clone(),
            None => ray,
        };

        let intersection_point =
            LineF64::from_point_pair(*aperture, VecF64::<2>::new(aperture[0], 10.0))
                .rays_intersect(&out_ray)
                .unwrap();
        VecF64::<1>::new(intersection_point[1] - aperture[1])
    }
}

impl HasResidualFn<1, 1, Arc<BiConvexLens2F64>, VecF64<1>> for ChiefRayCost {
    fn eval(
        &self,
        global_constants: &Arc<BiConvexLens2F64>,
        idx: [usize; 1],
        angle_rad: VecF64<1>,
        derivatives: [VarKind; 1],
        robust_kernel: Option<RobustKernel>,
    ) -> EvaluatedCostTerm<1, 1> {
        let residual = Self::residual::<f64, 0, 0>(
            angle_rad,
            &self.scene_point,
            global_constants,
            &self.aperture,
        );

        let dx_res_fn = |x: VecF64<1>| -> VecF64<1> {
            Self::residual::<DualScalar<1, 1>, 1, 1>(
                x,
                &self.scene_point,
                global_constants,
                &self.aperture,
            )
        };
        (|| {
            VectorValuedVectorMap::<f64, 1>::sym_diff_quotient_jacobian(dx_res_fn, angle_rad, 0.001)
        },)
            .make(idx, derivatives, residual, robust_kernel, None)
    }

    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }
}
