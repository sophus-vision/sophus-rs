use sophus_autodiff::{
    dual::DualVector,
    linalg::VecF64,
};
use sophus_lie::{
    Isometry3,
    Isometry3F64,
};
use sophus_sensor::{
    camera_enum::perspective_camera::PinholeCameraF64,
    PinholeCamera,
};

use crate::{
    nlls::cost::evaluated_term::EvaluatedCostTerm,
    prelude::*,
    robust_kernel,
    variables::VarKind,
};

/// Pinhole camera reprojection cost term
#[derive(Clone, Debug)]
pub struct PinholeCameraReprojectionCostTerm {
    /// Pixel measurement
    pub uv_in_image: VecF64<2>,
    /// camera/intrinsics index, pose index, point index
    pub entity_indices: [usize; 3],
}

impl PinholeCameraReprojectionCostTerm {
    /// Compute the residual
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        intrinscs: PinholeCamera<Scalar, 1, DM, DN>,
        world_from_camera: Isometry3<Scalar, 1, DM, DN>,
        point_in_world: Scalar::Vector<3>,
        uv_in_image: Scalar::Vector<2>,
    ) -> Scalar::Vector<2> {
        let point_in_cam = world_from_camera.inverse().transform(point_in_world);
        uv_in_image - intrinscs.cam_proj(point_in_cam)
    }
}

impl IsCostTerm<13, 3, (), (PinholeCameraF64, Isometry3F64, VecF64<3>)>
    for PinholeCameraReprojectionCostTerm
{
    fn idx_ref(&self) -> &[usize; 3] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 3],
        (intrinsics, world_from_camera_pose, point_in_world): (
            PinholeCameraF64,
            Isometry3F64,
            VecF64<3>,
        ),
        var_kinds: [VarKind; 3],
        robust_kernel: Option<robust_kernel::RobustKernel>,
    ) -> EvaluatedCostTerm<13, 3> {
        // calculate residual
        let residual = Self::residual(
            intrinsics,
            world_from_camera_pose,
            point_in_world,
            self.uv_in_image,
        );

        // calculate jacobian wrt intrinsics
        let d0_res_fn = |x: DualVector<4, 4, 1>| {
            Self::residual(
                PinholeCamera::from_params_and_size(x, intrinsics.image_size()),
                world_from_camera_pose.to_dual_c(),
                DualVector::from_real_vector(point_in_world),
                DualVector::from_real_vector(self.uv_in_image),
            )
        };
        // calculate jacobian wrt world_from_camera_pose
        let d1_res_fn = |x: DualVector<6, 6, 1>| -> DualVector<2, 6, 1> {
            Self::residual(
                PinholeCamera::from_params_and_size(
                    DualVector::from_real_vector(*intrinsics.params()),
                    intrinsics.image_size(),
                ),
                Isometry3::exp(x) * world_from_camera_pose.to_dual_c(),
                DualVector::from_real_vector(point_in_world),
                DualVector::from_real_vector(self.uv_in_image),
            )
        };
        // calculate jacobian wrt point_in_world
        let d2_res_fn = |x: DualVector<3, 3, 1>| -> DualVector<2, 3, 1> {
            Self::residual(
                PinholeCamera::from_params_and_size(
                    DualVector::from_real_vector(*intrinsics.params()),
                    intrinsics.image_size(),
                ),
                world_from_camera_pose.to_dual_c(),
                x,
                DualVector::from_real_vector(self.uv_in_image),
            )
        };

        (
            || d0_res_fn(DualVector::var(*intrinsics.params())).jacobian(),
            || d1_res_fn(DualVector::var(VecF64::<6>::zeros())).jacobian(),
            || d2_res_fn(DualVector::var(point_in_world)).jacobian(),
        )
            .make(idx, var_kinds, residual, robust_kernel, None)
    }
}
