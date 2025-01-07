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
use sophus_lie::Isometry3;
use sophus_lie::Isometry3F64;
use sophus_sensor::camera_enum::perspective_camera::PinholeCameraF64;
use sophus_sensor::PinholeCamera;

/// Camera re-projection cost function
#[derive(Copy, Clone)]
pub struct ReprojectionCostFn {}

fn res_fn<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
    intrinscs: PinholeCamera<Scalar, 1, DM, DN>,
    world_from_camera: Isometry3<Scalar, 1, DM, DN>,
    point_in_world: Scalar::Vector<3>,
    uv_in_image: Scalar::Vector<2>,
) -> Scalar::Vector<2> {
    let point_in_cam = world_from_camera.inverse().transform(point_in_world);
    uv_in_image - intrinscs.cam_proj(point_in_cam)
}

/// Reprojection term
#[derive(Clone)]
pub struct ReprojTerm {
    /// Pixel measurement
    pub uv_in_image: VecF64<2>,
    /// camera/intrinsics index, pose index, point index
    pub entity_indices: [usize; 3],
}

impl IsTerm<3> for ReprojTerm {
    type Constants = VecF64<2>;

    fn c_ref(&self) -> &Self::Constants {
        &self.uv_in_image
    }

    fn idx_ref(&self) -> &[usize; 3] {
        &self.entity_indices
    }

    const DOF_TUPLE: [i64; 3] = [4, 6, 3];
}

impl IsResidualFn<13, 3, (), (PinholeCameraF64, Isometry3F64, VecF64<3>), VecF64<2>>
    for ReprojectionCostFn
{
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
        uv_in_image: &VecF64<2>,
    ) -> EvaluatedCostTerm<13, 3> {
        // calculate residual
        let residual = res_fn(
            intrinsics,
            world_from_camera_pose,
            point_in_world,
            *uv_in_image,
        );

        // calculate jacobian wrt intrinsics
        let d0_res_fn = |x: DualVector<4, 4, 1>| -> DualVector<2, 4, 1> {
            res_fn(
                PinholeCamera::<DualScalar<4, 1>, 1, 4, 1>::from_params_and_size(
                    x,
                    intrinsics.image_size(),
                ),
                world_from_camera_pose.to_dual_c(),
                DualVector::from_real_vector(point_in_world),
                DualVector::from_real_vector(*uv_in_image),
            )
        };
        // calculate jacobian wrt world_from_camera_pose
        let d1_res_fn = |x: DualVector<6, 6, 1>| -> DualVector<2, 6, 1> {
            res_fn(
                PinholeCamera::<DualScalar<6, 1>, 1, 6, 1>::from_params_and_size(
                    DualVector::from_real_vector(*intrinsics.params()),
                    intrinsics.image_size(),
                ),
                Isometry3::<DualScalar<6, 1>, 1, 6, 1>::exp(x) * world_from_camera_pose.to_dual_c(),
                DualVector::from_real_vector(point_in_world),
                DualVector::from_real_vector(*uv_in_image),
            )
        };
        // calculate jacobian wrt point_in_world
        let d2_res_fn = |x: DualVector<3, 3, 1>| -> DualVector<2, 3, 1> {
            res_fn(
                PinholeCamera::<DualScalar<3, 1>, 1, 3, 1>::from_params_and_size(
                    DualVector::from_real_vector(*intrinsics.params()),
                    intrinsics.image_size(),
                ),
                world_from_camera_pose.to_dual_c(),
                x,
                DualVector::from_real_vector(*uv_in_image),
            )
        };

        (
            || {
                VectorValuedVectorMap::<DualScalar<4, 1>, 1, 4, 1>::fw_autodiff_jacobian(
                    d0_res_fn,
                    *intrinsics.params(),
                )
            },
            || {
                VectorValuedVectorMap::<DualScalar<6, 1>, 1, 6, 1>::fw_autodiff_jacobian(
                    d1_res_fn,
                    VecF64::<6>::zeros(),
                )
            },
            || {
                VectorValuedVectorMap::<DualScalar<3, 1>, 1, 3, 1>::fw_autodiff_jacobian(
                    d2_res_fn,
                    point_in_world,
                )
            },
        )
            .make(idx, var_kinds, residual, robust_kernel, None)
    }
}
