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
use sophus_core::linalg::VecF64;
use sophus_lie::Isometry3;
use sophus_sensor::PinholeCamera;

/// Camera re-projection cost function
#[derive(Copy, Clone)]
pub struct ReprojectionCostFn {}

fn res_fn<Scalar: IsSingleScalar + IsScalar<1>>(
    intrinscs: PinholeCamera<Scalar, 1>,
    world_from_camera: Isometry3<Scalar, 1>,
    point_in_world: Scalar::Vector<3>,
    uv_in_image: Scalar::Vector<2>,
) -> Scalar::Vector<2> {
    let point_in_cam = world_from_camera.inverse().transform(&point_in_world);
    uv_in_image - intrinscs.cam_proj(&point_in_cam)
}

/// Reprojection term signature
#[derive(Clone)]
pub struct ReprojTermSignature {
    /// Pixel measurement
    pub uv_in_image: VecF64<2>,
    /// camera/intrinsics index, pose index, point index
    pub entity_indices: [usize; 3],
}

impl IsTermSignature<3> for ReprojTermSignature {
    type Constants = VecF64<2>;

    fn c_ref(&self) -> &Self::Constants {
        &self.uv_in_image
    }

    fn idx_ref(&self) -> &[usize; 3] {
        &self.entity_indices
    }

    const DOF_TUPLE: [i64; 3] = [4, 6, 3];
}

impl IsResidualFn<13, 3, (PinholeCamera<f64, 1>, Isometry3<f64, 1>, VecF64<3>), VecF64<2>>
    for ReprojectionCostFn
{
    fn eval(
        &self,
        (intrinsics, world_from_camera_pose, point_in_world): (
            PinholeCamera<f64, 1>,
            Isometry3<f64, 1>,
            VecF64<3>,
        ),
        var_kinds: [VarKind; 3],
        robust_kernel: Option<robust_kernel::RobustKernel>,
        uv_in_image: &VecF64<2>,
    ) -> Term<13, 3> {
        // calculate residual
        let residual = res_fn(
            intrinsics,
            world_from_camera_pose,
            point_in_world,
            *uv_in_image,
        );

        // calculate jacobian wrt intrinsics
        let d0_res_fn = |x: DualVector<4>| -> DualVector<2> {
            res_fn(
                PinholeCamera::<DualScalar, 1>::from_params_and_size(&x, intrinsics.image_size()),
                world_from_camera_pose.to_dual_c(),
                DualVector::from_real_vector(point_in_world),
                DualVector::from_real_vector(*uv_in_image),
            )
        };
        // calculate jacobian wrt world_from_camera_pose
        let d1_res_fn = |x: DualVector<6>| -> DualVector<2> {
            res_fn(
                PinholeCamera::<DualScalar, 1>::from_params_and_size(
                    &DualVector::from_real_vector(*intrinsics.params()),
                    intrinsics.image_size(),
                ),
                Isometry3::<DualScalar, 1>::exp(&x).group_mul(&world_from_camera_pose.to_dual_c()),
                DualVector::from_real_vector(point_in_world),
                DualVector::from_real_vector(*uv_in_image),
            )
        };
        // calculate jacobian wrt point_in_world
        let d2_res_fn = |x: DualVector<3>| -> DualVector<2> {
            res_fn(
                PinholeCamera::<DualScalar, 1>::from_params_and_size(
                    &DualVector::from_real_vector(*intrinsics.params()),
                    intrinsics.image_size(),
                ),
                world_from_camera_pose.to_dual_c(),
                x,
                DualVector::from_real_vector(*uv_in_image),
            )
        };

        (
            || {
                VectorValuedMapFromVector::<DualScalar, 1>::static_fw_autodiff(
                    d0_res_fn,
                    *intrinsics.params(),
                )
            },
            || {
                VectorValuedMapFromVector::<DualScalar, 1>::static_fw_autodiff(
                    d1_res_fn,
                    VecF64::<6>::zeros(),
                )
            },
            || {
                VectorValuedMapFromVector::<DualScalar, 1>::static_fw_autodiff(
                    d2_res_fn,
                    point_in_world,
                )
            },
        )
            .make_term(var_kinds, residual, robust_kernel, None)
    }
}
