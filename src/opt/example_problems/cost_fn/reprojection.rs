use crate::calculus::dual::dual_scalar::Dual;
use crate::calculus::dual::dual_vector::DualV;
use crate::calculus::maps::vector_valued_maps::VectorValuedMapFromVector;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::IsVector;
use crate::calculus::types::VecF64;
use crate::lie::rotation3::Isometry3;
use crate::opt::cost_fn::IsResidualFn;
use crate::opt::cost_fn::IsTermSignature;
use crate::opt::robust_kernel;
use crate::opt::term::MakeTerm;
use crate::opt::term::Term;
use crate::opt::variables::IsVariable;
use crate::opt::variables::VarKind;
use crate::sensor::perspective_camera::PinholeCamera;

/// Camera re-projection cost function
#[derive(Copy, Clone)]
pub struct ReprojectionCostFn {}

impl IsVariable for PinholeCamera<f64> {
    const DOF: usize = 4;

    fn update(&mut self, delta: nalgebra::DVectorView<f64>) {
        let new_params = *self.params() + delta;
        self.set_params(&new_params);
    }
}

fn res_fn<Scalar: IsScalar>(
    intrinscs: PinholeCamera<Scalar>,
    world_from_camera: Isometry3<Scalar>,
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

impl IsResidualFn<13, 3, (PinholeCamera<f64>, Isometry3<f64>, VecF64<3>), VecF64<2>>
    for ReprojectionCostFn
{
    fn eval(
        &self,
        (intrinsics, world_from_camera_pose, point_in_world): (
            PinholeCamera<f64>,
            Isometry3<f64>,
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
        let d0_res_fn = |x: DualV<4>| -> DualV<2> {
            res_fn(
                PinholeCamera::<Dual>::from_params_and_size(&x, intrinsics.image_size()),
                world_from_camera_pose.to_dual_c(),
                DualV::c(point_in_world),
                DualV::c(*uv_in_image),
            )
        };
        // calculate jacobian wrt world_from_camera_pose
        let d1_res_fn = |x: DualV<6>| -> DualV<2> {
            res_fn(
                PinholeCamera::<Dual>::from_params_and_size(
                    &DualV::c(*intrinsics.params()),
                    intrinsics.image_size(),
                ),
                Isometry3::<Dual>::exp(&x).group_mul(&world_from_camera_pose.to_dual_c()),
                DualV::c(point_in_world),
                DualV::c(*uv_in_image),
            )
        };
        // calculate jacobian wrt point_in_world
        let d2_res_fn = |x: DualV<3>| -> DualV<2> {
            res_fn(
                PinholeCamera::<Dual>::from_params_and_size(
                    &DualV::c(*intrinsics.params()),
                    intrinsics.image_size(),
                ),
                world_from_camera_pose.to_dual_c(),
                x,
                DualV::c(*uv_in_image),
            )
        };

        (
            || VectorValuedMapFromVector::static_fw_autodiff(d0_res_fn, *intrinsics.params()),
            || VectorValuedMapFromVector::static_fw_autodiff(d1_res_fn, VecF64::<6>::zeros()),
            || VectorValuedMapFromVector::static_fw_autodiff(d2_res_fn, point_in_world),
        )
            .make_term(var_kinds, residual, robust_kernel, None)
    }
}
