use sophus_autodiff::linalg::VecF64;
use sophus_lie::{
    Isometry3,
    Isometry3F64,
};
use sophus_sensor::{
    PinholeCamera,
    PinholeCameraF64,
    projections::PerspectiveProjectionImpl,
};

use crate::{
    nlls::cost::evaluated_term::EvaluatedCostTerm,
    prelude::*,
    robust_kernel,
    variables::VarKind,
};

/// Pinhole camera reprojection residual cost term.
///
///
/// `g(p, ʷT꜀, xʷ) = z - πₚ((ʷT꜀)⁻¹ * xʷ)`
///
/// where `ʷT꜀ ∈ SE(3)` is the camera pose in the world reference frame, `xʷ ∈ ℝ³` is the 3D point
/// in world coordinates, `p` camera intrinsic parameters, `π` is the camera projection function,
/// and `z ∈ ℝ²` is the pixel measurement.
#[derive(Clone, Debug)]
pub struct PinholeCameraReprojectionCostTerm {
    /// Pixel measurement.
    pub uv_in_image: VecF64<2>,
    /// Entity indices:
    ///  - 0: ith intrinsics `p`
    ///  - 1: jth camera pose `ʷT꜀`
    ///  - 2: kth 3D point `xʷ`
    pub entity_indices: [usize; 3],
}

impl PinholeCameraReprojectionCostTerm {
    /// Compute the residual:
    ///
    /// `g(p, ʷT꜀, xʷ) = z - πₚ((ʷT꜀)⁻¹ * xʷ)`
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        intrinsics: PinholeCamera<Scalar, 1, DM, DN>,
        world_from_camera: Isometry3<Scalar, 1, DM, DN>,
        point_in_world: Scalar::Vector<3>,
        uv_in_image: Scalar::Vector<2>,
    ) -> Scalar::Vector<2> {
        let point_in_cam = world_from_camera.inverse().transform(point_in_world);
        uv_in_image - intrinsics.cam_proj(point_in_cam)
    }
}

impl HasResidualFn<13, 3, (), (PinholeCameraF64, Isometry3F64, VecF64<3>)>
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
        let camera_from_world = world_from_camera_pose.inverse();
        let x_cam = camera_from_world.transform(point_in_world);
        let proj_z1 = PerspectiveProjectionImpl::<f64, 1, 0, 0>::proj(x_cam);

        let residual = self.uv_in_image - intrinsics.cam_proj(x_cam);

        // d_pi * R_cw  (2×3) — shared by J_pose and J_point, computed once.
        // where d_π = dx_distort_x(proj_z1) * dx_proj_x(x_cam)  [2×3]
        let d_pi_r_cw = intrinsics.dx_distort_x(proj_z1)
            * PerspectiveProjectionImpl::<f64, 1, 0, 0>::dx_proj_x(x_cam)
            * camera_from_world.rotation().matrix();

        // J_cam (2×4): dr/d[fx,fy,cx,cy] = -d(distort)/d(params)
        let d0 = || -intrinsics.dx_distort_params(proj_z1);

        // J_pose (2×6): left perturbation exp(δ)*T_wc, tangent = [ω,ν]
        // dr/dδ = d_pi_r_cw * dx_exp_x_times_point_at_0(x_w)
        let d1 = || d_pi_r_cw * Isometry3F64::dx_exp_x_times_point_at_0(point_in_world);

        // J_pt (2×3): dr/dx_w = -d_pi_r_cw
        let d2 = || -d_pi_r_cw;

        (d0, d1, d2).make(idx, var_kinds, residual, robust_kernel, None)
    }
}
