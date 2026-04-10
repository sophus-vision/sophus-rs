use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use sophus_autodiff::linalg::VecF64;
use sophus_image::ImageSize;
use sophus_lie::{
    Isometry3,
    Isometry3F64,
    Rotation3,
};
use sophus_sensor::{
    EnhancedUnifiedCamera,
    EnhancedUnifiedCameraF64,
    projections::PerspectiveProjectionImpl,
};
use sophus_solver::LinearSolverEnum;

use crate::{
    nlls::{
        CostFn,
        CostTerms,
        EvaluatedCostTerm,
        OptParams,
        optimize_nlls,
    },
    prelude::*,
    robust_kernel,
    variables::{
        VarBuilder,
        VarFamilies,
        VarFamily,
        VarKind,
    },
};

extern crate alloc;

/// Enhanced Unified Camera Model (EUCM) reprojection residual cost term.
///
/// `g(p, ʷT꜀, xʷ) = z - πₚ((ʷT꜀)⁻¹ * xʷ)`
///
/// where `p = [fx, fy, cx, cy, alpha, beta]` are the 6 EUCM intrinsic parameters.
/// The alpha/beta distortion breaks the focal-depth scale ambiguity that plagues
/// pinhole-only bundle adjustment.
#[derive(Clone, Debug)]
pub struct EucmCameraReprojectionCostTerm {
    /// Pixel measurement.
    pub uv_in_image: VecF64<2>,
    /// Entity indices:
    ///  - 0: ith intrinsics `p`
    ///  - 1: jth camera pose `ʷT꜀`
    ///  - 2: kth 3D point `xʷ`
    pub entity_indices: [usize; 3],
}

impl EucmCameraReprojectionCostTerm {
    /// Compute the residual: `g(p, ʷT꜀, xʷ) = z - πₚ((ʷT꜀)⁻¹ * xʷ)`
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        intrinsics: EnhancedUnifiedCamera<Scalar, 1, DM, DN>,
        world_from_camera: Isometry3<Scalar, 1, DM, DN>,
        point_in_world: Scalar::Vector<3>,
        uv_in_image: Scalar::Vector<2>,
    ) -> Scalar::Vector<2> {
        let point_in_cam = world_from_camera.inverse().transform(point_in_world);
        uv_in_image - intrinsics.cam_proj(point_in_cam)
    }
}

// INPUT_DIM = 6 (eucm params) + 6 (se3 pose) + 3 (point) = 15
impl HasResidualFn<15, 3, (), (EnhancedUnifiedCameraF64, Isometry3F64, VecF64<3>)>
    for EucmCameraReprojectionCostTerm
{
    fn idx_ref(&self) -> &[usize; 3] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 3],
        (intrinsics, world_from_camera_pose, point_in_world): (
            EnhancedUnifiedCameraF64,
            Isometry3F64,
            VecF64<3>,
        ),
        var_kinds: [VarKind; 3],
        robust_kernel: Option<robust_kernel::RobustKernel>,
    ) -> EvaluatedCostTerm<15, 3> {
        let camera_from_world = world_from_camera_pose.inverse();
        let x_cam = camera_from_world.transform(point_in_world);
        let proj_z1 = PerspectiveProjectionImpl::<f64, 1, 0, 0>::proj(x_cam);

        let residual = self.uv_in_image - intrinsics.cam_proj(x_cam);

        // d_pi * R_cw  (2×3) — shared by J_pose and J_point, computed once.
        let d_pi_r_cw = intrinsics.dx_distort_x(proj_z1)
            * PerspectiveProjectionImpl::<f64, 1, 0, 0>::dx_proj_x(x_cam)
            * camera_from_world.rotation().matrix();

        // J_cam (2×6): dr/d[fx,fy,cx,cy,alpha,beta]
        let d0 = || -intrinsics.dx_distort_params(proj_z1);

        // J_pose (2×6): left perturbation exp(δ)*T_wc
        let d1 = || d_pi_r_cw * Isometry3F64::dx_exp_x_times_point_at_0(point_in_world);

        // J_pt (2×3): dr/dx_w = -d_pi_r_cw
        let d2 = || -d_pi_r_cw;

        (d0, d1, d2).make(idx, var_kinds, residual, robust_kernel, None)
    }
}

/// True focal length (fx = fy) for the synthetic EUCM camera.
pub const TRUE_FOCAL: f64 = 300.0;

const IMAGE_WIDTH: usize = 640;
const IMAGE_HEIGHT: usize = 480;
const RING_RADIUS: f64 = 8.0;
const BOX_CENTER_Z: f64 = 10.0;

/// Synthetic bundle adjustment problem.
///
/// `num_cameras` cameras are placed on a ring of radius `RING_RADIUS`, all
/// looking inward at an open box of `num_points` 3-D points centred at
/// `(0, 0, BOX_CENTER_Z)`. A single shared EUCM intrinsic is estimated.
pub struct BaProblem {
    /// Ground-truth shared intrinsic.
    pub true_intrinsics: EnhancedUnifiedCameraF64,
    /// Ground-truth world-from-camera poses.
    pub true_world_from_cameras: alloc::vec::Vec<Isometry3F64>,
    /// Ground-truth 3-D points (compacted to those with ≥2 observations).
    pub true_points_in_world: alloc::vec::Vec<VecF64<3>>,
    /// Perturbed initial intrinsic (single-element vec, shared across all poses).
    pub init_intrinsics: alloc::vec::Vec<EnhancedUnifiedCameraF64>,
    /// Perturbed initial world-from-camera poses.
    pub world_from_cameras: alloc::vec::Vec<Isometry3F64>,
    /// Perturbed initial 3-D points.
    pub points_in_world: alloc::vec::Vec<VecF64<3>>,
    /// Reprojection observations (noisily generated from GT).
    pub observations: alloc::vec::Vec<EucmCameraReprojectionCostTerm>,
}

impl BaProblem {
    /// Family name for the shared camera intrinsic.
    pub const CAMS: &str = "cams";
    /// Family name for the camera poses.
    pub const POSES: &str = "poses";
    /// Family name for the 3-D points.
    pub const POINTS: &str = "points";

    /// Build a new synthetic problem.
    ///
    /// `num_cameras` cameras are placed uniformly on a ring; pose 0 and pose 1
    /// are kept fixed to resolve the gauge freedom.
    pub fn new(num_cameras: usize, num_points: usize) -> Self {
        let mut rng = ChaCha12Rng::from_seed(Default::default());

        let image_size = ImageSize {
            width: IMAGE_WIDTH,
            height: IMAGE_HEIGHT,
        };

        // Single shared EUCM intrinsic: alpha=0.5, beta=1.0 breaks the focal-depth
        // scale ambiguity that plagues pinhole bundle adjustment.
        let true_intrinsics = EnhancedUnifiedCameraF64::from_params_and_size(
            VecF64::<6>::new(TRUE_FOCAL, TRUE_FOCAL, 320.0, 240.0, 0.5, 1.0),
            image_size,
        );

        // Cameras on a ring in the XZ plane, all looking inward at the box centre.
        let true_world_from_cameras: alloc::vec::Vec<Isometry3F64> = (0..num_cameras)
            .map(|j| {
                let theta = core::f64::consts::TAU * j as f64 / num_cameras as f64;
                Isometry3::from_rotation_and_translation(
                    Rotation3::exp(VecF64::<3>::new(0.0, -theta, 0.0)),
                    VecF64::<3>::new(
                        RING_RADIUS * theta.sin(),
                        0.0,
                        BOX_CENTER_Z - RING_RADIUS * theta.cos(),
                    ),
                )
            })
            .collect();

        // Points on an open box (5 faces; front face open toward the cameras).
        // Box spans x∈[-3,3], y∈[-2,2], z∈[8,12]; open at z=8.
        let (bw, bh, z_near, z_far) = (3.0_f64, 2.0_f64, 8.0_f64, 12.0_f64);
        let all_true_points: alloc::vec::Vec<VecF64<3>> = (0..num_points)
            .map(|k| match k % 5 {
                0 => VecF64::<3>::new(
                    (rng.random::<f64>() - 0.5) * 2.0 * bw,
                    (rng.random::<f64>() - 0.5) * 2.0 * bh,
                    z_far,
                ),
                1 => VecF64::<3>::new(
                    -bw,
                    (rng.random::<f64>() - 0.5) * 2.0 * bh,
                    z_near + rng.random::<f64>() * (z_far - z_near),
                ),
                2 => VecF64::<3>::new(
                    bw,
                    (rng.random::<f64>() - 0.5) * 2.0 * bh,
                    z_near + rng.random::<f64>() * (z_far - z_near),
                ),
                3 => VecF64::<3>::new(
                    (rng.random::<f64>() - 0.5) * 2.0 * bw,
                    bh,
                    z_near + rng.random::<f64>() * (z_far - z_near),
                ),
                _ => VecF64::<3>::new(
                    (rng.random::<f64>() - 0.5) * 2.0 * bw,
                    -bh,
                    z_near + rng.random::<f64>() * (z_far - z_near),
                ),
            })
            .collect();

        // Build observations; count per-point to discard under-constrained points.
        let mut raw_obs: alloc::vec::Vec<EucmCameraReprojectionCostTerm> = alloc::vec![];
        let mut point_obs_count: alloc::vec::Vec<usize> = alloc::vec![0; num_points];

        // Max visibility distance: cameras only see nearby points.
        // This creates realistic sparse coupling (not every camera sees every point).
        // Only applied when there are enough cameras for sufficient coverage.
        let max_vis_dist = if num_cameras >= 20 {
            RING_RADIUS * 1.0
        } else {
            f64::INFINITY // small problems: all cameras see all points
        };

        for (pose_idx, pose) in true_world_from_cameras.iter().enumerate() {
            let cam_pos = pose.translation();
            let cam_from_world = pose.inverse();
            for (point_idx, pt) in all_true_points.iter().enumerate() {
                // Distance check: skip points too far from this camera.
                let dist = (cam_pos - *pt).norm();
                if dist > max_vis_dist {
                    continue;
                }
                let p_in_cam = cam_from_world.transform(*pt);
                if p_in_cam[2] <= 0.1 {
                    continue;
                }
                let uv = true_intrinsics.cam_proj(p_in_cam);
                if uv[0] < 0.0
                    || uv[0] >= IMAGE_WIDTH as f64
                    || uv[1] < 0.0
                    || uv[1] >= IMAGE_HEIGHT as f64
                {
                    continue;
                }
                let noise =
                    0.15 * VecF64::<2>::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5);
                raw_obs.push(EucmCameraReprojectionCostTerm {
                    uv_in_image: uv + noise,
                    entity_indices: [0, pose_idx, point_idx],
                });
                point_obs_count[point_idx] += 1;
            }
        }

        // Compact: keep only points with >=2 observations; remap indices.
        let mut old_to_new: alloc::vec::Vec<Option<usize>> = alloc::vec![None; num_points];
        let mut new_idx = 0usize;
        for (old, &cnt) in point_obs_count.iter().enumerate() {
            if cnt >= 2 {
                old_to_new[old] = Some(new_idx);
                new_idx += 1;
            }
        }

        let true_points_in_world: alloc::vec::Vec<VecF64<3>> = all_true_points
            .iter()
            .enumerate()
            .filter_map(|(i, p)| old_to_new[i].map(|_| *p))
            .collect();

        let observations: alloc::vec::Vec<EucmCameraReprojectionCostTerm> = raw_obs
            .into_iter()
            .filter_map(|obs| {
                old_to_new[obs.entity_indices[2]].map(|new_pt| EucmCameraReprojectionCostTerm {
                    uv_in_image: obs.uv_in_image,
                    entity_indices: [0, obs.entity_indices[1], new_pt],
                })
            })
            .collect();

        assert!(
            !true_points_in_world.is_empty(),
            "BaProblem: no points with >=2 observations"
        );

        // Perturbed initial guesses.
        // Small perturbation from GT — enough to exercise the optimizer but close
        // enough that 20 iterations reach the noise floor.
        let init_intrinsics = alloc::vec![EnhancedUnifiedCameraF64::from_params_and_size(
            VecF64::<6>::new(
                TRUE_FOCAL * 0.99,
                TRUE_FOCAL * 0.99,
                321.0,
                241.0,
                0.495,
                1.0
            ),
            image_size,
        )];

        let world_from_cameras: alloc::vec::Vec<Isometry3F64> = true_world_from_cameras
            .iter()
            .enumerate()
            .map(|(j, &pose)| {
                if j == 0 || j == 1 {
                    return pose; // fixed — no perturbation needed
                }
                let noise = VecF64::<6>::new(
                    (rng.random::<f64>() - 0.5) * 0.08,
                    (rng.random::<f64>() - 0.5) * 0.08,
                    (rng.random::<f64>() - 0.5) * 0.04,
                    (rng.random::<f64>() - 0.5) * 0.06,
                    (rng.random::<f64>() - 0.5) * 0.06,
                    (rng.random::<f64>() - 0.5) * 0.02,
                );
                Isometry3::exp(noise) * pose
            })
            .collect();

        let points_in_world: alloc::vec::Vec<VecF64<3>> = true_points_in_world
            .iter()
            .map(|p| {
                *p + VecF64::<3>::new(
                    (rng.random::<f64>() - 0.5) * 0.4,
                    (rng.random::<f64>() - 0.5) * 0.4,
                    (rng.random::<f64>() - 0.5) * 0.4,
                )
            })
            .collect();

        Self {
            true_intrinsics,
            true_world_from_cameras,
            true_points_in_world,
            init_intrinsics,
            world_from_cameras,
            points_in_world,
            observations,
        }
    }

    /// Build the initial variable families.
    ///
    /// Poses 0 and 1 are held constant to resolve the gauge freedom.
    /// When `use_schur` is true the point family is marginalized.
    pub fn build_initial_variables(&self, use_schur: bool) -> VarFamilies {
        let mut fixed = alloc::collections::BTreeMap::new();
        fixed.insert(0, ());
        fixed.insert(1, ());

        let points_kind = if use_schur {
            VarKind::Marginalized
        } else {
            VarKind::Free
        };

        VarBuilder::new()
            .add_family(
                Self::CAMS,
                VarFamily::new(VarKind::Free, self.init_intrinsics.clone()),
            )
            .add_family(
                Self::POSES,
                VarFamily::new_with_const_ids(
                    VarKind::Free,
                    self.world_from_cameras.clone(),
                    fixed,
                ),
            )
            .add_family(
                Self::POINTS,
                VarFamily::new(points_kind, self.points_in_world.clone()),
            )
            .build()
    }

    /// Build the reprojection cost function.
    pub fn build_cost(&self) -> alloc::boxed::Box<dyn crate::nlls::IsCostFn> {
        CostFn::new_boxed(
            (),
            CostTerms::new(
                [Self::CAMS, Self::POSES, Self::POINTS],
                self.observations.clone(),
            ),
        )
    }

    /// Number of reprojection observations.
    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }

    /// RMS pose translation error and RMS point position error against GT.
    pub fn gt_errors(&self, vars: &VarFamilies) -> (f64, f64) {
        let est_poses = vars.get_members::<Isometry3F64>(Self::POSES);
        let pose_sq: f64 = est_poses
            .iter()
            .zip(self.true_world_from_cameras.iter())
            .map(|(e, gt)| (e.translation() - gt.translation()).norm_squared())
            .sum();
        let pose_rms = (pose_sq / est_poses.len() as f64).sqrt();

        let est_points = vars.get_members::<VecF64<3>>(Self::POINTS);
        let pt_sq: f64 = est_points
            .iter()
            .zip(self.true_points_in_world.iter())
            .map(|(e, gt)| (e - gt).norm_squared())
            .sum();
        let point_rms = (pt_sq / est_points.len() as f64).sqrt();

        (pose_rms, point_rms)
    }

    fn run(
        &self,
        cams_kind: VarKind,
        points_kind: VarKind,
        solver: LinearSolverEnum,
        parallelize: bool,
    ) -> VarFamilies {
        optimize_nlls(
            self.build_variables(cams_kind, points_kind),
            alloc::vec![self.build_cost()],
            alloc::vec![],
            OptParams {
                num_iterations: 20,
                initial_lm_damping: 1.0,
                parallelize,
                solver,
                skip_final_hessian: false,
                ..Default::default()
            },
        )
        .unwrap()
        .variables
    }

    fn build_variables(&self, cams_kind: VarKind, points_kind: VarKind) -> VarFamilies {
        let mut fixed = alloc::collections::BTreeMap::new();
        fixed.insert(0, ());
        fixed.insert(1, ());

        // For the conditioned path the intrinsics are assumed known (calibrated camera),
        // so we start from GT rather than the perturbed initial guess.
        let cam_init = if cams_kind == VarKind::Conditioned {
            alloc::vec![self.true_intrinsics]
        } else {
            self.init_intrinsics.clone()
        };

        VarBuilder::new()
            .add_family(Self::CAMS, VarFamily::new(cams_kind, cam_init))
            .add_family(
                Self::POSES,
                VarFamily::new_with_const_ids(
                    VarKind::Free,
                    self.world_from_cameras.clone(),
                    fixed,
                ),
            )
            .add_family(
                Self::POINTS,
                VarFamily::new(points_kind, self.points_in_world.clone()),
            )
            .build()
    }

    /// Optimize with all variables free (standard solve, no Schur complement).
    pub fn optimize_all_free(&self, solver: LinearSolverEnum, parallelize: bool) -> VarFamilies {
        self.run(VarKind::Free, VarKind::Free, solver, parallelize)
    }

    /// Optimize with points marginalized (Schur complement path).
    pub fn optimize_marg_points(&self, solver: LinearSolverEnum, parallelize: bool) -> VarFamilies {
        self.run(VarKind::Free, VarKind::Marginalized, solver, parallelize)
    }

    /// Optimize with cameras conditioned (fixed intrinsics), all other variables free.
    pub fn optimize_all_free_const_cams(
        &self,
        solver: LinearSolverEnum,
        parallelize: bool,
    ) -> VarFamilies {
        self.run(VarKind::Conditioned, VarKind::Free, solver, parallelize)
    }

    /// Optimize with cameras conditioned (fixed intrinsics), points marginalized.
    pub fn optimize_marg_points_const_cams(
        &self,
        solver: LinearSolverEnum,
        parallelize: bool,
    ) -> VarFamilies {
        self.run(
            VarKind::Conditioned,
            VarKind::Marginalized,
            solver,
            parallelize,
        )
    }
}

#[cfg(test)]
mod tests {
    use sophus_solver::{
        LinearSolverEnum,
        ldlt::BlockSparseLdlt,
    };

    use super::BaProblem;

    #[test]
    fn ba_converges() {
        let ba = BaProblem::new(4, 20);

        // All-free path with BlockSparseLdlt.
        let vars_free = ba.optimize_all_free(
            LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default()),
            false,
        );
        let (pose_rms_free, pt_rms_free) = ba.gt_errors(&vars_free);
        assert!(
            pose_rms_free < 0.01,
            "all-free pose_rms={pose_rms_free:.4} >= 0.01"
        );
        assert!(pt_rms_free < 0.1, "all-free pt_rms={pt_rms_free:.4} >= 0.1");

        // Marginalized-points path with SchurBlockSparseLdlt.
        let vars_marg = ba.optimize_marg_points(
            LinearSolverEnum::SchurBlockSparseLdlt(BlockSparseLdlt::default()),
            false,
        );
        let (pose_rms_marg, pt_rms_marg) = ba.gt_errors(&vars_marg);
        assert!(
            pose_rms_marg < 0.01,
            "marg-points pose_rms={pose_rms_marg:.4} >= 0.01"
        );
        assert!(
            pt_rms_marg < 0.1,
            "marg-points pt_rms={pt_rms_marg:.4} >= 0.1"
        );
    }
}
