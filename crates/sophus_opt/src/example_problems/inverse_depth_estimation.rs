use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use sophus_autodiff::{
    dual::{
        DualScalar,
        DualVector,
    },
    linalg::VecF64,
};
use sophus_geo::InverseDepthPoint2F64;
use sophus_lie::{
    Isometry2,
    Isometry2F64,
    Rotation2,
};
use sophus_sensor::Pinhole1dCameraF64;
use sophus_solver::LinearSolverEnum;

use crate::{
    nlls::{
        CostFn,
        CostTerms,
        EqConstraintFn,
        EqConstraints,
        EvaluatedCostTerm,
        EvaluatedEqConstraint,
        OptParams,
        OptimizationSolution,
        optimize_nlls_with_eq_constraints,
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

/// Bundle adjustment reprojection cost term using 2D inverse depth parameterization.
///
/// Two variables: an SE(2) pose (`Isometry2F64`, 3 DOF) and a 2D inverse depth
/// point (`VecF64<2>`, 2 DOF: `(a, ψ) = (x/z, 1/z)`).
///
/// The 1D camera measures only the horizontal coordinate `u`:
///   `u = fx * p_cam[0] / p_cam[1] + cx`
///
/// The scaled transform avoids dividing by ψ:
///   `scaled_p = R * (a, 1)^T + ψ * t`
/// then project: `u = fx * scaled_p[0] / scaled_p[1] + cx`.
///
/// Jacobian blocks:
/// - J_pose (1×3): derivative w.r.t. left se2 perturbation.
/// - J_point (1×2): derivative w.r.t. `(a, ψ)`.
#[derive(Clone, Debug)]
struct InverseDepthBACostTerm {
    /// Measured horizontal pixel coordinate.
    u_measured: f64,
    /// Pinhole focal length.
    focal: f64,
    /// Principal point x.
    cx: f64,
    /// Entity indices: `[pose_idx, point_idx]`.
    entity_indices: [usize; 2],
}

// INPUT_DIM = 3 (se2 pose) + 2 (inverse depth 2d point) = 5, N = 2 arguments
impl HasResidualFn<5, 2, (), (Isometry2F64, VecF64<2>)> for InverseDepthBACostTerm {
    fn idx_ref(&self) -> &[usize; 2] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 2],
        (world_from_camera, a_psi): (Isometry2F64, VecF64<2>),
        var_kinds: [VarKind; 2],
        robust_kernel: Option<robust_kernel::RobustKernel>,
    ) -> EvaluatedCostTerm<5, 2> {
        let camera_from_world = world_from_camera.inverse();
        let rot = camera_from_world.rotation().matrix();
        let t = camera_from_world.translation();

        let inv_pt = InverseDepthPoint2F64::from_params(a_psi);
        let pinhole = Pinhole1dCameraF64::new(self.focal, self.cx);

        // Scaled point in camera frame: R * (a, 1)^T + ψ * t
        let scaled_p = inv_pt.scaled_point(rot, t);

        // Projected pixel (1D)
        let u_proj = pinhole.proj(scaled_p);
        let residual = VecF64::<1>::new(self.u_measured - u_proj);

        // --- J_point (1x2): d(residual)/d(a, ψ) ---
        // Chain: -d(proj)/d(scaled_p) * d(scaled_p)/d(a, ψ)
        let j_point = -pinhole.dx_proj_x(scaled_p) * inv_pt.dx_scaled_point_d_params(rot, t);

        // --- J_pose (1x3): d(residual)/d(se2 perturbation) ---
        // The point in world frame (Cartesian)
        let p_world = inv_pt
            .to_cartesian()
            .expect("ψ must be nonzero for pose Jacobian");
        let p_cam = rot * p_world + t;

        // Chain: d(proj)/d(p_cam) * R_cw * dx_exp_x_times_point_at_0(p_world)
        let d_pi_r_cw = pinhole.dx_proj_x(p_cam) * rot;
        let j_pose = d_pi_r_cw * Isometry2F64::dx_exp_x_times_point_at_0(p_world);

        let d0 = move || j_pose;
        let d1 = move || j_point;

        (d0, d1).make(idx, var_kinds, residual, robust_kernel, None)
    }
}

/// Convert a 2D Cartesian point `(x, z)` to 2D inverse depth `(a, psi)`.
///
/// Panics if `z ≈ 0`. Delegates to [`sophus_geo::InverseDepthPoint2F64`].
fn to_inverse_depth_2d(xz: &VecF64<2>) -> VecF64<2> {
    sophus_geo::InverseDepthPoint2F64::from_cartesian(*xz)
        .expect("z must be nonzero for inverse depth conversion")
        .params
}

/// Convert a 2D inverse depth point `(a, psi)` to Cartesian `(x, z)`.
///
/// Panics if `ψ ≈ 0`. Delegates to [`sophus_geo::InverseDepthPoint2F64`].
pub fn to_cartesian_2d(a_psi: &VecF64<2>) -> VecF64<2> {
    sophus_geo::InverseDepthPoint2F64::from_params(*a_psi)
        .to_cartesian()
        .expect("psi must be nonzero for cartesian conversion")
}

/// Equality constraint on the translation norm of an SE(2) pose.
///
/// `|t(T)| = r` where `T ∈ SE(2)`.
#[derive(Clone, Debug)]
struct TranslationNormConstraint2 {
    /// Target translation norm.
    radius: f64,
    /// Entity index into the poses family.
    entity_indices: [usize; 1],
}

impl TranslationNormConstraint2 {
    fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        pose: Isometry2<Scalar, 1, DM, DN>,
        radius: Scalar,
    ) -> Scalar::Vector<1> {
        let t = pose.translation();
        let norm = t.norm();
        Scalar::Vector::<1>::from_array([norm - radius])
    }
}

impl HasEqConstraintResidualFn<1, 3, 1, (), Isometry2F64> for TranslationNormConstraint2 {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        pose: Isometry2F64,
        var_kinds: [VarKind; 1],
    ) -> EvaluatedEqConstraint<1, 3, 1> {
        let residual = Self::residual(pose, self.radius);

        let dx_res_fn = |x: DualVector<f64, 3, 3, 1>| -> DualVector<f64, 1, 3, 1> {
            let radius_dual = DualScalar::from_f64(self.radius);
            Self::residual::<DualScalar<f64, 3, 1>, 3, 1>(
                Isometry2::exp(x) * pose.to_dual_c(),
                radius_dual,
            )
        };

        (|| dx_res_fn(DualVector::var(VecF64::<3>::zeros())).jacobian(),)
            .make_eq(idx, var_kinds, residual)
    }
}

/// Synthetic 2D inverse depth bundle adjustment problem.
///
/// 5 cameras in a 2D plane observe 20 points parameterized as 2D inverse depth
/// `(a, ψ) = (x/z, 1/z)`.
///
/// Poses are `Isometry2F64` (SE(2), 3 DOF: 1 rotation + 2 translation).
/// The 1D camera measures only the horizontal coordinate `u = fx * x_cam / z_cam + cx`.
///
/// Pose 0 is fixed (gauge fix for rotation + translation).
/// Pose 1 has a scale constraint (`|t₁| = baseline`) fixing the scale gauge.
/// Poses 2-4 and all points are free variables.
pub struct InverseDepthProblem {
    /// Ground truth camera poses (world from camera).
    pub true_world_from_cameras: alloc::vec::Vec<Isometry2F64>,
    /// Perturbed initial camera poses (world from camera).
    world_from_cameras: alloc::vec::Vec<Isometry2F64>,
    /// Ground truth points in 2D inverse depth `(a, psi)`.
    true_points_inv: alloc::vec::Vec<VecF64<2>>,
    /// Initial guess for points in 2D inverse depth.
    pub init_points_inv: alloc::vec::Vec<VecF64<2>>,
    /// BA reprojection observations.
    ba_observations: alloc::vec::Vec<InverseDepthBACostTerm>,
    /// Target translation norm for pose 1 (scale constraint).
    target_radius: f64,
}

impl InverseDepthProblem {
    /// Family name for the camera poses.
    pub const POSES: &str = "poses";
    /// Family name for the inverse depth points.
    pub const POINTS: &str = "points";

    /// Build a new synthetic problem.
    pub fn new() -> Self {
        let mut rng = ChaCha12Rng::from_seed(Default::default());

        let focal = 300.0;
        let cx = 320.0;
        let image_width = 640.0;

        // 5 cameras in the xz-plane, ordered left to right.
        // SE(2) coordinates: (lateral, depth).
        // Camera 0 (far left): fixed. Camera 1 (middle left): scale-constrained.
        // Cameras 2-4: free.
        let true_world_from_cameras = alloc::vec![
            // Camera 0: far left (fixed)
            Isometry2::from_rotation_and_translation(
                Rotation2::exp(VecF64::<1>::new(0.3)),
                VecF64::<2>::new(-4.0, 1.0),
            ),
            // Camera 1: middle left (scale-constrained)
            Isometry2::from_rotation_and_translation(
                Rotation2::exp(VecF64::<1>::new(0.15)),
                VecF64::<2>::new(-2.0, 0.3),
            ),
            // Camera 2: center
            Isometry2::from_rotation_and_translation(
                Rotation2::exp(VecF64::<1>::new(0.0)),
                VecF64::<2>::new(0.0, 0.0),
            ),
            // Camera 3: middle right
            Isometry2::from_rotation_and_translation(
                Rotation2::exp(VecF64::<1>::new(-0.15)),
                VecF64::<2>::new(2.0, 0.3),
            ),
            // Camera 4: far right
            Isometry2::from_rotation_and_translation(
                Rotation2::exp(VecF64::<1>::new(-0.3)),
                VecF64::<2>::new(4.0, 1.0),
            ),
        ];

        // Ground truth 2D points (x, z) via rejection sampling.
        // Generate random points, keep those visible in ≥2 cameras, until we have N.
        let num_target_points = 80;
        let mut true_points_xz: alloc::vec::Vec<VecF64<2>> = alloc::vec![];
        let mut ba_observations = alloc::vec![];

        while true_points_xz.len() < num_target_points {
            // Random depth (log-uniform from 3 to 60) and lateral position
            let z = 3.0 * (60.0_f64 / 3.0).powf(rng.random::<f64>());
            let x = (rng.random::<f64>() - 0.5) * 0.8 * z; // |x/z| < 0.4

            let pt = VecF64::<2>::new(x, z);
            let pt_idx = true_points_xz.len();

            // Count how many cameras see this point
            let mut obs_for_point = alloc::vec![];
            for (pose_idx, pose) in true_world_from_cameras.iter().enumerate() {
                let camera_from_world = pose.inverse();
                let p_cam: VecF64<2> = camera_from_world.transform(pt);
                if p_cam[1] <= 0.1 {
                    continue;
                }
                let u = focal * p_cam[0] / p_cam[1] + cx;
                if !(0.0..image_width).contains(&u) {
                    continue;
                }
                let noise = 0.1 * (rng.random::<f64>() - 0.5);
                obs_for_point.push(InverseDepthBACostTerm {
                    u_measured: u + noise,
                    focal,
                    cx,
                    entity_indices: [pose_idx, pt_idx],
                });
            }

            // Keep point only if visible in ≥2 cameras
            if obs_for_point.len() >= 2 {
                true_points_xz.push(pt);
                ba_observations.extend(obs_for_point);
            }
        }

        let true_points_inv: alloc::vec::Vec<VecF64<2>> =
            true_points_xz.iter().map(to_inverse_depth_2d).collect();

        // Perturbed initial poses (pose 0 stays exact since it will be fixed)
        let world_from_cameras: alloc::vec::Vec<Isometry2F64> = true_world_from_cameras
            .iter()
            .enumerate()
            .map(|(j, &pose)| {
                if j == 0 {
                    return pose; // fixed — no perturbation needed
                }
                let noise = VecF64::<3>::new(
                    (rng.random::<f64>() - 0.5) * 0.04,
                    (rng.random::<f64>() - 0.5) * 0.03,
                    (rng.random::<f64>() - 0.5) * 0.03,
                );
                Isometry2::exp(noise) * pose
            })
            .collect();

        // Perturbed initial guess for points
        let init_points_inv: alloc::vec::Vec<VecF64<2>> = true_points_inv
            .iter()
            .map(|p| {
                *p + VecF64::<2>::new(
                    (rng.random::<f64>() - 0.5) * 0.02,
                    (rng.random::<f64>() - 0.5) * 0.01,
                )
            })
            .collect();

        let target_radius = true_world_from_cameras[1].translation().norm();

        Self {
            true_world_from_cameras,
            world_from_cameras,
            true_points_inv,
            init_points_inv,
            ba_observations,
            target_radius,
        }
    }

    /// Build BA cost function (pose + point variables).
    pub fn build_cost(&self) -> alloc::boxed::Box<dyn crate::nlls::IsCostFn> {
        CostFn::new_boxed(
            (),
            CostTerms::new([Self::POSES, Self::POINTS], self.ba_observations.clone()),
        )
    }

    /// Build the variable families.
    ///
    /// Pose 0 is held constant (gauge fix). Poses 1-4 and all points are free.
    pub fn build_variables(&self) -> VarFamilies {
        let mut fixed = alloc::collections::BTreeMap::new();
        fixed.insert(0, ());

        VarBuilder::new()
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
                VarFamily::new(VarKind::Free, self.init_points_inv.clone()),
            )
            .build()
    }

    /// Build the scale constraint on pose 1's translation norm.
    pub fn build_eq_constraint(&self) -> alloc::boxed::Box<dyn crate::nlls::IsEqConstraintsFn> {
        EqConstraintFn::new_boxed(
            (),
            EqConstraints::new(
                [Self::POSES],
                alloc::vec![TranslationNormConstraint2 {
                    radius: self.target_radius,
                    entity_indices: [1],
                }],
            ),
        )
    }

    /// Build variables with points marginalized for the Schur path.
    pub fn build_variables_schur(&self) -> VarFamilies {
        let mut fixed = alloc::collections::BTreeMap::new();
        fixed.insert(0, ());
        VarBuilder::new()
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
                VarFamily::new(VarKind::Marginalized, self.init_points_inv.clone()),
            )
            .build()
    }

    /// Run the optimization and return the solution.
    ///
    /// Default: `SchurBlockSparseLdlt` with points marginalized.
    pub fn optimize(&self) -> OptimizationSolution {
        use sophus_solver::ldlt::BlockSparseLdlt;
        self.optimize_with_solver(
            LinearSolverEnum::SchurBlockSparseLdlt(BlockSparseLdlt::default()),
            true,
        )
    }

    /// Run the optimization with a specific solver.
    ///
    /// Set `use_schur` to true when the solver is a Schur variant (points marginalized).
    pub fn optimize_with_solver(
        &self,
        solver: LinearSolverEnum,
        use_schur: bool,
    ) -> OptimizationSolution {
        let vars = if use_schur {
            self.build_variables_schur()
        } else {
            self.build_variables()
        };
        optimize_nlls_with_eq_constraints(
            vars,
            alloc::vec![self.build_cost()],
            alloc::vec![self.build_eq_constraint()],
            OptParams {
                num_iterations: 50,
                initial_lm_damping: 1.0,
                parallelize: false,
                solver,
                skip_final_hessian: false,
                ..Default::default()
            },
        )
        .unwrap()
    }

    /// Ground truth points in Cartesian coordinates (x, z).
    pub fn ground_truth_cartesian(&self) -> alloc::vec::Vec<VecF64<2>> {
        self.true_points_inv.iter().map(to_cartesian_2d).collect()
    }

    /// RMS pose translation error and RMS point position error against GT.
    pub fn gt_errors(&self, vars: &VarFamilies) -> (f64, f64) {
        let est_poses = vars.get_members::<Isometry2F64>(Self::POSES);
        let pose_sq: f64 = est_poses
            .iter()
            .zip(self.true_world_from_cameras.iter())
            .map(|(e, gt)| (e.translation() - gt.translation()).norm_squared())
            .sum();
        let pose_rms = (pose_sq / est_poses.len() as f64).sqrt();

        let est_points = vars.get_members::<VecF64<2>>(Self::POINTS);
        let pt_sq: f64 = est_points
            .iter()
            .zip(self.true_points_inv.iter())
            .map(|(e, gt)| {
                let e_xz = to_cartesian_2d(e);
                let gt_xz = to_cartesian_2d(gt);
                (e_xz - gt_xz).norm_squared()
            })
            .sum();
        let point_rms = (pt_sq / est_points.len() as f64).sqrt();

        (pose_rms, point_rms)
    }
}

impl Default for InverseDepthProblem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_depth_ba_converges() {
        let problem = InverseDepthProblem::new();
        let solution = problem.optimize();

        let (pose_rms, pt_rms) = problem.gt_errors(&solution.variables);

        // Poses should converge well (small perturbation, good geometry)
        assert!(pose_rms < 0.05, "pose_rms={pose_rms:.4} >= 0.05");

        // Points: distant points have larger error, so use a generous threshold
        assert!(pt_rms < 5.0, "pt_rms={pt_rms:.4} >= 5.0");

        // Check individual points in inverse depth space — well-conditioned even for distant pts.
        let est_points = solution
            .variables
            .get_members::<VecF64<2>>(InverseDepthProblem::POINTS);

        for (est, gt) in est_points.iter().zip(problem.true_points_inv.iter()) {
            let err = (est - gt).norm();
            assert!(
                err < 0.05,
                "inverse depth error {err:.6}, est={est:?}, gt={gt:?}"
            );
        }
    }

    #[test]
    fn inverse_depth_point_covariance_positive_definite() {
        let problem = InverseDepthProblem::new();
        let mut solution = problem.optimize();

        let cov = solution
            .covariance_block(
                InverseDepthProblem::POINTS,
                7,
                InverseDepthProblem::POINTS,
                7,
            )
            .expect("point 7 covariance");

        assert_eq!(cov.nrows(), 2);
        assert_eq!(cov.ncols(), 2);

        let eig =
            nalgebra::SymmetricEigen::new(nalgebra::Matrix2::from_iterator(cov.iter().copied()));
        let positive_count = eig.eigenvalues.iter().filter(|&&ev| ev > 1e-15).count();
        assert!(
            positive_count == 2,
            "expected 2 positive eigenvalues for point 7, got {positive_count}"
        );
    }

    #[test]
    fn inverse_depth_pose_covariance_positive_definite() {
        let problem = InverseDepthProblem::new();
        let mut solution = problem.optimize();

        // Pose 1 (first free pose — pose 0 is fixed).
        let cov = solution
            .covariance_block(InverseDepthProblem::POSES, 1, InverseDepthProblem::POSES, 1)
            .expect("pose 1 covariance");

        assert_eq!(cov.nrows(), 3);
        assert_eq!(cov.ncols(), 3);

        // Translation sub-block: rows/cols 1..3 of the 3×3 se2 covariance.
        // Pose 1 has a scale constraint (|t₁| = r), so one translation DOF is constrained.
        // The constrained covariance should have exactly 1 positive eigenvalue (tangential)
        // and 1 near-zero eigenvalue (radial — constrained by |t| = r).
        let trans_cov = cov.view((1, 1), (2, 2)).into_owned();
        let eig = nalgebra::SymmetricEigen::new(nalgebra::Matrix2::from_iterator(
            trans_cov.iter().copied(),
        ));
        let positive_count = eig.eigenvalues.iter().filter(|&&ev| ev > 1e-15).count();
        assert!(
            positive_count >= 1,
            "expected at least 1 positive eigenvalue, got {positive_count}: {:?}",
            eig.eigenvalues
        );
        // Verify the constraint effect: one eigenvalue should be much smaller than the other.
        let mut evs: Vec<f64> = eig.eigenvalues.iter().copied().collect();
        evs.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!(
            evs[1].abs() < 1e-6,
            "radial eigenvalue should be near-zero due to scale constraint, got {:.2e}",
            evs[1]
        );
    }

    #[test]
    fn covariance_block_returns_none_for_fixed_pose() {
        let problem = InverseDepthProblem::new();
        let mut solution = problem.optimize();

        // Pose 0 is conditioned (fixed) — covariance should be None.
        let result =
            solution.covariance_block(InverseDepthProblem::POSES, 0, InverseDepthProblem::POSES, 0);
        assert!(result.is_none(), "fixed pose should have no covariance");
    }

    #[test]
    fn covariance_block_cross_family() {
        let problem = InverseDepthProblem::new();
        let mut solution = problem.optimize();

        // Cross-family: point 0 × pose 1. Should return a 2×3 matrix.
        let cov = solution
            .covariance_block(
                InverseDepthProblem::POINTS,
                0,
                InverseDepthProblem::POSES,
                1,
            )
            .expect("point-pose cross covariance");
        assert_eq!(cov.nrows(), 2);
        assert_eq!(cov.ncols(), 3);
    }

    #[test]
    fn inverse_depth_covariance_depth_dominates() {
        let problem = InverseDepthProblem::new();
        let mut solution = problem.optimize();

        let est_points = solution
            .variables
            .get_members::<VecF64<2>>(InverseDepthProblem::POINTS);

        // Use a distant point (last layer) — depth eigenvalue should dominate.
        let pt_idx = est_points.len() - 1;
        let cov_apsi = solution
            .covariance_block(
                InverseDepthProblem::POINTS,
                pt_idx,
                InverseDepthProblem::POINTS,
                pt_idx,
            )
            .expect("distant point covariance");

        // Map covariance to Cartesian via Σ_cart = J · Σ_(a,ψ) · Jᵀ.
        // Cartesian mapping: (a, ψ) → (a/ψ, 1/ψ)
        // J = [[1/psi, -a/psi^2],
        //      [0,     -1/psi^2]]
        let a_psi = est_points[pt_idx];
        let a = a_psi[0];
        let psi = a_psi[1];
        let psi2 = psi * psi;

        let j_cart = nalgebra::Matrix2::new(1.0 / psi, -a / psi2, 0.0, -1.0 / psi2);

        let cov_apsi_mat = nalgebra::Matrix2::from_iterator(cov_apsi.iter().copied());
        let cov_cart = j_cart * cov_apsi_mat * j_cart.transpose();

        let eig = nalgebra::SymmetricEigen::new(cov_cart);
        let mut evs: alloc::vec::Vec<f64> = eig.eigenvalues.iter().copied().collect();
        evs.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // The largest eigenvalue (depth direction) should be significantly larger
        // than the smallest (lateral direction) for a distant point.
        let ratio = evs[0] / evs[1];
        assert!(
            ratio > 5.0,
            "eigenvalue ratio {ratio:.2} should be > 5 for distant point (depth >> lateral)"
        );
    }

    // Note: Schur + constraint on marginalized family death test is in
    // solver_validation::tests::death_schur_eq_on_marg.
}
