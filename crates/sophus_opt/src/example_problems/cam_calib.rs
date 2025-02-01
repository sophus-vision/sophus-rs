use crate::nlls::functor_library::costs::isometry3_prior::Isometry3PriorCostTerm;
use crate::nlls::functor_library::costs::pinhole_reprojection::PinholeCameraReprojectionCostTerm;
use crate::nlls::optimize;
use crate::nlls::quadratic_cost::cost_fn::CostFn;
use crate::nlls::quadratic_cost::cost_term::CostTerms;
use crate::nlls::LinearSolverType;
use crate::nlls::OptParams;
use crate::prelude::*;
use crate::robust_kernel::HuberKernel;
use crate::variables::var_builder::VarBuilder;
use crate::variables::var_family::VarFamily;
use crate::variables::VarKind;
use sophus_autodiff::linalg::MatF64;
use sophus_autodiff::linalg::VecF64;
use sophus_image::ImageSize;
use sophus_lie::Isometry3;
use sophus_lie::Isometry3F64;
use sophus_lie::Rotation3;
use sophus_sensor::camera_enum::perspective_camera::PinholeCameraF64;

extern crate alloc;

/// Camera calibration problem
#[derive(Clone)]
pub struct CamCalibProblem {
    /// intrinsics
    pub intrinsics: PinholeCameraF64,
    /// world from camera isometries
    pub world_from_cameras: alloc::vec::Vec<Isometry3F64>,
    /// points in world
    pub points_in_world: alloc::vec::Vec<VecF64<3>>,
    /// observations
    pub observations: alloc::vec::Vec<PinholeCameraReprojectionCostTerm>,

    /// true intrinsics
    pub true_intrinsics: PinholeCameraF64,
    /// true world from camera isometries
    pub true_world_from_cameras: alloc::vec::Vec<Isometry3F64>,
    /// true points in world
    pub true_points_in_world: alloc::vec::Vec<VecF64<3>>,
}

impl CamCalibProblem {
    const CAMS: &str = "cams";
    const POSES: &str = "poses";
    const POINTS: &str = "points";

    /// create new camera calibration problem
    pub fn new(spurious_matches: bool) -> Self {
        let true_world_from_cameras = alloc::vec![
            Isometry3::identity(),
            Isometry3::from_translation_and_factor(
                VecF64::<3>::new(0.0, 1.0, 0.0),
                Rotation3::identity(),
            ),
            Isometry3::from_translation_and_factor(
                VecF64::<3>::new(0.0, 2.0, 0.0),
                Rotation3::identity(),
            ),
        ];
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(0);

        let image_size = ImageSize {
            width: 640,
            height: 480,
        };

        let true_intrinsics = PinholeCameraF64::from_params_and_size(
            VecF64::<4>::new(600.0, 600.0, 320.0, 240.0),
            image_size,
        );

        let mut observations = alloc::vec![];
        let mut true_points_in_world = alloc::vec![];

        for i in 0..40 {
            let u = rng.gen::<f64>() * (image_size.width as f64 - 1.0);
            let v = rng.gen::<f64>() * (image_size.height as f64 - 1.0);
            let true_uv_in_img0 = VecF64::<2>::new(u, v);
            let img_noise = VecF64::<2>::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);
            observations.push(PinholeCameraReprojectionCostTerm {
                uv_in_image: true_uv_in_img0 + img_noise,
                entity_indices: [0, 0, i],
            });

            let true_point_in_cam0 = true_intrinsics.cam_unproj_with_z(true_uv_in_img0, 10.0);
            true_points_in_world.push(true_world_from_cameras[0].transform(true_point_in_cam0));
            let true_uv_in_img0_proof = true_intrinsics.cam_proj(true_point_in_cam0);
            approx::assert_abs_diff_eq!(true_uv_in_img0, true_uv_in_img0_proof, epsilon = 0.1);

            let true_cam1_from_cam0 =
                true_world_from_cameras[1].inverse() * true_world_from_cameras[0];
            let true_point_in_cam1 = true_cam1_from_cam0.transform(true_point_in_cam0);
            let true_uv_in_img1 = true_intrinsics.cam_proj(true_point_in_cam1);
            let img_noise = VecF64::<2>::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);

            if spurious_matches && i == 0 {
                let u = rng.gen::<f64>() * (image_size.width as f64 - 1.0);
                let v = rng.gen::<f64>() * (image_size.height as f64 - 1.0);
                observations.push(PinholeCameraReprojectionCostTerm {
                    uv_in_image: VecF64::<2>::new(u, v),
                    entity_indices: [0, 1, i],
                });
            } else {
                observations.push(PinholeCameraReprojectionCostTerm {
                    uv_in_image: true_uv_in_img1 + img_noise,
                    entity_indices: [0, 1, i],
                });
            }

            let true_cam2_from_cam0 =
                true_world_from_cameras[2].inverse() * true_world_from_cameras[0];
            let true_point_in_cam2 = true_cam2_from_cam0.transform(true_point_in_cam0);
            let true_uv_in_img2 = true_intrinsics.cam_proj(true_point_in_cam2);
            let img_noise = VecF64::<2>::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);
            observations.push(PinholeCameraReprojectionCostTerm {
                uv_in_image: true_uv_in_img2 + img_noise,
                entity_indices: [0, 2, i],
            });
        }

        Self {
            world_from_cameras: alloc::vec![
                Isometry3F64::identity(),
                true_world_from_cameras[1],
                true_world_from_cameras[1],
            ],
            observations,
            intrinsics: true_intrinsics,
            points_in_world: true_points_in_world.clone(),

            true_intrinsics,
            true_points_in_world,
            true_world_from_cameras,
        }
    }

    /// optimize with two poses fixed
    pub fn optimize_with_two_poses_fixed(
        &self,
        intrinsics_var_kind: VarKind,
        solver: LinearSolverType,
    ) {
        let reproj_obs = CostTerms::new(
            [Self::CAMS, Self::POSES, Self::POINTS],
            self.observations.clone(),
        );

        let mut id = alloc::collections::BTreeMap::new();
        id.insert(0, ());
        id.insert(1, ());

        let variables = VarBuilder::new()
            .add_family(
                Self::CAMS,
                VarFamily::new(intrinsics_var_kind, alloc::vec![self.intrinsics]),
            )
            .add_family(
                Self::POSES,
                VarFamily::new_with_const_ids(
                    VarKind::Free,
                    self.world_from_cameras.clone(),
                    id.clone(),
                ),
            )
            .add_family(
                Self::POINTS,
                VarFamily::new(VarKind::Free, self.points_in_world.clone()),
            )
            .build();

        let solution = optimize(
            variables,
            alloc::vec![
                // robust kernel to deal with outliers
                CostFn::new_robust(
                    (),
                    reproj_obs.clone(),
                    crate::robust_kernel::RobustKernel::Huber(HuberKernel::new(1.0)),
                ),
            ],
            OptParams {
                num_iterations: 25, // should converge in single iteration
                initial_lm_damping: 1.0,
                parallelize: true,
                linear_solver: solver,
            },
        )
        .unwrap();
        let refined_variables = solution.variables;
        let refined_world_from_robot = refined_variables.get_members::<Isometry3F64>(Self::POSES);

        approx::assert_abs_diff_eq!(
            refined_world_from_robot[2].translation(),
            self.true_world_from_cameras[2].translation(),
            epsilon = 0.05
        );
    }

    /// optimize with priors
    pub fn optimize_with_priors(&self, solver: LinearSolverType) {
        let priors = CostTerms::new(
            [Self::POSES],
            alloc::vec![
                Isometry3PriorCostTerm {
                    entity_indices: [0],
                    isometry_prior_mean: (
                        self.true_world_from_cameras[0],
                        MatF64::<6, 6>::new_scaling(10000.0),
                    ),
                },
                Isometry3PriorCostTerm {
                    entity_indices: [1],
                    isometry_prior_mean: (
                        self.true_world_from_cameras[1],
                        MatF64::<6, 6>::new_scaling(10000.0),
                    ),
                },
            ],
        );

        let reproj_obs = CostTerms::new(
            [Self::CAMS, Self::POSES, Self::POINTS],
            self.observations.clone(),
        );

        let var_pool = VarBuilder::new()
            .add_family(
                Self::CAMS,
                VarFamily::new(VarKind::Conditioned, alloc::vec![self.intrinsics]),
            )
            .add_family(
                Self::POSES,
                VarFamily::new(VarKind::Free, self.world_from_cameras.clone()),
            )
            .add_family(
                Self::POINTS,
                VarFamily::new(VarKind::Free, self.points_in_world.clone()),
            )
            .build();

        let solution = optimize(
            var_pool,
            alloc::vec![
                CostFn::new_box((), priors.clone()),
                CostFn::new_box((), reproj_obs.clone()),
            ],
            OptParams {
                num_iterations: 10,
                initial_lm_damping: 1.0,
                parallelize: true,
                linear_solver: solver,
            },
        )
        .unwrap();
        let refined_variables = solution.variables;
        let refined_world_from_robot = refined_variables.get_members::<Isometry3F64>(Self::POSES);

        approx::assert_abs_diff_eq!(
            refined_world_from_robot[2].translation(),
            self.true_world_from_cameras[2].translation(),
            epsilon = 0.1
        );
    }
}

mod tests {

    #[test]
    fn simple_cam_tests() {
        use crate::example_problems::cam_calib::CamCalibProblem;
        use crate::nlls::LinearSolverType;
        use crate::variables::VarKind;

        for solver in LinearSolverType::sparse_solvers() {
            CamCalibProblem::new(true).optimize_with_two_poses_fixed(VarKind::Free, solver);
            CamCalibProblem::new(false).optimize_with_two_poses_fixed(VarKind::Conditioned, solver);
            CamCalibProblem::new(false).optimize_with_priors(solver);
        }
    }
}
