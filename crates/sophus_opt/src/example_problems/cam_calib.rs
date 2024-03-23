use crate::cost_fn::CostFn;
use crate::cost_fn::CostSignature;
use crate::example_problems::cost_fn::isometry3_prior::Isometry3PriorCostFn;
use crate::example_problems::cost_fn::isometry3_prior::Isometry3PriorTermSignature;
use crate::example_problems::cost_fn::reprojection::ReprojTermSignature;
use crate::example_problems::cost_fn::reprojection::ReprojectionCostFn;
use crate::nlls::optimize;
use crate::nlls::OptParams;
use crate::robust_kernel::HuberKernel;
use crate::variables::VarFamily;
use crate::variables::VarKind;
use crate::variables::VarPoolBuilder;

use sophus_calculus::types::MatF64;
use sophus_calculus::types::VecF64;
use sophus_image::image_view::ImageSize;
use sophus_lie::isometry3::Isometry3;
use sophus_lie::rotation3::Rotation3;
use sophus_lie::traits::IsTranslationProductGroup;
use sophus_sensor::perspective_camera::PinholeCamera;
use std::collections::HashMap;

/// Camera calibration problem
#[derive(Clone)]
pub struct CamCalibProblem {
    /// intrinsics
    pub intrinsics: PinholeCamera<f64>,
    /// world from camera isometries
    pub world_from_cameras: Vec<Isometry3<f64>>,
    /// points in world
    pub points_in_world: Vec<VecF64<3>>,
    /// observations
    pub observations: Vec<ReprojTermSignature>,

    /// true intrinsics
    pub true_intrinsics: PinholeCamera<f64>,
    /// true world from camera isometries
    pub true_world_from_cameras: Vec<Isometry3<f64>>,
    /// true points in world
    pub true_points_in_world: Vec<VecF64<3>>,
}

impl CamCalibProblem {
    /// create new camera calibration problem
    pub fn new(spurious_matches: bool) -> Self {
        let true_world_from_cameras = vec![
            Isometry3::identity(),
            Isometry3::from_translation_and_factor(
                &VecF64::<3>::new(0.0, 1.0, 0.0),
                &Rotation3::identity(),
            ),
            Isometry3::from_translation_and_factor(
                &VecF64::<3>::new(0.0, 2.0, 0.0),
                &Rotation3::identity(),
            ),
        ];
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(0);

        let image_size = ImageSize {
            width: 640,
            height: 480,
        };

        let true_intrinsics = PinholeCamera::<f64>::from_params_and_size(
            &VecF64::<4>::new(600.0, 600.0, 320.0, 240.0),
            image_size,
        );

        let mut observations = vec![];
        let mut true_points_in_world = vec![];

        for i in 0..40 {
            let u = rng.gen::<f64>() * (image_size.width as f64 - 1.0);
            let v = rng.gen::<f64>() * (image_size.height as f64 - 1.0);
            let true_uv_in_img0 = VecF64::<2>::new(u, v);
            let img_noise = VecF64::<2>::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);
            observations.push(ReprojTermSignature {
                uv_in_image: true_uv_in_img0 + img_noise,
                entity_indices: [0, 0, i],
            });

            let true_point_in_cam0 = true_intrinsics.cam_unproj_with_z(&true_uv_in_img0, 10.0);
            true_points_in_world.push(true_world_from_cameras[0].transform(&true_point_in_cam0));
            let true_uv_in_img0_proof = true_intrinsics.cam_proj(&true_point_in_cam0);
            approx::assert_abs_diff_eq!(true_uv_in_img0, true_uv_in_img0_proof, epsilon = 0.1);

            let true_cam1_from_cam0 = true_world_from_cameras[1]
                .inverse()
                .group_mul(&true_world_from_cameras[0]);
            let true_point_in_cam1 = true_cam1_from_cam0.transform(&true_point_in_cam0);
            let true_uv_in_img1 = true_intrinsics.cam_proj(&true_point_in_cam1);
            let img_noise = VecF64::<2>::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);

            if spurious_matches && i == 0 {
                let u = rng.gen::<f64>() * (image_size.width as f64 - 1.0);
                let v = rng.gen::<f64>() * (image_size.height as f64 - 1.0);
                observations.push(ReprojTermSignature {
                    uv_in_image: VecF64::<2>::new(u, v),
                    entity_indices: [0, 1, i],
                });
            } else {
                observations.push(ReprojTermSignature {
                    uv_in_image: true_uv_in_img1 + img_noise,
                    entity_indices: [0, 1, i],
                });
            }

            let true_cam2_from_cam0 = true_world_from_cameras[2]
                .inverse()
                .group_mul(&true_world_from_cameras[0]);
            let true_point_in_cam2 = true_cam2_from_cam0.transform(&true_point_in_cam0);
            let true_uv_in_img2 = true_intrinsics.cam_proj(&true_point_in_cam2);
            let img_noise = VecF64::<2>::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5);
            observations.push(ReprojTermSignature {
                uv_in_image: true_uv_in_img2 + img_noise,
                entity_indices: [0, 2, i],
            });
        }

        Self {
            world_from_cameras: vec![
                Isometry3::<f64>::identity(),
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
    pub fn optimize_with_two_poses_fixed(&self, intrinsics_var_kind: VarKind) {
        let reproj_obs = CostSignature::<3, VecF64<2>, ReprojTermSignature> {
            family_names: ["cams".into(), "poses".into(), "points".into()],
            terms: self.observations.clone(),
        };

        let cam_family: VarFamily<PinholeCamera<f64>> =
            VarFamily::new(intrinsics_var_kind, vec![self.intrinsics]);

        let mut id = HashMap::new();
        id.insert(0, ());
        id.insert(1, ());

        let pose_family: VarFamily<Isometry3<f64>> = VarFamily::new_with_const_ids(
            VarKind::Free,
            self.world_from_cameras.clone(),
            id.clone(),
        );

        let point_family: VarFamily<VecF64<3>> =
            VarFamily::new(VarKind::Free, self.points_in_world.clone());

        let var_pool = VarPoolBuilder::new()
            .add_family("cams", cam_family)
            .add_family("poses", pose_family)
            .add_family("points", point_family)
            .build();

        let up_var_pool = optimize(
            var_pool,
            vec![
                // robust kernel to deal with outliers
                CostFn::new_robust(
                    reproj_obs.clone(),
                    ReprojectionCostFn {},
                    crate::robust_kernel::RobustKernel::Huber(HuberKernel::new(1.0)),
                ),
            ],
            OptParams {
                num_iter: 25,       // should converge in single iteration
                initial_lm_nu: 1.0, // if lm prior param is tiny
            },
        );

        let refined_world_from_robot = up_var_pool.get_members::<Isometry3<f64>>("poses".into());

        approx::assert_abs_diff_eq!(
            refined_world_from_robot[2].translation(),
            self.true_world_from_cameras[2].translation(),
            epsilon = 0.05
        );
    }

    /// optimize with priors
    pub fn optimize_with_priors(&self) {
        let priors =
            CostSignature::<1, (Isometry3<f64>, MatF64<6, 6>), Isometry3PriorTermSignature> {
                family_names: ["poses".into()],
                terms: vec![
                    Isometry3PriorTermSignature {
                        entity_indices: [0],
                        isometry_prior: (
                            self.true_world_from_cameras[0],
                            MatF64::<6, 6>::new_scaling(10000.0),
                        ),
                    },
                    Isometry3PriorTermSignature {
                        entity_indices: [1],
                        isometry_prior: (
                            self.true_world_from_cameras[1],
                            MatF64::<6, 6>::new_scaling(10000.0),
                        ),
                    },
                ],
            };

        let reproj_obs = CostSignature::<3, VecF64<2>, ReprojTermSignature> {
            family_names: ["cams".into(), "poses".into(), "points".into()],
            terms: self.observations.clone(),
        };

        let cam_family: VarFamily<PinholeCamera<f64>> =
            VarFamily::new(VarKind::Conditioned, vec![self.intrinsics]);

        let pose_family: VarFamily<Isometry3<f64>> =
            VarFamily::new(VarKind::Free, self.world_from_cameras.clone());

        let point_family: VarFamily<VecF64<3>> =
            VarFamily::new(VarKind::Free, self.points_in_world.clone());

        let var_pool = VarPoolBuilder::new()
            .add_family("cams", cam_family)
            .add_family("poses", pose_family)
            .add_family("points", point_family)
            .build();

        let up_var_pool = optimize(
            var_pool,
            vec![
                CostFn::new(priors.clone(), Isometry3PriorCostFn {}),
                CostFn::new(reproj_obs.clone(), ReprojectionCostFn {}),
            ],
            OptParams {
                num_iter: 5,
                initial_lm_nu: 1.0,
            },
        );

        let refined_world_from_robot = up_var_pool.get_members::<Isometry3<f64>>("poses".into());

        println!(
            "refined_world_from_robot[0].translation(): {:?}",
            refined_world_from_robot[0].translation()
        );
        println!(
            "refined_world_from_robot[1].translation(): {:?}",
            refined_world_from_robot[1].translation()
        );

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
        use crate::variables::VarKind;

        CamCalibProblem::new(true).optimize_with_two_poses_fixed(VarKind::Free);
        CamCalibProblem::new(false).optimize_with_two_poses_fixed(VarKind::Conditioned);
        CamCalibProblem::new(false).optimize_with_priors();
    }
}
