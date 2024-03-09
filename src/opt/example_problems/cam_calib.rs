use std::collections::BTreeMap;
use std::collections::HashMap;

use nalgebra::ComplexField;
use nalgebra::Rotation2;

use crate::calculus::dual::dual_scalar::Dual;
use crate::calculus::dual::dual_vector::DualV;
use crate::calculus::maps::vector_valued_maps::VectorValuedMapFromVector;
use crate::calculus::types::scalar::IsScalar;
use crate::calculus::types::vector::IsVector;
use crate::calculus::types::V;
use crate::image::view::ImageSize;
use crate::lie::rotation2::Isometry2;
use crate::lie::rotation3::Isometry3;
use crate::lie::rotation3::Rotation3;
use crate::lie::traits::IsTranslationProductGroup;
use crate::opt::nlls::optimize_one_cost;
use crate::opt::nlls::Cost;
use crate::opt::nlls::CostSignature;
use crate::opt::nlls::EvaluatedTerm;
use crate::opt::nlls::IsResidualFn;
use crate::opt::nlls::IsTermSignature;
use crate::opt::nlls::IsVarFamily;
use crate::opt::nlls::IsVariable;
use crate::opt::nlls::OptParams;
use crate::opt::nlls::VarFamily;
use crate::opt::nlls::VarKind;
use crate::opt::nlls::VarPool;
use crate::opt::nlls::VarPoolBuilder;
use crate::sensor::perspective_camera::PinholeCamera;
use crate::tensor::element::SVec;

impl IsVariable for PinholeCamera<f64> {
    const DOF: usize = 4;

    fn update(&mut self, delta: nalgebra::DVectorView<f64>) {
        let new_params = self.params().clone() + delta;
        self.set_params(&new_params);
    }

    type Arg = Isometry2<f64>;
}

fn res_fn<Scalar: IsScalar>(
    intrinscs: PinholeCamera<Scalar>,
    world_from_camera: Isometry3<Scalar>,
    point_in_world: Scalar::Vector<3>,
    obs: Scalar::Vector<2>,
) -> Scalar::Vector<2> {
    let point_in_cam = world_from_camera.inverse().transform(&point_in_world);
    obs - intrinscs.cam_proj(&point_in_cam)
}

#[derive(Clone)]
struct ReprojTermSignature {
    obs: V<2>,
    entity_indices: [usize; 3],
}

impl IsTermSignature<3> for ReprojTermSignature {
    type Constants = V<2>;

    fn c_ref(&self) -> &Self::Constants {
        &self.obs
    }

    fn idx_ref(&self) -> &[usize; 3] {
        &self.entity_indices
    }

    const DOF_TUPLE: [i64; 3] = [4, 6, 3];
}

#[derive(Copy, Clone)]
struct CamFn {}

impl IsResidualFn<13, 3, (PinholeCamera<f64>, Isometry3<f64>, V<3>), V<2>> for CamFn {
    fn eval(
        &self,
        (intrinsics, world_from_camera, point_in_world): (PinholeCamera<f64>, Isometry3<f64>, V<3>),
        derivatives: [VarKind; 3],
        obs: &V<2>,
    ) -> EvaluatedTerm<13, 3> {
        // let _ = point_in_world;

        //println!("{:?} {} {} {}", intrinsics, world_from_camera, point_in_world, obs);

        let residual = res_fn(intrinsics, world_from_camera, point_in_world, obs.clone());

        // println!("r: {:?}", residual);

        let d0_res_fn = |x: DualV<4>| -> DualV<2> {
            res_fn(
                PinholeCamera::<Dual>::from_params_and_size(&x, intrinsics.image_size()),
                world_from_camera.to_dual_c(),
                DualV::c(point_in_world),
                DualV::c(obs.clone()),
            )
        };
        let d1_res_fn = |x: DualV<6>| -> DualV<2> {
            res_fn(
                PinholeCamera::<Dual>::from_params_and_size(
                    &DualV::c(intrinsics.params().clone()),
                    intrinsics.image_size(),
                ),
                Isometry3::<Dual>::exp(&x).group_mul(&world_from_camera.to_dual_c()),
                DualV::c(point_in_world),
                DualV::c(obs.clone()),
            )
        };
        let d2_res_fn = |x: DualV<3>| -> DualV<2> {
            res_fn(
                PinholeCamera::<Dual>::from_params_and_size(
                    &DualV::c(intrinsics.params().clone()),
                    intrinsics.image_size(),
                ),
                world_from_camera.to_dual_c(),
                x,
                DualV::c(obs.clone()),
            )
        };

        let mut maybe_dx0 = None;
        let mut maybe_dx1 = None;
        let mut maybe_dx2 = None;

        if derivatives[0] != VarKind::Conditioned {
            maybe_dx0 = Some(VectorValuedMapFromVector::static_fw_autodiff(
                d0_res_fn,
                intrinsics.params().clone(),
            ));
        }
        if derivatives[1] != VarKind::Conditioned {
            let zeros: V<6> = V::<6>::zeros();
            maybe_dx1 = Some(VectorValuedMapFromVector::static_fw_autodiff(
                d1_res_fn, zeros,
            ));
        }
        if derivatives[2] != VarKind::Conditioned {
            maybe_dx2 = Some(VectorValuedMapFromVector::static_fw_autodiff(
                d2_res_fn,
                point_in_world,
            ));
        }

        EvaluatedTerm::new3(maybe_dx0, maybe_dx1, maybe_dx2, residual)
    }
}

#[derive(Clone)]
struct CamCalibProblem {
    intrinsics: PinholeCamera<f64>,
    world_from_cameras: Vec<Isometry3<f64>>,
    points_in_world: Vec<V<3>>,
    obs: Vec<ReprojTermSignature>,

    true_world_from_cameras: Vec<Isometry3<f64>>,
}

impl CamCalibProblem {
    pub fn new() -> Self {
        let true_world_from_cameras = vec![
            Isometry3::identity(),
            Isometry3::from_translation_and_factor(
                &V::<3>::new(0.0, 1.0, 0.0),
                &Rotation3::identity(),
            ),
            Isometry3::from_translation_and_factor(
                &V::<3>::new(0.0, 2.0, 0.0),
                &Rotation3::identity(),
            ),
        ];
        use rand::prelude::*;
        let mut rng = rand::thread_rng();

        let image_size = ImageSize {
            width: 640,
            height: 480,
        };

        let intrinsics = PinholeCamera::<f64>::from_params_and_size(
            &V::<4>::new(600.0, 600.0, 320.0, 240.0),
            image_size,
        );

        let mut obs = vec![];
        let mut points_in_world = vec![];

        for i in 0..40 {
            let u = rng.gen::<f64>() * (image_size.width as f64 - 1.0);
            let v = rng.gen::<f64>() * (image_size.height as f64 - 1.0);
            let depth = rng.gen::<f64>() * 2.0 + 1.0;

            obs.push(ReprojTermSignature {
                obs: V::<2>::new(u, v)
                    + V::<2>::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5),
                entity_indices: [0, 0, i],
            });
            println!("uv {} {} {}", u, v, depth);

            let point_in_left = intrinsics.cam_unproj_with_z(&V::<2>::new(u, v), 10.0);
            //  let point_in_left2 = intrinsics.cam_unproj_with_z(&V::<2>::new(u, v), depth);
            // if i ==0 {
            //     point_in_left = point_in_left2
            // }

            let uv2 = intrinsics.cam_proj(&point_in_left);

            println!("uv2 {}  {}", uv2, point_in_left[2]);

            points_in_world.push(true_world_from_cameras[0].transform(&point_in_left));

            let right_from_left = true_world_from_cameras[1]
                .inverse()
                .group_mul(&true_world_from_cameras[0]);
            let point_in_right = right_from_left.transform(&point_in_left);

            let uv_right = intrinsics.cam_proj(&point_in_right);

            obs.push(ReprojTermSignature {
                obs: uv_right,
                entity_indices: [0, 1, i],
            });

            let rright_from_left = true_world_from_cameras[2]
                .inverse()
                .group_mul(&true_world_from_cameras[0]);
            let point_in_rright = rright_from_left.transform(&point_in_left);

            let uv_rright = intrinsics.cam_proj(&point_in_rright);

            obs.push(ReprojTermSignature {
                obs: uv_rright + V::<2>::new(rng.gen::<f64>() - 0.5, rng.gen::<f64>() - 0.5),
                entity_indices: [0, 2, i],
            });
        }

        Self {
            intrinsics,
            world_from_cameras: vec![
                Isometry3::<f64>::identity(),
                true_world_from_cameras[1],
                true_world_from_cameras[1],
            ],
            points_in_world,
            obs,
            true_world_from_cameras,
        }
    }

    pub fn test(&self) {
        let obs_pose_a_from_pose_b_poses = CostSignature::<3, V<2>, ReprojTermSignature> {
            family_names: ["cams".into(), "poses".into(), "points".into()],
            terms: self.obs.clone(),
        };

        let cam_family: VarFamily<PinholeCamera<f64>> =
            VarFamily::new(VarKind::Conditioned, vec![self.intrinsics], HashMap::new());

        let mut id = HashMap::new();
        id.insert(0, ());
        id.insert(1, ());

        let pose_family: VarFamily<Isometry3<f64>> =
            VarFamily::new(VarKind::Free, self.world_from_cameras.clone(), id.clone());

        // let mut point_id = HashMap::new();
        // point_id.insert(0, ());
        //   point_id.insert(1, ());
        // point_id.insert(2, ());
        // point_id.insert(3, ());
        let point_family: VarFamily<V<3>> = VarFamily::new(
            VarKind::Marginalized,
            self.points_in_world.clone(),
            HashMap::new(),
        );

        // let cam_box: Box<dyn IsVarFamily> = Box::new(cam_family);
        // let pose_box: Box<dyn IsVarFamily> = Box::new(pose_family);
        // let point_box: Box<dyn IsVarFamily> = Box::new(point_family);

        // let mut map = BTreeMap::new();
        // map.insert("cams".into(), cam_box);
        // map.insert("poses".into(), pose_box);
        // map.insert("points".into(), point_box);

        let var_pool = VarPoolBuilder::new()
            .add_family("cams", cam_family)
            .add_family("poses", pose_family)
            .add_family("points", point_family)
            .build();

        let up_var_pool = optimize_one_cost(
            var_pool,
            Cost::new(obs_pose_a_from_pose_b_poses.clone(), CamFn {}),
            OptParams {
                num_iter: 5,        // should converge in single iteration
                initial_lm_nu: 1.0, // if lm prior param is tiny
            },
        );

        let refined_world_from_robot = up_var_pool.get_members::<Isometry3<f64>>("poses".into());
        let points = up_var_pool.get_members::<V<3>>("points".into());

        println!("poses[0] {}", refined_world_from_robot[0].compact());
        println!("poses[1] {}", refined_world_from_robot[1].compact());
        println!("poses[2] {}", refined_world_from_robot[2].compact());

        println!("points {:?}", points);

        assert!(false);
    }
}

mod tests {

    #[test]
    fn simple_cam_tests() {
        use super::CamCalibProblem;

        CamCalibProblem::new().test();
    }
}
