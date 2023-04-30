use sophus_rs::image::layout::ImageSize;
use sophus_rs::opt::cost_args::CostArgTuple;
use sophus_rs::sensor::perspective_camera::PinholeCamera;
use sophus_rs::{
    lie::rotation3::{Isometry3, Rotation3},
    opt::{
        cost_args::{CostFnArg, CostTermArg},
        nlls::{apply, CostTerm, CostTermSignature, ResidualFn},
    },
};
type V<const N: usize> = nalgebra::SVector<f64, N>;
use tuple_list::{tuple_list, tuple_list_type, Tuple};

#[derive(Copy, Clone)]
struct StereoBaCostTermSignature {
    c: nalgebra::Vector2<f64>,
    entity_indices: tuple_list_type!(usize, usize),
}

impl CostTermSignature<2> for StereoBaCostTermSignature {
    type EntityIndexTuple = tuple_list_type!(usize, usize);
    type Constants = nalgebra::Vector2<f64>;

    fn c_ref(&self) -> &Self::Constants {
        &self.c
    }

    fn idx_ref(&self) -> &Self::EntityIndexTuple {
        &self.entity_indices
    }

    const DOF_TUPLE: [i64; 2] = [3, 6];
}

#[derive(Copy, Clone)]
struct StereoBa<const E1: char, const E2: char> {
    pinhole_camera: PinholeCamera,
}

impl<const E1: char, const E2: char> ResidualFn<9, 2> for StereoBa<E1, E2> {
    type Args = tuple_list_type!(
        CostTermArg<Isometry3,E1>,
        CostTermArg<nalgebra::Vector3::<f64>,E2>
    );

    type Constants = nalgebra::Vector2<f64>;

    fn cost(&self, args: &Self::Args, obs: &Self::Constants) -> CostTerm<9, 2> {
        let mut cost = CostTerm::new(Self::Args::get_dims());

        let world_from_camera_pose = args.0.arg;
        let point_in_world = args.1 .0.arg;

        let point_in_camera = world_from_camera_pose.inverse().transform(&point_in_world);

        let point_in_image = self.pinhole_camera.cam_proj(&point_in_camera);

        // TODO: fill out Jacobian correctly
        if E1 == 'v' {
            cost.hessian.block(0);
        }
        if E2 == 'v' {
            cost.hessian.block(1);
        }

        let residual = point_in_image - obs;
        cost.cost = residual.norm_squared();
        cost
    }
}

fn main() {
    let world_from_camera_pose_family = vec![
        Isometry3::identity(),
        Isometry3::from_t_and_subgroup(
            &nalgebra::Vector3::new(0.0, 0.0, 1.0),
            &Rotation3::identity(),
        ),
    ];

    let point_family = vec![
        nalgebra::Vector3::repeat(0.0),
        nalgebra::Vector3::repeat(1.0),
    ];

    let tracking_families = (
        CostFnArg::var(&world_from_camera_pose_family),
        CostFnArg::cond(&point_family),
    );
    let ba_families = (
        CostFnArg::var(&world_from_camera_pose_family),
        CostFnArg::var(&point_family),
    );

    let cost_terms = vec![StereoBaCostTermSignature {
        c: nalgebra::Vector2::repeat(1.0),
        entity_indices: tuple_list!(1, 0),
    }];

    let pinhole_camera = PinholeCamera::from_params_and_size(
        &V::<4>::new(500.0, 500.0, 320.0, 240.0),
        ImageSize::new(640, 480),
    );

    apply(
        StereoBa { pinhole_camera },
        &cost_terms,
        &tracking_families.into_tuple_list(),
    );
    apply(
        StereoBa { pinhole_camera },
        &cost_terms,
        &ba_families.into_tuple_list(),
    );
}
