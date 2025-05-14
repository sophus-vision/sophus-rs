use std::sync::Arc;

use sophus_autodiff::linalg::VecF64;
use sophus_opt::{
    nlls::{
        CostFn,
        CostTerms,
        OptParams,
        optimize_nlls,
    },
    variables::{
        VarBuilder,
        VarFamily,
        VarKind,
    },
};

use crate::examples::optics_sim::{
    convex_lens::BiConvexLens2F64,
    cost::ChiefRayCost,
};

/// Aperture stop in an optical system.
pub mod aperture_stop;
/// BiConvex lens in an optical system.
pub mod convex_lens;
/// Cost terms for the optimization process.
pub mod cost;
/// Detector in an optical system.
pub mod detector;
/// Elements in an optical system.
pub mod element;
/// The light path of rays through the optical system.
pub mod light_path;
/// Scene points in the optical system.
pub mod scene_point;

/// Refines the angle of the chief ray using a nonlinear least squares optimization.
pub fn refine_chief_ray_angle(
    angle: f64,
    scene_point: VecF64<2>,
    lens: Arc<BiConvexLens2F64>,
    target_point: VecF64<2>,
) -> f64 {
    let family: VarFamily<VecF64<1>> = VarFamily::new(VarKind::Free, vec![VecF64::<1>::new(angle)]);
    let variables = VarBuilder::new().add_family("angle", family).build();
    let mut chief_ray_cost = CostTerms::new(["angle"], vec![]);
    chief_ray_cost.collection.push(ChiefRayCost {
        scene_point,
        aperture: target_point,
        entity_indices: [0],
    });
    let solution = optimize_nlls(
        variables,
        vec![CostFn::new_boxed(lens, chief_ray_cost)],
        OptParams {
            num_iterations: 100,
            initial_lm_damping: 1.0,
            parallelize: true,
            solver: Default::default(),
        },
    )
    .unwrap();
    solution.variables.get_members::<VecF64<1>>("angle")[0][0]
}
