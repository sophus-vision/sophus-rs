use dfdx::prelude::*;
use tuple_list::*;

use crate::{
    calculus::{batch_types::*, vector_valued_maps::*},
    lie::isometry2::*,
    opt::{cost_args::*, nlls::*},
};

#[derive(Clone)]
struct SimplePriorCostTermSignature {
    c: Isometry2<1>,
    entity_indices: [usize; 1],
}

impl CostTermSignature<1> for SimplePriorCostTermSignature {
    type Constants = Isometry2<1>;

    fn c_ref(&self) -> &Self::Constants {
        &self.c
    }

    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    const DOF_TUPLE: [i64; 1] = [6];
}

#[derive(Copy, Clone)]
struct SimplePrior<const E1: char> {}

impl<const E1: char> ResidualFn<6, 1> for SimplePrior<E1> {
    type Args = tuple_list_type!(CostTermArg<Isometry2<1>, E1>,);

    type Constants = Isometry2<1>;

    fn cost(&self, args: &Self::Args, obs: &Self::Constants) -> CostTerm<6, 1> {
        let mut cost = CostTerm::new(Self::Args::get_dims());

        let world_from_robot_pose: Isometry2<1> = args.0.arg.clone();


        fn res_fn<GenTape: SophusTape, GenTape2: SophusTape>(
            world_from_robot_pose: GenTapedIsometry2<1, GenTape>,
            obs: GenTapedIsometry2<1, GenTape2>,
        ) -> GenV<1, 3, GenTape>
        where
            GenTape: SophusTape + Merge<GenTape2>,
            GenTape2: SophusTape,
        {
            GenTapedIsometry2::<1, GenTape>::mul(world_from_robot_pose, obs.into_inverse())
                .into_log()
        }

        let residual = res_fn(world_from_robot_pose.clone(), obs.clone());
        let dx_res_fn = |x: TapedV<1, 3>| -> TapedV<1, 3> {
            let pp: GenTapedIsometry2<1, SharedTape> = GenTapedIsometry2::<1, SharedTape>::mul(
                GenTapedIsometry2::<1, SharedTape>::exp(x),
                world_from_robot_pose.retaped(),
            );
            res_fn(pp, obs.clone())
        };

        let residual: S<3> = residual.reshape();

        let zeros: V<1, 3> = dfdx::tensor::Cpu::default().zeros();
        let dx_res =
            VectorValuedMapFromVector::sym_diff_quotient_from_taped(dx_res_fn, zeros, 1e-5);
        let dx_res: V<3, 3> = dx_res.reshape();

        if E1 == 'v' || E1 == 'm' {
            let gradient = dx_res.clone().matmul(residual);
            let hessian = dx_res.clone().permute::<_, Axes2<1, 0>>().matmul(dx_res);
            cost.gradient.set_block(0, gradient.clone().realize());
            cost.hessian.set_block(0, 0, hessian.clone().realize());

            println!("gradient: {:?}", gradient.array());
            println!("hessian: {:?}", hessian.array());
        }

        cost
    }
}

pub struct SimplePriorProblem {
    true_world_from_robot: Isometry2<1>,
    est_world_from_robot: Isometry2<1>,
}

impl SimplePriorProblem {
    pub fn new() -> Self {
        let dev: Cpu = Default::default();
        let p = dev.tensor(&[1.0, 0.0, 1.0, 0.0]);
        Self {
            true_world_from_robot: Isometry2::<1>::from_params(p.reshape()),
            est_world_from_robot: Isometry2::<1>::identity(),
        }
    }

    pub fn test(&self) {
        let cost_signature = vec![SimplePriorCostTermSignature {
            c: self.true_world_from_robot.clone(),
            entity_indices: [0],
        }];

        let family = vec![self.est_world_from_robot.clone()];

        let families = CostFnArg::var(family);

        let families = OneFamilyProblem::optimize(SimplePrior {}, cost_signature, families);
    }
}
