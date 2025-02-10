use sophus_autodiff::{
    dual::{
        DualScalar,
        DualVector,
    },
    linalg::{
        MatF64,
        VecF64,
        EPS_F64,
    },
};
use sophus_geo::region::{box_region::NonEmptyBoxRegion, interval::NonEmptyInterval, IsRegionBase};
use sophus_lie::{
    prelude::*,
    Isometry2,
    Isometry2F64,
};

use crate::{
    nlls::{
        constraint::{
            evaluated_ineq_constraint::{
                EvaluatedIneqConstraint,
                MakeEvaluatedIneqConstraint,
            },
            ineq_constraint::{
                IneqConstraints,
                IsIneqConstraint,
            },
            ineq_constraint_fn::IneqConstraintFn,
        },
        cost::{
            cost_fn::CostFn,
            cost_term::CostTerms,
        },
        functor_library::{
            costs::{
                pose_graph::PoseGraphCostTerm,
                quadratic1::Quadratic1CostTerm,
                quadratic2::Quadratic2CostTerm,
            },
            ineq_constraints::{
                linear_ineq::LinearIneqConstraint,
                nonlinear_ineq::NonLinearIneqConstraint,
            },
        },
        optimize_nlls,
        experimental_sqp::experimental_sqp,
        OptParams,
    },
    variables::{
        var_builder::VarBuilder,
        var_family::VarFamily,
        VarKind,
    },
};

extern crate alloc;

/// Simple linear inequality constraint problem
pub struct LinearIneqToyProblem {}

impl Default for LinearIneqToyProblem {
    fn default() -> Self {
        Self::new()
    }
}

/// linear equality constraint
#[derive(Clone, Debug)]
pub struct AboveConstraint {
    /// lower/upper bound
    pub bounds: NonEmptyBoxRegion<1>,
    /// entity index
    pub entity_indices: [usize; 1],
}

// 0 < x2 - ln(x1) < infinity
// 0 < x1 < infinity

impl AboveConstraint {
    /// Compute the residual
    pub fn constraint<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        pose: Isometry2<Scalar, 1, DM, DN>,
    ) -> Scalar::Vector<1> {
        Scalar::Vector::<1>::from_array([pose.translation().elem(1)])
    }
}

impl IsIneqConstraint<1, 3, 1, (), Isometry2F64> for AboveConstraint {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn bounds(&self) -> NonEmptyBoxRegion<1> {
        self.bounds
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        args: Isometry2F64,
        derivatives: [VarKind; 1],
    ) -> EvaluatedIneqConstraint<1, 3, 1> {
        let constraint_value: VecF64<1> = Self::constraint::<f64, 0, 0>(args);

        (|| {
            Self::constraint::<DualScalar<3, 1>, 3, 1>(
                Isometry2::exp(DualVector::var(VecF64::<3>::zeros())) * args.to_dual_c(),
            )
            .jacobian()
        },)
            .make_ineq(idx, derivatives, constraint_value, self.bounds)
    }
}

impl LinearIneqToyProblem {
    fn new() -> Self {
        Self {}
    }

    /// Test the linear equality constraint problem
    pub fn test(&self) {
        // use sophus_autodiff::linalg::EPS_F64;

        // let true_world_from_robot0 = Isometry2F64::trans_x(0.0);
        // let true_world_from_robot1 = Isometry2F64::trans_x(1.0);
        // let true_world_from_robot2 = Isometry2F64::trans_x(2.0);

        // let est_world_from_robot0 = true_world_from_robot0.clone();
        // let est_world_from_robot1 = true_world_from_robot0.clone();
        // let est_world_from_robot2 = true_world_from_robot2.clone();
        // const POSE: &str = "poses";

        // let obs_pose_a_from_pose_b_poses = CostTerms::new(
        //     [POSE, POSE],
        //     alloc::vec![
        //         PoseGraphCostTerm {
        //             pose_a_from_pose_b: true_world_from_robot0.inverse() *
        // true_world_from_robot1,             entity_indices: [0, 1],
        //         },
        //         PoseGraphCostTerm {
        //             pose_a_from_pose_b: true_world_from_robot0.inverse() *
        // true_world_from_robot1,             entity_indices: [1, 2],
        //         }
        //     ],
        // );
        // let mut constants = alloc::collections::BTreeMap::new();
        // constants.insert(0, ());
        // constants.insert(2, ());

        // let variables = VarBuilder::new()
        //     .add_family(
        //         POSE,
        //         VarFamily::new_with_const_ids(
        //             VarKind::Free,
        //             alloc::vec![
        //                 est_world_from_robot0,
        //                 est_world_from_robot1,
        //                 est_world_from_robot2,
        //             ],
        //             constants,
        //         ),
        //     )
        //     .build();
        //   let ineq_constraints = IneqConstraints::new(
        //     [POSE],
        //     vec![
        //         AboveConstraint{
        //             lower: VecF64::<1>::from_array([0.5]),
        //             upper: VecF64::<1>::from_array([f64::INFINITY]),
        //             entity_indices: [0],
        //         }
        //     ]
        // );
        // let solution = optimize_sqp_with_constraints(
        //     variables,
        //     alloc::vec![CostFn::new_box((), obs_pose_a_from_pose_b_poses.clone(),)],
        //     alloc::vec![IneqConstraintFn::new_box((), ineq_constraints)],
        //     OptParams {
        //         num_iterations: 10,
        //         initial_lm_damping: 0.1,
        //         parallelize: true,
        //         ..Default::default()
        //     },
        // )
        // .unwrap();

        // let refined_world_from_robot = solution.variables.get_members::<Isometry2F64>(POSE);

        // for i in 0..refined_world_from_robot.len() {
        //     println!("{}", refined_world_from_robot[i].matrix());
        // }

        // println!("{:?}", refined_world_from_robot);

        const VAR_X: &str = "x";

        let initial_x = VecF64::<2>::new(0.1, 0.1);

        let cost_terms = CostTerms::new(
            [VAR_X],
            alloc::vec![Quadratic2CostTerm {
                z: VecF64::<2>::new(2.0, 4.0),
                entity_indices: [0]
            },],
        );
        let ineq_constraints = IneqConstraints::new(
            [VAR_X],
            // vec![LinearIneqConstraint {
            //     mat_a: MatF64::<3, 2>::from_array2([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
            //     entity_indices: [0],
            //     lower: VecF64::<3>::from_array([1.0, 0.0, 0.0]),
            //     upper: VecF64::<3>::from_array([1.0, 0.7, 0.7]),
            // }],
            vec![NonLinearIneqConstraint {
                bounds: NonEmptyInterval::from_bounds(0.0, f64::INFINITY).to_box_region(),
                entity_indices: [0],
            }],
        );

        let solution = experimental_sqp(
            VarBuilder::new()
                .add_family(VAR_X, VarFamily::new(VarKind::Free, alloc::vec![initial_x]))
                .build(),
            alloc::vec![CostFn::new_box((), cost_terms.clone(),)],
            alloc::vec![IneqConstraintFn::new_box((), ineq_constraints,)],
            OptParams {
                num_iterations: 10,
                initial_lm_damping: EPS_F64,
                parallelize: true,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("{}", e));

        // let refined_variables = solution.variables;
        // let refined_x = refined_variables.get_members::<VecF64<1>>(VAR_X);

        // let x0 = refined_x[0];
        // let x1 = refined_x[1];
        // approx::assert_abs_diff_eq!(x0[0], 0.5, epsilon = 1e-6);
        // approx::assert_abs_diff_eq!(x1[0], 1.5, epsilon = 1e-6);

        // // converged solution should satisfy the equality constraint
        // approx::assert_abs_diff_eq!(
        //     LinearEqConstraint1::residual(x0, x1, EQ_CONSTRAINT_RHS)[0],
        //     0.0,
        //     epsilon = 1e-6
        // );
    }
}

#[test]
fn normalize_opt_tests() {
    LinearIneqToyProblem::new().test();
}
