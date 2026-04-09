//! Bundle adjustment with a scale constraint.
//!
//! Variant of [`crate::example_problems::ba_problem::BaProblem`] where only pose 0 is fixed
//! (resolving rotation/translation gauge), while the scale gauge is resolved via a
//! `TranslationNormConstraint` on pose 1.
//!
//! This exercises the Schur complement path with equality constraints:
//! - Free variables: shared EUCM camera intrinsic + poses 1..N
//! - Marginalized: 3-D points (Schur path only)
//! - Equality constraint: `|t(pose_1)| = r` (fixes scale)

use sophus_autodiff::{
    dual::{
        DualScalar,
        DualVector,
    },
    linalg::VecF64,
};
use sophus_lie::{
    IsAffineGroup,
    Isometry3,
    Isometry3F64,
};
use sophus_solver::LinearSolverEnum;

use crate::{
    example_problems::ba_problem::BaProblem,
    nlls::{
        EqConstraintFn,
        EqConstraints,
        EvaluatedEqConstraint,
        OptParams,
        optimize_nlls_with_eq_constraints,
    },
    prelude::*,
    variables::{
        VarFamilies,
        VarKind,
    },
};

extern crate alloc;

/// Equality constraint on the translation norm of an SE(3) pose.
///
/// `|t(T)| = r`
///
/// where `T ∈ SE(3)` and `r` is the target translation norm.  The corresponding
/// constraint residual is:
///
/// `c(T) = |t(T)| − r`.
///
/// This is useful for fixing the scale in bundle adjustment: pose 0 is held fixed
/// (gauge freedom for rotation/translation), and a `TranslationNormConstraint` on
/// pose 1 fixes the scale degree of freedom.
#[derive(Clone, Debug)]
pub struct TranslationNormConstraint {
    /// Target translation norm `r`.
    pub radius: f64,
    /// Entity index for `T`.
    pub entity_indices: [usize; 1],
}

impl TranslationNormConstraint {
    /// Compute the constraint residual `c(T) = |t(T)| − r`.
    pub fn residual<Scalar: IsSingleScalar<DM, DN>, const DM: usize, const DN: usize>(
        pose: Isometry3<Scalar, 1, DM, DN>,
        radius: Scalar,
    ) -> Scalar::Vector<1> {
        let t = pose.translation();
        let norm = t.norm();
        Scalar::Vector::<1>::from_array([norm - radius])
    }
}

impl HasEqConstraintResidualFn<1, 6, 1, (), Isometry3F64> for TranslationNormConstraint {
    fn idx_ref(&self) -> &[usize; 1] {
        &self.entity_indices
    }

    fn eval(
        &self,
        _global_constants: &(),
        idx: [usize; 1],
        pose: Isometry3F64,
        var_kinds: [VarKind; 1],
    ) -> EvaluatedEqConstraint<1, 6, 1> {
        let residual = Self::residual(pose, self.radius);

        // Jacobian via left-perturbation: T(x) = exp(x) * T, differentiated at x = 0.
        let dx_res_fn = |x: DualVector<f64, 6, 6, 1>| -> DualVector<f64, 1, 6, 1> {
            let radius_dual = DualScalar::from_f64(self.radius);
            Self::residual::<DualScalar<f64, 6, 1>, 6, 1>(
                Isometry3::exp(x) * pose.to_dual_c(),
                radius_dual,
            )
        };

        (|| dx_res_fn(DualVector::var(VecF64::<6>::zeros())).jacobian(),)
            .make_eq(idx, var_kinds, residual)
    }
}

/// Bundle adjustment with a scale constraint on pose 1's translation norm.
pub struct BaScaleConstraintProblem {
    /// The underlying BA problem geometry and observations.
    pub ba: BaProblem,
    /// Target translation norm for pose 1 (ground-truth value).
    pub target_radius: f64,
}

impl BaScaleConstraintProblem {
    /// Build a new scale-constrained BA problem.
    ///
    /// `num_cameras` cameras are placed on a ring; only pose 0 is fixed (gauge for
    /// rotation/translation). The scale gauge is fixed by constraining `|t(pose_1)| = r`
    /// where `r` is the ground-truth translation norm of pose 1.
    pub fn new(num_cameras: usize, num_points: usize) -> Self {
        let ba = BaProblem::new(num_cameras, num_points);
        let target_radius = ba.true_world_from_cameras[1].translation().norm();
        Self { ba, target_radius }
    }

    /// Build variable families for the given point kind.
    pub fn build_variables(&self, points_kind: VarKind) -> VarFamilies {
        use std::collections::BTreeMap;

        use crate::variables::{
            VarBuilder,
            VarFamily,
        };

        // Only pose 0 is fixed; pose 1 is free but constrained via TranslationNormConstraint.
        let mut fixed = BTreeMap::new();
        fixed.insert(0, ());

        VarBuilder::new()
            .add_family(
                BaProblem::CAMS,
                VarFamily::new(VarKind::Free, self.ba.init_intrinsics.clone()),
            )
            .add_family(
                BaProblem::POSES,
                VarFamily::new_with_const_ids(
                    VarKind::Free,
                    self.ba.world_from_cameras.clone(),
                    fixed,
                ),
            )
            .add_family(
                BaProblem::POINTS,
                VarFamily::new(points_kind, self.ba.points_in_world.clone()),
            )
            .build()
    }

    /// Build the equality constraint function (scale constraint on pose 1).
    pub fn build_eq_constraint_fn(&self) -> alloc::boxed::Box<dyn crate::nlls::IsEqConstraintsFn> {
        EqConstraintFn::new_boxed(
            (),
            EqConstraints::new(
                [BaProblem::POSES],
                alloc::vec![TranslationNormConstraint {
                    radius: self.target_radius,
                    entity_indices: [1],
                }],
            ),
        )
    }

    fn run(
        &self,
        points_kind: VarKind,
        solver: LinearSolverEnum,
        parallelize: bool,
    ) -> VarFamilies {
        optimize_nlls_with_eq_constraints(
            self.build_variables(points_kind),
            alloc::vec![self.ba.build_cost()],
            alloc::vec![self.build_eq_constraint_fn()],
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

    /// Optimize with all variables free (no Schur complement).
    pub fn optimize_all_free(&self, solver: LinearSolverEnum, parallelize: bool) -> VarFamilies {
        self.run(VarKind::Free, solver, parallelize)
    }

    /// Optimize with points marginalized (Schur complement path with scale constraint).
    pub fn optimize_marg_points(&self, solver: LinearSolverEnum, parallelize: bool) -> VarFamilies {
        self.run(VarKind::Marginalized, solver, parallelize)
    }

    /// Ground-truth errors: (pose translation RMS, point position RMS).
    pub fn gt_errors(&self, vars: &VarFamilies) -> (f64, f64) {
        self.ba.gt_errors(vars)
    }

    /// Check that the scale constraint is satisfied: `|t(pose_1)| ≈ target_radius`.
    pub fn constraint_satisfied(&self, vars: &VarFamilies) -> bool {
        let poses = vars.get_members::<Isometry3F64>(BaProblem::POSES);
        let t_norm = poses[1].translation().norm();
        (t_norm - self.target_radius).abs() < 1e-4
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use sophus_solver::{
        LinearSolverEnum,
        ldlt::{
            BlockSparseLdlt,
            FaerSparseLdlt,
        },
        lu::FaerSparseLu,
    };

    use super::BaScaleConstraintProblem;

    // Looser tolerances than fixed-gauge BA: pose 1 is only scale-constrained (not fully fixed),
    // so more degrees of freedom remain and convergence is harder.
    const POSE_TOL: f64 = 0.1;
    const POINT_TOL: f64 = 0.1;

    /// Non-Schur (standard) solvers must converge and satisfy the scale constraint.
    #[test]
    fn standard_solvers_converge() {
        let prob = BaScaleConstraintProblem::new(4, 20);

        let solver = LinearSolverEnum::FaerSparseLu(FaerSparseLu {});
        let vars = prob.optimize_all_free(solver, false);
        let (pose_rms, pt_rms) = prob.gt_errors(&vars);
        assert!(
            pose_rms < POSE_TOL,
            "solver={solver:?} pose_rms={pose_rms:.4} >= {POSE_TOL}",
        );
        assert!(
            pt_rms < POINT_TOL,
            "solver={solver:?} pt_rms={pt_rms:.4} >= {POINT_TOL}",
        );
        assert!(
            prob.constraint_satisfied(&vars),
            "solver={solver:?} scale constraint not satisfied",
        );
    }

    /// Schur solvers must converge and satisfy the scale constraint with points marginalized.
    #[test]
    fn schur_solvers_converge() {
        let prob = BaScaleConstraintProblem::new(4, 20);

        for solver in [
            LinearSolverEnum::SchurBlockSparseLdlt(BlockSparseLdlt::default()),
            LinearSolverEnum::SchurFaerSparseLdlt(FaerSparseLdlt::default()),
        ] {
            let vars = prob.optimize_marg_points(solver, false);
            let (pose_rms, pt_rms) = prob.gt_errors(&vars);
            assert!(
                pose_rms < POSE_TOL,
                "solver={:?} pose_rms={pose_rms:.4} >= {POSE_TOL}",
                solver
            );
            assert!(
                pt_rms < POINT_TOL,
                "solver={:?} pt_rms={pt_rms:.4} >= {POINT_TOL}",
                solver
            );
            assert!(
                prob.constraint_satisfied(&vars),
                "solver={:?} scale constraint not satisfied",
                solver
            );
        }
    }

    /// Schur and standard solvers must agree within tolerance.
    #[test]
    fn schur_matches_standard() {
        let prob = BaScaleConstraintProblem::new(4, 20);

        let standard_vars =
            prob.optimize_all_free(LinearSolverEnum::FaerSparseLu(FaerSparseLu {}), false);
        let schur_vars = prob.optimize_marg_points(
            LinearSolverEnum::SchurBlockSparseLdlt(BlockSparseLdlt::default()),
            false,
        );

        let (s_pose_rms, s_pt_rms) = prob.gt_errors(&standard_vars);
        let (h_pose_rms, h_pt_rms) = prob.gt_errors(&schur_vars);

        // Both must reach GT within tolerance.
        assert_abs_diff_eq!(s_pose_rms, h_pose_rms, epsilon = POSE_TOL);
        assert_abs_diff_eq!(s_pt_rms, h_pt_rms, epsilon = POINT_TOL);
    }
}
