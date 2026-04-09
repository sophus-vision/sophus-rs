//! Comprehensive tests for solver × variable-kind × constraint combinations.
//!
//! Validates that:
//! - Valid configurations converge correctly.
//! - Invalid configurations (Schur without marg, Schur with constraint on marg, etc.) produce clear
//!   errors early (death tests).

#[cfg(test)]
mod tests {
    use sophus_autodiff::linalg::{
        MatF64,
        VecF64,
    };
    use sophus_solver::{
        LinearSolverEnum,
        ldlt::{
            BlockSparseLdlt,
            FaerSparseLdlt,
        },
        lu::FaerSparseLu,
        qr::FaerSparseQr,
    };

    use crate::{
        nlls::{
            CostFn,
            CostTerms,
            EqConstraintFn,
            EqConstraints,
            EvaluatedCostTerm,
            EvaluatedEqConstraint,
            OptParams,
            optimize_nlls,
            optimize_nlls_with_eq_constraints,
        },
        prelude::*,
        robust_kernel,
        variables::{
            VarBuilder,
            VarFamily,
            VarKind,
        },
    };

    extern crate alloc;

    // ── Minimal cost terms for testing ──────────────────────────────────

    #[derive(Clone, Debug)]
    struct ScalarPriorCost {
        target: f64,
        entity_indices: [usize; 1],
    }

    impl HasResidualFn<1, 1, (), VecF64<1>> for ScalarPriorCost {
        fn idx_ref(&self) -> &[usize; 1] {
            &self.entity_indices
        }

        fn eval(
            &self,
            _: &(),
            idx: [usize; 1],
            x: VecF64<1>,
            var_kinds: [VarKind; 1],
            robust_kernel: Option<robust_kernel::RobustKernel>,
        ) -> EvaluatedCostTerm<1, 1> {
            let residual = VecF64::<1>::new(x[0] - self.target);
            let d0 = || MatF64::<1, 1>::identity();
            (d0,).make(idx, var_kinds, residual, robust_kernel, None)
        }
    }

    #[derive(Clone, Debug)]
    struct ScalarPairCost {
        entity_indices: [usize; 2],
    }

    impl HasResidualFn<2, 2, (), (VecF64<1>, VecF64<1>)> for ScalarPairCost {
        fn idx_ref(&self) -> &[usize; 2] {
            &self.entity_indices
        }

        fn eval(
            &self,
            _: &(),
            idx: [usize; 2],
            (a, b): (VecF64<1>, VecF64<1>),
            var_kinds: [VarKind; 2],
            robust_kernel: Option<robust_kernel::RobustKernel>,
        ) -> EvaluatedCostTerm<2, 2> {
            let residual = VecF64::<1>::new(a[0] - b[0]);
            let d0 = || MatF64::<1, 1>::identity();
            let d1 = || -MatF64::<1, 1>::identity();
            (d0, d1).make(idx, var_kinds, residual, robust_kernel, None)
        }
    }

    #[derive(Clone, Debug)]
    struct ScalarNormConstraint {
        target: f64,
        entity_indices: [usize; 1],
    }

    impl HasEqConstraintResidualFn<1, 1, 1, (), VecF64<1>> for ScalarNormConstraint {
        fn idx_ref(&self) -> &[usize; 1] {
            &self.entity_indices
        }

        fn eval(
            &self,
            _: &(),
            idx: [usize; 1],
            x: VecF64<1>,
            var_kinds: [VarKind; 1],
        ) -> EvaluatedEqConstraint<1, 1, 1> {
            let residual = VecF64::<1>::new(x[0] - self.target);
            let d0 = || MatF64::<1, 1>::identity();
            (d0,).make_eq(idx, var_kinds, residual)
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    fn default_params(solver: LinearSolverEnum) -> OptParams {
        OptParams {
            num_iterations: 20,
            initial_lm_damping: 1e-6,
            parallelize: false,
            solver,
            ..Default::default()
        }
    }

    fn build_two_family_problem(
        num_poses: usize,
        num_points: usize,
        pose_kind: VarKind,
        point_kind: VarKind,
        target: f64,
    ) -> (
        crate::variables::VarFamilies,
        alloc::vec::Vec<alloc::boxed::Box<dyn crate::nlls::IsCostFn>>,
    ) {
        let poses: alloc::vec::Vec<VecF64<1>> =
            (0..num_poses).map(|_| VecF64::<1>::new(0.0)).collect();
        let points: alloc::vec::Vec<VecF64<1>> =
            (0..num_points).map(|_| VecF64::<1>::new(0.0)).collect();

        let vars = VarBuilder::new()
            .add_family("points", VarFamily::new(point_kind, points))
            .add_family("poses", VarFamily::new(pose_kind, poses))
            .build();

        let pose_priors: alloc::vec::Vec<ScalarPriorCost> = (0..num_poses)
            .map(|i| ScalarPriorCost {
                target,
                entity_indices: [i],
            })
            .collect();
        let pose_cost: alloc::boxed::Box<dyn crate::nlls::IsCostFn> =
            CostFn::new_boxed((), CostTerms::new(["poses"], pose_priors));

        let pair_costs: alloc::vec::Vec<ScalarPairCost> = (0..num_poses)
            .map(|i| ScalarPairCost {
                entity_indices: [i, i % num_points],
            })
            .collect();
        let pair_cost: alloc::boxed::Box<dyn crate::nlls::IsCostFn> =
            CostFn::new_boxed((), CostTerms::new(["poses", "points"], pair_costs));

        let point_priors: alloc::vec::Vec<ScalarPriorCost> = (0..num_points)
            .map(|i| ScalarPriorCost {
                target,
                entity_indices: [i],
            })
            .collect();
        let point_cost: alloc::boxed::Box<dyn crate::nlls::IsCostFn> =
            CostFn::new_boxed((), CostTerms::new(["points"], point_priors));

        (vars, alloc::vec![pose_cost, pair_cost, point_cost])
    }

    // ── Convergence: no equality constraints ────────────────────────────

    #[test]
    fn all_free_no_eq_converges() {
        for solver in [
            LinearSolverEnum::FaerSparseLu(FaerSparseLu {}),
            LinearSolverEnum::FaerSparseQr(FaerSparseQr {}),
            LinearSolverEnum::FaerSparseLdlt(FaerSparseLdlt::default()),
            LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default()),
        ] {
            let (vars, costs) = build_two_family_problem(3, 4, VarKind::Free, VarKind::Free, 5.0);
            let solution = optimize_nlls(vars, costs, default_params(solver))
                .unwrap_or_else(|e| panic!("solver {solver:?} failed: {e}"));
            assert!(
                solution.final_cost < 1e-4,
                "solver {solver:?}: cost={:.6}",
                solution.final_cost
            );
        }
    }

    #[test]
    fn schur_free_marg_no_eq_converges() {
        for solver in [
            LinearSolverEnum::SchurBlockSparseLdlt(BlockSparseLdlt::default()),
            LinearSolverEnum::SchurFaerSparseLdlt(FaerSparseLdlt::default()),
        ] {
            let (vars, costs) =
                build_two_family_problem(3, 4, VarKind::Free, VarKind::Marginalized, 5.0);
            let solution = optimize_nlls(vars, costs, default_params(solver))
                .unwrap_or_else(|e| panic!("solver {solver:?} failed: {e}"));
            assert!(
                solution.final_cost < 1e-4,
                "solver {solver:?}: cost={:.6}",
                solution.final_cost
            );
        }
    }

    #[test]
    fn non_schur_with_marg_no_eq_converges() {
        let solver = LinearSolverEnum::FaerSparseLu(FaerSparseLu {});
        let (vars, costs) =
            build_two_family_problem(3, 4, VarKind::Free, VarKind::Marginalized, 5.0);
        let solution = optimize_nlls(vars, costs, default_params(solver)).unwrap();
        assert!(solution.final_cost < 1e-4);
    }

    // ── Convergence: with equality constraints ──────────────────────────

    #[test]
    fn all_free_with_eq_converges() {
        for solver in [
            LinearSolverEnum::FaerSparseLu(FaerSparseLu {}),
            LinearSolverEnum::FaerSparseQr(FaerSparseQr {}),
        ] {
            let (vars, costs) = build_two_family_problem(3, 4, VarKind::Free, VarKind::Free, 5.0);
            let eq = EqConstraintFn::new_boxed(
                (),
                EqConstraints::new(
                    ["poses"],
                    alloc::vec![ScalarNormConstraint {
                        target: 3.0,
                        entity_indices: [1],
                    }],
                ),
            );
            let solution = optimize_nlls_with_eq_constraints(
                vars,
                costs,
                alloc::vec![eq],
                default_params(solver),
            )
            .unwrap_or_else(|e| panic!("solver {solver:?} failed: {e}"));
            let poses = solution.variables.get_members::<VecF64<1>>("poses");
            assert!(
                (poses[1][0] - 3.0).abs() < 1e-3,
                "solver {solver:?}: poses[1]={:.4}, expected ~3.0",
                poses[1][0]
            );
        }
    }

    #[test]
    fn schur_with_eq_on_free_converges() {
        for solver in [
            LinearSolverEnum::SchurBlockSparseLdlt(BlockSparseLdlt::default()),
            LinearSolverEnum::SchurFaerSparseLdlt(FaerSparseLdlt::default()),
        ] {
            let (vars, costs) =
                build_two_family_problem(3, 4, VarKind::Free, VarKind::Marginalized, 5.0);
            let eq = EqConstraintFn::new_boxed(
                (),
                EqConstraints::new(
                    ["poses"],
                    alloc::vec![ScalarNormConstraint {
                        target: 3.0,
                        entity_indices: [1],
                    }],
                ),
            );
            let solution = optimize_nlls_with_eq_constraints(
                vars,
                costs,
                alloc::vec![eq],
                default_params(solver),
            )
            .unwrap_or_else(|e| panic!("solver {solver:?} failed: {e}"));
            let poses = solution.variables.get_members::<VecF64<1>>("poses");
            assert!(
                (poses[1][0] - 3.0).abs() < 1e-3,
                "solver {solver:?}: poses[1]={:.4}, expected ~3.0",
                poses[1][0]
            );
        }
    }

    // ── Death tests ─────────────────────────────────────────────────────

    #[test]
    fn death_schur_no_marg() {
        let solver = LinearSolverEnum::SchurBlockSparseLdlt(BlockSparseLdlt::default());
        let (vars, costs) = build_two_family_problem(3, 4, VarKind::Free, VarKind::Free, 5.0);
        let err = optimize_nlls(vars, costs, default_params(solver))
            .err()
            .expect("should fail: Schur without marg");
        assert!(format!("{err}").contains("marginalized"), "error: {err}");
    }

    #[test]
    fn death_schur_no_free() {
        let solver = LinearSolverEnum::SchurBlockSparseLdlt(BlockSparseLdlt::default());
        let (vars, costs) =
            build_two_family_problem(3, 4, VarKind::Conditioned, VarKind::Marginalized, 5.0);
        let err = optimize_nlls(vars, costs, default_params(solver))
            .err()
            .expect("should fail: Schur without free");
        let msg = format!("{err}");
        assert!(
            msg.contains("free") || msg.contains("marginalized"),
            "error: {msg}"
        );
    }

    #[test]
    fn death_schur_eq_on_marg() {
        let solver = LinearSolverEnum::SchurBlockSparseLdlt(BlockSparseLdlt::default());
        let (vars, costs) =
            build_two_family_problem(3, 4, VarKind::Free, VarKind::Marginalized, 5.0);
        let eq = EqConstraintFn::new_boxed(
            (),
            EqConstraints::new(
                ["points"],
                alloc::vec![ScalarNormConstraint {
                    target: 3.0,
                    entity_indices: [0],
                }],
            ),
        );
        let err =
            optimize_nlls_with_eq_constraints(vars, costs, alloc::vec![eq], default_params(solver))
                .err()
                .expect("should fail: constraint on marginalized");
        assert!(format!("{err}").contains("marginalized"), "error: {err}");
    }

    #[test]
    fn death_ldlt_with_eq() {
        let solver = LinearSolverEnum::FaerSparseLdlt(FaerSparseLdlt::default());
        let (vars, costs) = build_two_family_problem(3, 4, VarKind::Free, VarKind::Free, 5.0);
        let eq = EqConstraintFn::new_boxed(
            (),
            EqConstraints::new(
                ["poses"],
                alloc::vec![ScalarNormConstraint {
                    target: 3.0,
                    entity_indices: [1],
                }],
            ),
        );
        let err =
            optimize_nlls_with_eq_constraints(vars, costs, alloc::vec![eq], default_params(solver))
                .err()
                .expect("should fail: LDLt doesn't support eq constraints");
        assert!(
            format!("{err}").contains("does not support equality"),
            "error: {err}"
        );
    }

    #[test]
    fn death_schur_two_marg_families_in_cost() {
        let solver = LinearSolverEnum::SchurBlockSparseLdlt(BlockSparseLdlt::default());

        let vars = VarBuilder::new()
            .add_family(
                "poses",
                VarFamily::new(VarKind::Free, alloc::vec![VecF64::<1>::new(0.0)]),
            )
            .add_family(
                "points_a",
                VarFamily::new(VarKind::Marginalized, alloc::vec![VecF64::<1>::new(0.0)]),
            )
            .add_family(
                "points_b",
                VarFamily::new(VarKind::Marginalized, alloc::vec![VecF64::<1>::new(0.0)]),
            )
            .build();

        let bad_cost: alloc::boxed::Box<dyn crate::nlls::IsCostFn> = CostFn::new_boxed(
            (),
            CostTerms::new(
                ["points_a", "points_b"],
                alloc::vec![ScalarPairCost {
                    entity_indices: [0, 0],
                }],
            ),
        );
        let pose_prior: alloc::boxed::Box<dyn crate::nlls::IsCostFn> = CostFn::new_boxed(
            (),
            CostTerms::new(
                ["poses"],
                alloc::vec![ScalarPriorCost {
                    target: 1.0,
                    entity_indices: [0],
                }],
            ),
        );

        let err = optimize_nlls(
            vars,
            alloc::vec![bad_cost, pose_prior],
            default_params(solver),
        )
        .err()
        .expect("should fail: cost connects two marg families");
        assert!(format!("{err}").contains("marginalized"), "error: {err}");
    }
}
