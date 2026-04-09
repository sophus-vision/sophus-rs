use nalgebra::{
    DMatrix,
    DVector,
};
use sophus_solver::{
    InvertibleMatrix,
    error::LinearSolverError,
    matrix::{
        IsSymmetricMatrix,
        PartitionBlockIndex,
        PartitionSet,
        SymmetricMatrixEnum,
    },
};

/// `H + νI` — Hessian with LM damping baked in.
///
/// The underlying matrix stores `H + νI` in the first `num_var_partitions` partitions.
/// When equality constraints are present, additional KKT rows (G, Gᵀ, 0) follow
/// the variable partitions — these do NOT have ν on the diagonal.
///
/// `nu` is stored separately so `H` blocks can be recovered via [`get_block`](Self::get_block).
///
/// For covariance queries, call [`into_covariance`](Self::into_covariance) which
/// extracts the variable-only H, re-factorizes, and returns a [`Covariance`].
#[derive(Clone)]
pub struct DampedHessian {
    /// The underlying symmetric matrix (variable partitions store `H + νI`,
    /// constraint partitions store the KKT rows without damping).
    pub matrix: SymmetricMatrixEnum,
    /// LM damping ν added to the variable diagonal only.
    pub nu: f64,
    /// Number of variable partitions (free + marginalized). Damping ν applies
    /// only to partitions `0..num_var_partitions`.
    num_var_partitions: usize,
    /// Solver used for factorization (needed for re-factorizing without ν).
    solver: sophus_solver::LinearSolverEnum,
}

impl DampedHessian {
    /// Create a new `DampedHessian`.
    pub fn new(
        matrix: SymmetricMatrixEnum,
        nu: f64,
        num_var_partitions: usize,
        solver: sophus_solver::LinearSolverEnum,
    ) -> Self {
        Self {
            matrix,
            nu,
            num_var_partitions,
            solver,
        }
    }

    /// LM damping value ν.
    pub fn lm_damping(&self) -> f64 {
        self.nu
    }

    /// Block of `H` (ν subtracted from variable diagonal blocks).
    ///
    /// For variable partitions (`0..num_var_partitions`), diagonal blocks have ν
    /// subtracted to recover H from the stored `H + νI`.
    /// For constraint partitions (KKT rows), blocks are returned as stored (no ν).
    pub fn get_block(
        &self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> DMatrix<f64> {
        let mut block = self.matrix.get_block(row_idx, col_idx);
        // Only subtract ν from variable diagonal blocks, not constraint blocks.
        if row_idx.partition == col_idx.partition
            && row_idx.block == col_idx.block
            && row_idx.partition < self.num_var_partitions
        {
            let n = block.nrows();
            for i in 0..n {
                block[(i, i)] -= self.nu;
            }
        }
        block
    }

    /// Block of `H + νI` as stored (no ν subtraction).
    pub fn get_block_with_damping(
        &self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> DMatrix<f64> {
        self.matrix.get_block(row_idx, col_idx)
    }

    /// Block of `(H + νI)⁻¹` via the Schur formula or min-norm LDLᵀ.
    ///
    /// For Schur systems this reuses the cached factorization. For direct systems
    /// it uses the min-norm pseudo-inverse.
    pub fn inverse_block_with_damping(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> Result<DMatrix<f64>, LinearSolverError> {
        self.matrix.inverse_block(row_idx, col_idx)
    }

    /// Consume the `DampedHessian` and produce a [`Covariance`] for block-wise queries.
    ///
    /// Handles all cases uniformly:
    /// - With/without KKT constraint rows (extracts variable-only H when KKT present)
    /// - With/without equality constraints (applies rank-k projection when G provided)
    /// - Full-rank or rank-deficient H (min-norm pseudo-inverse handles gauge freedom)
    ///
    /// The resulting [`Covariance`] provides `covariance_block(i,j)` — Σ(i,j) with
    /// constraint projection (or H⁺(i,j) when no constraints).
    pub fn into_covariance(
        self,
        constraint_jacobian: Option<&DMatrix<f64>>,
    ) -> sophus_solver::covariance::Covariance {
        let num_var_partitions = self.num_var_partitions;
        use sophus_solver::{
            IsLinearSolver,
            ldlt::BlockSparseLdlt,
            matrix::{
                IsSymmetricMatrixBuilder,
                block_sparse::BlockSparseSymmetricMatrixBuilder,
                direct_solve::{
                    DirectSolve,
                    DirectSolveMatrix,
                },
            },
        };

        let all_specs = self.matrix.partitions().specs();
        let has_kkt_rows = all_specs.len() > num_var_partitions;

        if has_kkt_rows {
            // KKT rows present: extract variable-only H into a block-sparse matrix.
            let var_specs = all_specs[..num_var_partitions].to_vec();
            let var_partitions = PartitionSet::new(var_specs);

            let mut builder = BlockSparseSymmetricMatrixBuilder::zero(var_partitions.clone());
            for row_p in 0..num_var_partitions {
                for row_b in 0..var_partitions.specs()[row_p].block_count {
                    let row_idx = PartitionBlockIndex {
                        partition: row_p,
                        block: row_b,
                    };
                    for col_p in 0..=row_p {
                        let max_b = if col_p == row_p {
                            row_b
                        } else {
                            var_partitions.specs()[col_p].block_count - 1
                        };
                        for col_b in 0..=max_b {
                            let col_idx = PartitionBlockIndex {
                                partition: col_p,
                                block: col_b,
                            };
                            // get_block subtracts ν from diagonal blocks.
                            let block = self.get_block(row_idx, col_idx);
                            builder.add_lower_block(row_idx, col_idx, &block.as_view());
                        }
                    }
                }
            }
            let block_sparse = builder.build();

            // Factorize with BlockSparseLdlt (PSD solver, efficient block queries).
            let factor = BlockSparseLdlt::default()
                .factorize(&block_sparse)
                .expect("variable H factorization for covariance");
            let factor_enum = sophus_solver::FactorEnum::BlockSparseLdlt(factor);
            let invertible = factor_enum.into_invertible().unwrap_or_else(|| {
                // Fallback to dense if block-sparse fails.
                let dense_sym = block_sparse.to_dense_symmetric();
                let dense_h = DMatrix::from(dense_sym.view());
                let mat = sophus_solver::matrix::dense::DenseSymmetricMatrix::new(
                    dense_h,
                    var_partitions.clone(),
                );
                let dense_factor = sophus_solver::ldlt::DenseLdlt::default()
                    .factorize(&mat)
                    .expect("dense fallback factorization");
                InvertibleMatrix::Dense(
                    sophus_solver::ldlt::min_norm_ldlt::dense_min_norm_ldlt::DenseMinNormFactor::new(
                        dense_factor,
                    ),
                )
            });

            sophus_solver::covariance::Covariance::new(
                invertible,
                var_partitions,
                constraint_jacobian,
            )
        } else {
            // No KKT rows: factorize the full matrix directly, preserving storage format.
            let mut matrix = self.matrix;
            if self.nu != 0.0 {
                matrix.subtract_scalar_diagonal(self.nu);
            }

            let effective_solver = if self.solver.is_schur() {
                self.solver.schur_inner_solver()
            } else {
                self.solver
            };

            let partitions = matrix.partitions().clone();
            let h = match matrix {
                SymmetricMatrixEnum::Direct(ds) => {
                    SymmetricMatrixEnum::Direct(DirectSolve::new(ds.inner, effective_solver))
                }
                SymmetricMatrixEnum::Schur(s) => SymmetricMatrixEnum::Direct(DirectSolve::new(
                    DirectSolveMatrix::BlockSparseLower(s.inner),
                    effective_solver,
                )),
            };

            let factor = effective_solver
                .factorize(&h)
                .expect("H factorization for covariance");
            let invertible = factor.into_invertible().unwrap_or_else(|| {
                let dense = h.to_dense();
                let parts = h.partitions().clone();
                let mat =
                    sophus_solver::matrix::dense::DenseSymmetricMatrix::new(dense, parts);
                let dense_factor = sophus_solver::ldlt::DenseLdlt::default()
                    .factorize(&mat)
                    .expect("dense fallback factorization");
                InvertibleMatrix::Dense(
                    sophus_solver::ldlt::min_norm_ldlt::dense_min_norm_ldlt::DenseMinNormFactor::new(
                        dense_factor,
                    ),
                )
            });

            sophus_solver::covariance::Covariance::new(invertible, partitions, constraint_jacobian)
        }
    }

    /// Solve `(H + νI) x = rhs`.
    pub fn solve(&mut self, rhs: &DVector<f64>) -> Result<DVector<f64>, LinearSolverError> {
        self.matrix.solve(rhs)
    }

    /// Partitions of the underlying matrix.
    pub fn partitions(&self) -> &PartitionSet {
        self.matrix.partitions()
    }
}

/// Compute the Schur complement fill-in pattern for the free-variable blocks.
///
/// Given a symmetric matrix with `num_free_partitions` free partitions followed by
/// marginalized partitions, returns which free-block pairs `(gi, gj)` will be non-zero
/// in the Schur complement `S_ff = H_ff - H_fm H_mm⁻¹ H_mf`.
///
/// Includes both the original `H_ff` sparsity and fill-in from shared marginalized blocks.
/// Global block indices enumerate all blocks across the free partitions in order.
pub fn schur_fill_in(
    matrix: &SymmetricMatrixEnum,
    num_free_partitions: usize,
) -> std::collections::HashSet<(usize, usize)> {
    let specs = matrix.partitions().specs();

    // Map global block index → (partition, block) for free partitions.
    let mut block_map: Vec<(usize, usize)> = Vec::new();
    for (partition, spec) in specs.iter().take(num_free_partitions).enumerate() {
        for block in 0..spec.block_count {
            block_map.push((partition, block));
        }
    }
    let total_free_blocks = block_map.len();

    let mut fill = std::collections::HashSet::new();

    // Original free-free sparsity.
    for gi in 0..total_free_blocks {
        let (pi, bi) = block_map[gi];
        let idx_i = PartitionBlockIndex {
            partition: pi,
            block: bi,
        };
        for gj in 0..total_free_blocks {
            let (pj, bj) = block_map[gj];
            let idx_j = PartitionBlockIndex {
                partition: pj,
                block: bj,
            };
            if matrix.has_block(idx_i, idx_j) {
                fill.insert((gi, gj));
            }
        }
    }

    // Fill-in from shared marginalized blocks: if free blocks i and j are both
    // coupled to the same marginalized block, then (i, j) is non-zero in S_ff.
    for marg_p in num_free_partitions..specs.len() {
        for marg_b in 0..specs[marg_p].block_count {
            let marg_idx = PartitionBlockIndex {
                partition: marg_p,
                block: marg_b,
            };
            let mut coupled_free: Vec<usize> = Vec::new();
            for gi in 0..total_free_blocks {
                let (pi, bi) = block_map[gi];
                let free_idx = PartitionBlockIndex {
                    partition: pi,
                    block: bi,
                };
                if matrix.has_block(free_idx, marg_idx) {
                    coupled_free.push(gi);
                }
            }
            for &a in &coupled_free {
                for &b in &coupled_free {
                    fill.insert((a, b));
                }
            }
        }
    }

    fill
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use sophus_solver::{
        LinearSolverEnum,
        ldlt::{
            BlockSparseLdlt,
            FaerSparseLdlt,
            SparseLdlt,
        },
        matrix::{
            IsSymmetricMatrix,
            PartitionBlockIndex,
            PartitionSet,
        },
    };

    use crate::{
        example_problems::ba_problem::BaProblem,
        nlls::{
            EvalMode,
            NllsError,
            OptParams,
            linear_system::{
                LinearSystem,
                cost_system::CostSystem,
                eq_system::EqSystem,
            },
        },
        variables::VarFamilies,
    };

    fn build_linear_system(
        ba: &BaProblem,
        vars: VarFamilies,
        nu: f64,
        solver: LinearSolverEnum,
    ) -> Result<LinearSystem, NllsError> {
        let params = OptParams {
            num_iterations: 1,
            initial_lm_damping: nu,
            parallelize: false,
            solver,
            skip_final_hessian: false,
            ..Default::default()
        };
        let mut cost_system = CostSystem::new(&vars, vec![ba.build_cost()], params).unwrap();
        let eq_system = EqSystem::new(&vars, vec![], params).unwrap();
        cost_system
            .eval(&vars, EvalMode::CalculateDerivatives, params)
            .unwrap();
        let (ls, _pattern) = LinearSystem::from_families_costs_and_constraints(
            &vars,
            &cost_system.evaluated_costs,
            nu,
            &eq_system,
            params.solver,
            params.parallelize,
            None,
        )
        .map_err(|e| NllsError::NllsCostSystemError { source: e })?;
        Ok(ls)
    }

    /// Dense pseudo-inverse of H + νI from the full matrix.
    fn dense_h_plus_nu_inv(ls: &LinearSystem) -> nalgebra::DMatrix<f64> {
        let dense = ls.damped_hessian().matrix.to_dense();
        nalgebra::SVD::new(dense, true, true)
            .pseudo_inverse(1e-10)
            .expect("pseudo-inverse of H+νI")
    }

    /// Dense pseudo-inverse of H (ν subtracted from diagonal) from the full matrix.
    fn dense_h_inv(ls: &LinearSystem) -> nalgebra::DMatrix<f64> {
        let mut dense = ls.damped_hessian().matrix.to_dense();
        let nu = ls.damped_hessian().nu;
        let n = dense.nrows();
        for i in 0..n {
            dense[(i, i)] -= nu;
        }
        nalgebra::SVD::new(dense, true, true)
            .pseudo_inverse(1e-10)
            .expect("pseudo-inverse of H")
    }

    fn all_block_indices(parts: &PartitionSet) -> Vec<(PartitionBlockIndex, PartitionBlockIndex)> {
        let mut pairs = vec![];
        for rp in 0..parts.len() {
            for rb in 0..parts.specs()[rp].block_count {
                for cp in 0..parts.len() {
                    for cb in 0..parts.specs()[cp].block_count {
                        pairs.push((
                            PartitionBlockIndex {
                                partition: rp,
                                block: rb,
                            },
                            PartitionBlockIndex {
                                partition: cp,
                                block: cb,
                            },
                        ));
                    }
                }
            }
        }
        pairs
    }

    /// `get_block_with_damping` must match the corresponding subblock of the stored dense matrix.
    fn check_get_block_with_damping(ls: &LinearSystem) {
        let dense = ls.damped_hessian().matrix.to_dense();
        let parts = ls.damped_hessian().matrix.partitions().clone();
        for (row_idx, col_idx) in all_block_indices(&parts) {
            let got = ls.damped_hessian().get_block_with_damping(row_idx, col_idx);
            let row_range = parts.block_range(row_idx);
            let col_range = parts.block_range(col_idx);
            let want = dense
                .view(
                    (row_range.start_idx, col_range.start_idx),
                    (row_range.block_dim, col_range.block_dim),
                )
                .into_owned();
            assert_abs_diff_eq!(got, want, epsilon = 1e-12);
        }
    }

    /// `get_block` must equal `get_block_with_damping` everywhere except on the diagonal, where ν
    /// is subtracted.
    fn check_get_block(ls: &LinearSystem) {
        let nu = ls.damped_hessian().nu;
        let parts = ls.damped_hessian().matrix.partitions().clone();
        for (row_idx, col_idx) in all_block_indices(&parts) {
            let with_damping = ls.damped_hessian().get_block_with_damping(row_idx, col_idx);
            let without = ls.damped_hessian().get_block(row_idx, col_idx);
            if row_idx.partition == col_idx.partition && row_idx.block == col_idx.block {
                let n = with_damping.nrows();
                for i in 0..n {
                    assert_abs_diff_eq!(
                        without[(i, i)],
                        with_damping[(i, i)] - nu,
                        epsilon = 1e-12
                    );
                }
                for i in 0..n {
                    for j in 0..n {
                        if i != j {
                            assert_abs_diff_eq!(
                                without[(i, j)],
                                with_damping[(i, j)],
                                epsilon = 1e-12
                            );
                        }
                    }
                }
            } else {
                assert_abs_diff_eq!(without, with_damping, epsilon = 1e-12);
            }
        }
    }

    /// `into_covariance().covariance_block()` must match the dense pseudo-inverse of H.
    fn check_covariance_blocks(ls: &LinearSystem) {
        let h_inv = dense_h_inv(ls);
        let parts = ls.damped_hessian().matrix.partitions().clone();
        let mut cov = ls.damped_hessian().clone().into_covariance(None);
        for (row_idx, col_idx) in all_block_indices(&parts) {
            let row_range = parts.block_range(row_idx);
            let col_range = parts.block_range(col_idx);
            let got = cov.covariance_block(row_idx, col_idx);
            let want = h_inv
                .view(
                    (row_range.start_idx, col_range.start_idx),
                    (row_range.block_dim, col_range.block_dim),
                )
                .into_owned();
            assert_abs_diff_eq!(got, want, epsilon = 1e-6);
        }
    }

    /// `inverse_block_with_damping` must match the corresponding block of (H+νI)⁻¹.
    fn check_inverse_block_with_damping(ls: &mut LinearSystem) {
        let h_plus_nu_inv = dense_h_plus_nu_inv(ls);
        let parts = ls.damped_hessian().matrix.partitions().clone();
        for (row_idx, col_idx) in all_block_indices(&parts) {
            let row_range = parts.block_range(row_idx);
            let col_range = parts.block_range(col_idx);
            let got = ls
                .damped_hessian_mut()
                .inverse_block_with_damping(row_idx, col_idx)
                .expect("inverse_block_with_damping");
            let want = h_plus_nu_inv
                .view(
                    (row_range.start_idx, col_range.start_idx),
                    (row_range.block_dim, col_range.block_dim),
                )
                .into_owned();
            assert_abs_diff_eq!(got, want, epsilon = 1e-6);
        }
    }

    /// Run all block-access and pseudo-inverse checks for one solver.
    fn run_all_checks(solver: LinearSolverEnum, use_gauge_fix: bool) {
        let ba = BaProblem::new(4, 10);
        let nu = 1.0_f64;
        let vars = ba.build_initial_variables(use_gauge_fix);
        let mut ls = build_linear_system(&ba, vars, nu, solver).expect("linear system");

        check_get_block_with_damping(&ls);
        check_get_block(&ls);
        check_covariance_blocks(&ls);
        check_inverse_block_with_damping(&mut ls);
    }

    // ── BlockSparseLdlt (direct, no Schur) ───────────────────────────────────

    #[test]
    fn block_sparse_ldlt_no_gauge() {
        run_all_checks(
            LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default()),
            false,
        );
    }

    #[test]
    fn block_sparse_ldlt_with_gauge() {
        run_all_checks(
            LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default()),
            true,
        );
    }

    // ── SchurBlockSparseLdlt ──────────────────────────────────────────────────

    #[test]
    fn schur_block_sparse_ldlt_no_gauge() {
        run_all_checks(
            LinearSolverEnum::SchurBlockSparseLdlt(BlockSparseLdlt::default()),
            false,
        );
    }

    #[test]
    fn schur_block_sparse_ldlt_with_gauge() {
        run_all_checks(
            LinearSolverEnum::SchurBlockSparseLdlt(BlockSparseLdlt::default()),
            true,
        );
    }

    // ── SchurSparseLdlt ───────────────────────────────────────────────────────

    #[test]
    fn schur_sparse_ldlt_no_gauge() {
        run_all_checks(
            LinearSolverEnum::SchurSparseLdlt(SparseLdlt::default()),
            false,
        );
    }

    #[test]
    fn schur_sparse_ldlt_with_gauge() {
        run_all_checks(
            LinearSolverEnum::SchurSparseLdlt(SparseLdlt::default()),
            true,
        );
    }

    // ── SchurFaerSparseLdlt ───────────────────────────────────────────────────

    #[test]
    fn schur_faer_sparse_ldlt_with_gauge() {
        run_all_checks(
            LinearSolverEnum::SchurFaerSparseLdlt(FaerSparseLdlt::default()),
            true,
        );
    }

    // ── Schur covariance tests ──────────────────────────────────────────────

    /// `inverse_block_with_damping` must match the dense pseudo-inverse of `H + νI`.
    ///
    /// Uses `SchurBlockSparseLdlt` so `inverse_block_with_damping` takes the efficient Schur path.
    #[test]
    fn schur_inverse_block_matches_dense_pseudo_inverse() {
        let ba = BaProblem::new(4, 20);
        let nu = 1.0_f64;
        let solver = LinearSolverEnum::SchurBlockSparseLdlt(BlockSparseLdlt::default());

        let vars_schur = ba.build_initial_variables(true);
        let mut ls = build_linear_system(&ba, vars_schur, nu, solver).expect("Schur linear system");

        let h_inv = {
            let dense = ls.damped_hessian().matrix.to_dense();
            nalgebra::SVD::new(dense, true, true)
                .pseudo_inverse(1e-10)
                .expect("pseudo-inverse")
        };

        let schur = ls
            .damped_hessian()
            .matrix
            .as_schur()
            .expect("Schur variant");
        let num_free_partitions = schur.num_free_partitions;
        let all_specs = schur.inner.partitions().specs().to_vec();
        let full_parts = PartitionSet::new(all_specs.clone());
        let free_parts = PartitionSet::new(all_specs[..num_free_partitions].to_vec());
        let n_parts = full_parts.len();

        let scalar_start = |p: usize, b: usize| -> usize {
            full_parts.scalar_offsets_by_partition()[p] + b * full_parts.specs()[p].block_dim
        };

        // free × free
        for row_p in 0..num_free_partitions {
            let row_dim = free_parts.specs()[row_p].block_dim;
            for row_b in 0..free_parts.specs()[row_p].block_count {
                for col_p in 0..num_free_partitions {
                    let col_dim = free_parts.specs()[col_p].block_dim;
                    for col_b in 0..free_parts.specs()[col_p].block_count {
                        let row_idx = PartitionBlockIndex {
                            partition: row_p,
                            block: row_b,
                        };
                        let col_idx = PartitionBlockIndex {
                            partition: col_p,
                            block: col_b,
                        };
                        let got = ls
                            .damped_hessian_mut()
                            .inverse_block_with_damping(row_idx, col_idx)
                            .expect("inverse_block_with_damping");
                        let row_start = scalar_start(row_p, row_b);
                        let col_start = scalar_start(col_p, col_b);
                        let want = h_inv
                            .view((row_start, col_start), (row_dim, col_dim))
                            .into_owned();
                        assert_abs_diff_eq!(got, want, epsilon = 1e-6);
                    }
                }
            }
        }

        // marg diagonal
        for marg_p in num_free_partitions..n_parts {
            let marg_dim = full_parts.specs()[marg_p].block_dim;
            for marg_b in 0..full_parts.specs()[marg_p].block_count {
                let marg_idx = PartitionBlockIndex {
                    partition: marg_p,
                    block: marg_b,
                };
                let got = ls
                    .damped_hessian_mut()
                    .inverse_block_with_damping(marg_idx, marg_idx)
                    .expect("inverse_block_with_damping");
                let marg_start = scalar_start(marg_p, marg_b);
                let want = h_inv
                    .view((marg_start, marg_start), (marg_dim, marg_dim))
                    .into_owned();
                assert_abs_diff_eq!(got, want, epsilon = 1e-6);
            }
        }

        // free × marg cross
        for free_p in 0..num_free_partitions {
            let free_dim = free_parts.specs()[free_p].block_dim;
            for free_b in 0..free_parts.specs()[free_p].block_count {
                let free_idx = PartitionBlockIndex {
                    partition: free_p,
                    block: free_b,
                };
                for marg_p in num_free_partitions..n_parts {
                    let marg_dim = full_parts.specs()[marg_p].block_dim;
                    for marg_b in 0..full_parts.specs()[marg_p].block_count {
                        let marg_idx = PartitionBlockIndex {
                            partition: marg_p,
                            block: marg_b,
                        };
                        let got = ls
                            .damped_hessian_mut()
                            .inverse_block_with_damping(free_idx, marg_idx)
                            .expect("inverse_block_with_damping");
                        let free_start = scalar_start(free_p, free_b);
                        let marg_start = scalar_start(marg_p, marg_b);
                        let want = h_inv
                            .view((free_start, marg_start), (free_dim, marg_dim))
                            .into_owned();
                        assert_abs_diff_eq!(got, want, epsilon = 1e-6);
                    }
                }
            }
        }
    }

    // ── schur_fill_in tests ─────────────────────────────────────────────────

    #[test]
    fn schur_fill_in_ba_problem() {
        let ba = BaProblem::new(4, 10);
        let nu = 1.0_f64;
        let solver = LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default());
        let vars = ba.build_initial_variables(true);
        let ls = build_linear_system(&ba, vars, nu, solver).expect("linear system");

        let num_free_parts = 1;
        let fill = super::schur_fill_in(&ls.damped_hessian().matrix, num_free_parts);

        assert!(!fill.is_empty(), "fill-in set should not be empty");
        assert!(
            fill.contains(&(0, 0)),
            "diagonal block (0,0) must be present"
        );
    }
}
