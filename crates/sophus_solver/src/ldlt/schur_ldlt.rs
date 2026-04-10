//! Schur-complement factorization for block-structured linear systems.
//!
//! For a partitioned symmetric matrix with free (`f`) and marginalized (`m`) blocks:
//! ```text
//! H = [H_ff  H_fm]
//!     [H_mf  H_mm]
//! ```
//! where `H_mm` is block-diagonal (one small block per marginalized variable), this module builds
//! the Schur complement `S = H_ff - H_fm H_mm⁻¹ H_mf`, factors `S`, and enables:
//!
//! - **solve**: back-substitution to recover the full step `dx = [dx_f; dx_m]`
//! - **covariance**: block extraction from `H⁻¹` via the Schur-complement inverse formulae

use nalgebra::{
    DMatrix,
    DVector,
};

use crate::{
    FactorEnum,
    InvertibleMatrix,
    IsFactor,
    IsInvertible,
    LinearSolverEnum,
    error::{
        LinearSolverError,
        SingularKktConstraintSnafu,
        SingularMargBlockSnafu,
    },
    ldlt::BlockSparseLdlt,
    matrix::{
        IsSymmetricMatrix,
        IsSymmetricMatrixBuilder,
        PartitionBlockIndex,
        PartitionSet,
        PartitionSpec,
        SymmetricMatrixBuilderEnum,
        SymmetricMatrixEnum,
        block_sparse::{
            BlockSparseSymmetricMatrix,
            BlockSparseSymmetricMatrixBuilder,
            BlockSparseSymmetricMatrixPattern,
        },
    },
};

extern crate alloc;

// ── Per-block shared data ─────────────────────────────────────────────────────

/// Precomputed quantities for one marginalized variable block.
///
/// Computed once in the forward pass and reused for S-accumulation, rhs-update,
/// back-substitution, and covariance queries.
struct MargBlockContrib {
    /// H_mm[b]⁻¹
    h_mm_inv: DMatrix<f64>,
    /// H_mm[b]⁻¹ g_m[b]  (for updating the reduced RHS)
    h_mm_inv_g: DVector<f64>,
    /// Non-zero free-variable block indices for H_mf[b]
    non_zero_free_indices: alloc::vec::Vec<PartitionBlockIndex>,
    /// H_mf[b, free] blocks, one per non-zero entry
    h_mf_blocks: alloc::vec::Vec<DMatrix<f64>>,
    /// H_mm[b]⁻¹ H_mf[b, free], one per non-zero entry
    h_mm_inv_h_mf: alloc::vec::Vec<DMatrix<f64>>,
}

impl MargBlockContrib {
    /// Extract `(free_idx, H_mf block)` pairs for storage in SchurFactor.
    fn h_mf_row_pairs(&self) -> alloc::vec::Vec<(PartitionBlockIndex, DMatrix<f64>)> {
        self.non_zero_free_indices
            .iter()
            .zip(self.h_mf_blocks.iter())
            .map(|(&fi, blk)| (fi, blk.clone()))
            .collect()
    }
}

/// Compute the Schur contribution for one marginalized variable block.
///
/// Returns `None` (→ `SingularMargBlock` error at call site) if `H_mm[b]` is singular.
fn compute_marg_block_contrib(
    h: &BlockSparseSymmetricMatrix,
    partitions: &PartitionSet,
    num_free_partitions: usize,
    marg_p: usize,
    marg_b: usize,
    g_all: &DVector<f64>,
) -> Option<MargBlockContrib> {
    let marg_idx = PartitionBlockIndex {
        partition: marg_p,
        block: marg_b,
    };
    let marg_dim = partitions.specs()[marg_p].block_dim;
    let marg_range = partitions.block_range(marg_idx);

    let h_mm_b = h
        .try_get_lower_block_view(marg_idx, marg_idx)
        .map(|v| v.clone_owned())
        .unwrap_or_else(|| h.get_block(marg_idx, marg_idx));
    let h_mm_inv = h_mm_b.try_inverse()?;

    let h_mm_inv_g = &h_mm_inv * g_all.rows(marg_range.start_idx, marg_dim);

    let mut non_zero_free_indices = alloc::vec::Vec::new();
    let mut h_mf_blocks = alloc::vec::Vec::new();
    let mut h_mm_inv_h_mf = alloc::vec::Vec::new();

    for free_p in 0..num_free_partitions {
        for free_b in 0..partitions.specs()[free_p].block_count {
            let free_idx = PartitionBlockIndex {
                partition: free_p,
                block: free_b,
            };
            let Some(h_mf_owned) = h
                .try_get_lower_block_view(marg_idx, free_idx)
                .map(|v| v.clone_owned())
            else {
                continue;
            };
            let inv_h_mf = &h_mm_inv * &h_mf_owned;
            non_zero_free_indices.push(free_idx);
            h_mf_blocks.push(h_mf_owned);
            h_mm_inv_h_mf.push(inv_h_mf);
        }
    }

    Some(MargBlockContrib {
        h_mm_inv,
        h_mm_inv_g,
        non_zero_free_indices,
        h_mf_blocks,
        h_mm_inv_h_mf,
    })
}

/// Apply one marg-block contribution to a sparse S-builder and rhs_bar (sequential path).
fn apply_contrib_to_sparse(
    contrib: &MargBlockContrib,
    free_part_set: &PartitionSet,
    s_builder: &mut SymmetricMatrixBuilderEnum,
    rhs_bar: &mut DVector<f64>,
) {
    let num_non_zero = contrib.non_zero_free_indices.len();
    for i in 0..num_non_zero {
        let free_row_idx = contrib.non_zero_free_indices[i];
        let free_row_range = free_part_set.block_range(free_row_idx);
        let h_mf_row = &contrib.h_mf_blocks[i];

        // rhs_bar[free_row] -= H_mf_row^T H_mm⁻¹ g_m
        rhs_bar
            .rows_mut(free_row_range.start_idx, free_row_range.block_dim)
            .gemv_tr(-1.0, h_mf_row, &contrib.h_mm_inv_g, 1.0);

        for j in 0..=i {
            let free_col_idx = contrib.non_zero_free_indices[j];
            let free_col_range = free_part_set.block_range(free_col_idx);

            // Lower-triangular ordering.
            let (actual_row, actual_col, idx_i, idx_j) = if free_row_idx.partition
                > free_col_idx.partition
                || (free_row_idx.partition == free_col_idx.partition
                    && free_row_idx.block >= free_col_idx.block)
            {
                (free_row_idx, free_col_idx, i, j)
            } else {
                (free_col_idx, free_row_idx, j, i)
            };

            let row_dim = free_part_set.specs()[actual_row.partition].block_dim;
            let col_dim = free_part_set.specs()[actual_col.partition].block_dim;

            let mut schur_block = DMatrix::<f64>::zeros(row_dim, col_dim);
            schur_block.gemm_tr(
                -1.0,
                &contrib.h_mf_blocks[idx_i],
                &contrib.h_mm_inv_h_mf[idx_j],
                0.0,
            );
            s_builder.add_lower_block(actual_row, actual_col, &schur_block.as_view());
            let _ = free_col_range; // used via actual_col / rd / cd
        }
    }
}

/// Apply one marg-block contribution to dense accumulators (parallel path).
fn apply_contrib_to_dense(
    contrib: &MargBlockContrib,
    free_part_set: &PartitionSet,
    s_delta: &mut DMatrix<f64>,
    r_delta: &mut DVector<f64>,
) {
    let num_non_zero = contrib.non_zero_free_indices.len();

    for k in 0..num_non_zero {
        let free_range = free_part_set.block_range(contrib.non_zero_free_indices[k]);
        r_delta
            .rows_mut(free_range.start_idx, free_range.block_dim)
            .gemv_tr(1.0, &contrib.h_mf_blocks[k], &contrib.h_mm_inv_g, 1.0);
    }

    for i in 0..num_non_zero {
        let free_range_i = free_part_set.block_range(contrib.non_zero_free_indices[i]);
        for j in 0..=i {
            let free_range_j = free_part_set.block_range(contrib.non_zero_free_indices[j]);

            let (row_start, col_start, row_dim, col_dim, idx_i, idx_j) =
                if free_range_i.start_idx >= free_range_j.start_idx {
                    (
                        free_range_i.start_idx,
                        free_range_j.start_idx,
                        free_range_i.block_dim,
                        free_range_j.block_dim,
                        i,
                        j,
                    )
                } else {
                    (
                        free_range_j.start_idx,
                        free_range_i.start_idx,
                        free_range_j.block_dim,
                        free_range_i.block_dim,
                        j,
                        i,
                    )
                };

            s_delta
                .view_mut((row_start, col_start), (row_dim, col_dim))
                .gemm_tr(
                    1.0,
                    &contrib.h_mf_blocks[idx_i],
                    &contrib.h_mm_inv_h_mf[idx_j],
                    1.0,
                );
        }
    }
}

// ── Forward-pass implementations ─────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn schur_forward_pass_sequential(
    h: &BlockSparseSymmetricMatrix,
    partitions: &PartitionSet,
    free_part_set: &PartitionSet,
    num_free_partitions: usize,
    total_active: usize,
    g_all: &DVector<f64>,
    rhs_bar: &mut DVector<f64>,
    s_builder: &mut SymmetricMatrixBuilderEnum,
    h_mm_inv_cache: &mut alloc::vec::Vec<DMatrix<f64>>,
    h_mf_rows: &mut alloc::vec::Vec<alloc::vec::Vec<(PartitionBlockIndex, DMatrix<f64>)>>,
) -> Result<(), LinearSolverError> {
    for marg_p in num_free_partitions..total_active {
        for marg_b in 0..partitions.specs()[marg_p].block_count {
            let contrib = compute_marg_block_contrib(
                h,
                partitions,
                num_free_partitions,
                marg_p,
                marg_b,
                g_all,
            )
            .ok_or_else(|| {
                SingularMargBlockSnafu {
                    partition: marg_p,
                    block: marg_b,
                }
                .build()
            })?;

            apply_contrib_to_sparse(&contrib, free_part_set, s_builder, rhs_bar);

            h_mf_rows.push(contrib.h_mf_row_pairs());
            h_mm_inv_cache.push(contrib.h_mm_inv);
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
fn schur_forward_pass_parallel(
    h: &BlockSparseSymmetricMatrix,
    partitions: &PartitionSet,
    free_part_set: &PartitionSet,
    num_free_partitions: usize,
    total_active: usize,
    num_free_scalars: usize,
    g_all: &DVector<f64>,
    rhs_bar: &mut DVector<f64>,
    s_builder: &mut SymmetricMatrixBuilderEnum,
    h_mm_inv_cache: &mut alloc::vec::Vec<DMatrix<f64>>,
    h_mf_rows: &mut alloc::vec::Vec<alloc::vec::Vec<(PartitionBlockIndex, DMatrix<f64>)>>,
) -> Result<(), LinearSolverError> {
    use rayon::prelude::*;

    let total_marg_blocks: usize = (num_free_partitions..total_active)
        .map(|p| partitions.specs()[p].block_count)
        .sum();
    let num_threads = rayon::current_num_threads().max(1);

    // Enumerate all marg blocks with a linear index that preserves iteration order for cache.
    let all_marg: alloc::vec::Vec<(usize, usize, usize)> = (num_free_partitions..total_active)
        .flat_map(|mp| {
            let bc = partitions.specs()[mp].block_count;
            (0..bc).map(move |mb| (mp, mb))
        })
        .enumerate()
        .map(|(i, (mp, mb))| (i, mp, mb))
        .collect();

    #[allow(clippy::type_complexity)]
    struct ParAcc {
        s_delta: DMatrix<f64>,
        r_delta: DVector<f64>,
        /// (linear_idx, H_mm⁻¹, H_mf_rows) for each processed block.
        block_list: alloc::vec::Vec<(
            usize,
            DMatrix<f64>,
            alloc::vec::Vec<(PartitionBlockIndex, DMatrix<f64>)>,
        )>,
        /// Error if any marg block was singular.
        err: Option<LinearSolverError>,
    }

    let result = all_marg
        .par_iter()
        .fold(
            || ParAcc {
                s_delta: DMatrix::zeros(num_free_scalars, num_free_scalars),
                r_delta: DVector::zeros(num_free_scalars),
                block_list: alloc::vec::Vec::with_capacity(total_marg_blocks / num_threads + 8),
                err: None,
            },
            |mut acc, &(linear_idx, marg_p, marg_b)| {
                if acc.err.is_some() {
                    return acc;
                }
                let Some(contrib) = compute_marg_block_contrib(
                    h,
                    partitions,
                    num_free_partitions,
                    marg_p,
                    marg_b,
                    g_all,
                ) else {
                    acc.err = Some(
                        SingularMargBlockSnafu {
                            partition: marg_p,
                            block: marg_b,
                        }
                        .build(),
                    );
                    return acc;
                };

                apply_contrib_to_dense(&contrib, free_part_set, &mut acc.s_delta, &mut acc.r_delta);

                let row_pairs = contrib.h_mf_row_pairs();
                acc.block_list
                    .push((linear_idx, contrib.h_mm_inv, row_pairs));
                acc
            },
        )
        .reduce(
            || ParAcc {
                s_delta: DMatrix::zeros(num_free_scalars, num_free_scalars),
                r_delta: DVector::zeros(num_free_scalars),
                block_list: alloc::vec::Vec::new(),
                err: None,
            },
            |mut a, mut b| {
                if a.err.is_some() {
                    return a;
                }
                if b.err.is_some() {
                    return b;
                }
                a.s_delta += b.s_delta;
                a.r_delta += b.r_delta;
                a.block_list.append(&mut b.block_list);
                a
            },
        );

    if let Some(e) = result.err {
        return Err(e);
    }

    // Restore marg-block order for back-sub and covariance.
    let mut block_list = result.block_list;
    block_list.sort_unstable_by_key(|&(idx, _, _)| idx);
    for (_, h_mm_inv, non_zero_rows) in block_list {
        h_mm_inv_cache.push(h_mm_inv);
        h_mf_rows.push(non_zero_rows);
    }

    // Apply -s_delta to s_builder (lower-triangular free-free blocks).
    let max_free_dim = (0..num_free_partitions)
        .map(|p| partitions.specs()[p].block_dim)
        .max()
        .unwrap_or(0);
    let mut neg_scratch = DMatrix::<f64>::zeros(max_free_dim, max_free_dim);
    for row_p in 0..num_free_partitions {
        for row_b in 0..partitions.specs()[row_p].block_count {
            let row_idx = PartitionBlockIndex {
                partition: row_p,
                block: row_b,
            };
            let row_range = free_part_set.block_range(row_idx);
            let row_dim = row_range.block_dim;
            for col_p in 0..=row_p {
                let col_b_end = if col_p < row_p {
                    partitions.specs()[col_p].block_count
                } else {
                    row_b + 1
                };
                for col_b in 0..col_b_end {
                    let col_idx = PartitionBlockIndex {
                        partition: col_p,
                        block: col_b,
                    };
                    let col_range = free_part_set.block_range(col_idx);
                    let col_dim = col_range.block_dim;
                    {
                        let mut v = neg_scratch.view_mut((0, 0), (row_dim, col_dim));
                        v.copy_from(&result.s_delta.view(
                            (row_range.start_idx, col_range.start_idx),
                            (row_dim, col_dim),
                        ));
                        v *= -1.0;
                    }
                    s_builder.add_lower_block(
                        row_idx,
                        col_idx,
                        &neg_scratch.view((0, 0), (row_dim, col_dim)),
                    );
                }
            }
        }
    }

    // Only update the free part of rhs_bar (constraint part is unaffected by Schur elimination).
    rhs_bar
        .rows_mut(0, num_free_scalars)
        .axpy(-1.0, &result.r_delta, 1.0);
    Ok(())
}

// ── Range-space KKT data ──────────────────────────────────────────────────────

/// Pre-factored data for solving the KKT reduced system via the **range-space method**.
///
/// When equality constraints are present the reduced Schur system is:
/// ```text
/// [S_ff   G_f^T] [dx_f] = [rhs_bar_f]
/// [G_f    0    ] [dlam]   [rhs_c    ]
/// ```
///
/// Rather than factorizing this indefinite system directly, we use the range-space solve:
///
/// 1. Factor S_ff once (PD, BlockSparseLdlt).
/// 2. Compute `Y = S_ff⁻¹ G_fᵀ`  (`num_constraints` sparse back-solves, one per constraint).
/// 3. Build `M = G_f Y`  (`num_constraints × num_constraints` dense, PD when constraints are
///    independent).
/// 4. Factor `M` (trivially cheap for `num_constraints = 1, 2, 3`).
///
/// **Solve** (`O(nnz)` + `O(num_constraints³)`):
/// - `u  = S_ff⁻¹ rhs_bar_f`
/// - `dλ = M⁻¹ (G_f u − rhs_c)`
/// - `dx_f = S_ff⁻¹ (rhs_bar_f − G_fᵀ dλ)`
#[derive(Clone, Debug)]
pub struct KktRangeSpaceData {
    /// `G_f` dense matrix  (`num_constraints × nf`).
    pub g_f: DMatrix<f64>,
    /// `M⁻¹ = (G_f S_ff⁻¹ G_fᵀ)⁻¹`  (`num_constraints × num_constraints`).
    pub m_inv: DMatrix<f64>,
    /// Constraint RHS `g_all[nf+nm .. nf+nm+num_constraints]` (`-c` from the KKT formulation).
    pub rhs_c: DVector<f64>,
}

// ── SchurFactor ───────────────────────────────────────────────────────────────

/// Schur-complement factorization of a block-structured symmetric system.
///
/// Stores the factored Schur complement `S` (for solving the reduced system) together with
/// all data needed for back-substitution and block-covariance extraction from `H⁻¹`.
#[derive(Clone, Debug)]
pub struct SchurFactor {
    /// Factored S_ff for the reduced solve.
    s_factor: FactorEnum,
    /// S_ff matrix kept for lazy min-norm refactorization (covariance queries).
    s_block: BlockSparseSymmetricMatrix,
    /// Reduced RHS: `g_f − Σ H_mf^T H_mm⁻¹ g_m` (size `nf`).
    rhs_bar: DVector<f64>,
    /// Marg RHS: `g_m` (for back-substitution).
    g_marg: DVector<f64>,
    /// H_mm⁻¹ per marginalized block (in partition × block iteration order).
    pub h_mm_inv_cache: alloc::vec::Vec<DMatrix<f64>>,
    /// H_mf rows: for each marg block, the non-zero `(free_idx, H_mf block)` pairs.
    pub h_mf_rows: alloc::vec::Vec<alloc::vec::Vec<(PartitionBlockIndex, DMatrix<f64>)>>,
    /// Free-variable partition set (first `num_free_partitions` partitions).
    pub free_part_set: PartitionSet,
    /// Full partition set (free + marg).
    pub full_partitions: PartitionSet,
    /// Number of free partitions.
    pub num_free_partitions: usize,
    /// Number of constraint scalars (`num_constraints = 0` when there are no equality
    /// constraints).
    num_constraints: usize,
    /// Range-space KKT data (present only when `num_constraints > 0`).
    kkt: Option<KktRangeSpaceData>,
    /// Lazy min-norm factor for S_ff (BlockSparseLdlt-based, for covariance).
    s_min_norm: Option<InvertibleMatrix>,
    /// Lazy full S_ff⁻¹ (num_free_scalars × num_free_scalars), computed via SVD fallback.
    s_inv: Option<DMatrix<f64>>,
    /// Cached S_ff sparsity pattern for reuse in the next optimizer iteration.
    pub(crate) cached_s_pattern: Option<BlockSparseSymmetricMatrixPattern>,
}

impl SchurFactor {
    /// Build a `SchurFactor` from a partitioned Hessian.
    ///
    /// Performs the Schur complement forward pass:
    /// - Builds `S_ff = H_ff − Σ H_mf^T H_mm⁻¹ H_mf`
    /// - Factors `S_ff` with `solver`
    /// - Caches `H_mm⁻¹` and `H_mf` rows for back-substitution and covariance
    ///
    /// `num_free_partitions` is the number of *free* partitions (the first `num_free_partitions`
    /// partitions of `h`). `total_var_partitions` is `num_free_partitions + mpc` — the first
    /// `mpc` partitions after the free ones are marginalized.  Any remaining partitions
    /// `total_var_partitions..h.partitions().len()` are equality-constraint partitions, handled
    /// via the **range-space KKT method**: `G_f` is extracted as a dense `num_constraints × nf`
    /// matrix and the `num_constraints × num_constraints` Schur complement `M = G_f S_ff⁻¹
    /// G_fᵀ` is factored separately.  See [`KktRangeSpaceData`].
    pub fn factorize(
        h: &BlockSparseSymmetricMatrix,
        num_free_partitions: usize,
        total_var_partitions: usize,
        solver: LinearSolverEnum,
        cached_s_pattern: Option<BlockSparseSymmetricMatrixPattern>,
        g_all: &DVector<f64>,
        parallelize: bool,
    ) -> Result<Self, LinearSolverError> {
        let partitions = h.partitions();
        // Only marginalize partitions in num_free_partitions..total_var_partitions.
        let total_active = total_var_partitions;

        let num_free_scalars = (0..num_free_partitions)
            .map(|p| partitions.specs()[p].block_count * partitions.specs()[p].block_dim)
            .sum::<usize>();
        let num_marg_scalars = (num_free_partitions..total_active)
            .map(|p| partitions.specs()[p].block_count * partitions.specs()[p].block_dim)
            .sum::<usize>();
        let num_constraints = (total_var_partitions..partitions.len())
            .map(|p| partitions.specs()[p].block_count * partitions.specs()[p].block_dim)
            .sum::<usize>();

        let free_specs: alloc::vec::Vec<PartitionSpec> =
            partitions.specs()[..num_free_partitions].to_vec();
        let free_part_set = PartitionSet::new(free_specs);

        // ── Start with S_ff = H_ff ───────────────────────────────────────────
        // S_ff has only the free partitions; equality constraints are handled separately.
        let s_build_solver = LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default());
        let mut s_builder: SymmetricMatrixBuilderEnum = match cached_s_pattern {
            Some(pat) => SymmetricMatrixBuilderEnum::from_block_sparse_pattern(pat, s_build_solver),
            None => SymmetricMatrixBuilderEnum::BlockSparseLower(
                BlockSparseSymmetricMatrixBuilder::zero(free_part_set.clone()),
                s_build_solver,
            ),
        };

        for row_p in 0..num_free_partitions {
            for row_b in 0..partitions.specs()[row_p].block_count {
                let row_idx = PartitionBlockIndex {
                    partition: row_p,
                    block: row_b,
                };
                for col_p in 0..=row_p {
                    let col_b_end = if col_p < row_p {
                        partitions.specs()[col_p].block_count
                    } else {
                        row_b + 1
                    };
                    for col_b in 0..col_b_end {
                        let col_idx = PartitionBlockIndex {
                            partition: col_p,
                            block: col_b,
                        };
                        if let Some(view) = h.try_get_lower_block_view(row_idx, col_idx) {
                            s_builder.add_lower_block(row_idx, col_idx, &view);
                        } else {
                            let block = h.get_block(row_idx, col_idx);
                            s_builder.add_lower_block(row_idx, col_idx, &block.as_view());
                        }
                    }
                }
            }
        }

        let total_marg_blocks: usize = (num_free_partitions..total_active)
            .map(|p| partitions.specs()[p].block_count)
            .sum();

        // rhs_bar covers only the free part (size num_free_scalars); constraint RHS is in
        // kkt.rhs_c.
        let mut rhs_bar = g_all.rows(0, num_free_scalars).into_owned();
        let g_marg = g_all.rows(num_free_scalars, num_marg_scalars).into_owned();
        let mut h_mm_inv_cache: alloc::vec::Vec<DMatrix<f64>> =
            alloc::vec::Vec::with_capacity(total_marg_blocks);
        let mut h_mf_rows: alloc::vec::Vec<alloc::vec::Vec<(PartitionBlockIndex, DMatrix<f64>)>> =
            alloc::vec::Vec::with_capacity(total_marg_blocks);

        // ── Forward pass ─────────────────────────────────────────────────────
        #[cfg(not(target_arch = "wasm32"))]
        if parallelize {
            schur_forward_pass_parallel(
                h,
                partitions,
                &free_part_set,
                num_free_partitions,
                total_active,
                num_free_scalars,
                g_all,
                &mut rhs_bar,
                &mut s_builder,
                &mut h_mm_inv_cache,
                &mut h_mf_rows,
            )?;
        } else {
            schur_forward_pass_sequential(
                h,
                partitions,
                &free_part_set,
                num_free_partitions,
                total_active,
                g_all,
                &mut rhs_bar,
                &mut s_builder,
                &mut h_mm_inv_cache,
                &mut h_mf_rows,
            )?;
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = parallelize;
            schur_forward_pass_sequential(
                h,
                partitions,
                &free_part_set,
                num_free_partitions,
                total_active,
                g_all,
                &mut rhs_bar,
                &mut s_builder,
                &mut h_mm_inv_cache,
                &mut h_mf_rows,
            )?;
        }

        // ── Factor S_ff ──────────────────────────────────────────────────────
        let (s_sym, cached_s_pattern) = s_builder.build_with_pattern();
        let s_block = s_sym
            .as_block_sparse_lower()
            .expect("S builder always produces BlockSparseLower")
            .clone();
        let s_factor = solver.factorize(&s_sym)?;

        // ── Build KKT range-space data (only when equality constraints are present) ──
        // Collect G_f as a dense num_constraints × nf matrix from the H constraint×free blocks.
        // Then compute M = G_f S_ff⁻¹ G_fᵀ and factor M (tiny num_constraints × num_constraints
        // dense matrix).
        let kkt = if num_constraints > 0 {
            let rhs_c = g_all
                .rows(num_free_scalars + num_marg_scalars, num_constraints)
                .into_owned();

            // Build G_f dense (num_constraints × num_free_scalars).
            let mut g_f = DMatrix::<f64>::zeros(num_constraints, num_free_scalars);
            let mut constraint_row = 0usize;
            for cp in total_var_partitions..partitions.len() {
                let cp_bd = partitions.specs()[cp].block_dim;
                for cb in 0..partitions.specs()[cp].block_count {
                    let h_row = PartitionBlockIndex {
                        partition: cp,
                        block: cb,
                    };
                    for fp in 0..num_free_partitions {
                        for fb in 0..partitions.specs()[fp].block_count {
                            let col_idx = PartitionBlockIndex {
                                partition: fp,
                                block: fb,
                            };
                            if let Some(view) = h.try_get_lower_block_view(h_row, col_idx) {
                                let free_range = free_part_set.block_range(col_idx);
                                g_f.view_mut(
                                    (constraint_row, free_range.start_idx),
                                    (cp_bd, free_range.block_dim),
                                )
                                .copy_from(&view);
                            }
                        }
                    }
                    constraint_row += cp_bd;
                }
            }

            // Y = S_ff⁻¹ G_fᵀ  (num_free_scalars × num_constraints): solve num_constraints sparse
            // systems.
            let mut y = DMatrix::<f64>::zeros(num_free_scalars, num_constraints);
            for k in 0..num_constraints {
                let col = g_f.row(k).transpose().into_owned();
                let yk = s_factor.solve(&col)?;
                y.column_mut(k).copy_from(&yk);
            }

            // M = G_f Y  (num_constraints × num_constraints, should be PD when constraints are
            // independent).
            let m = &g_f * &y;

            // M⁻¹ via nalgebra (trivially cheap for num_constraints = 1, 2, 3).
            let m_inv = m
                .try_inverse()
                .ok_or_else(|| SingularKktConstraintSnafu { num_constraints }.build())?;

            Some(KktRangeSpaceData { g_f, m_inv, rhs_c })
        } else {
            None
        };

        Ok(SchurFactor {
            s_factor,
            s_block,
            rhs_bar,
            g_marg,
            h_mm_inv_cache,
            h_mf_rows,
            free_part_set,
            full_partitions: PartitionSet::new(partitions.specs()[..total_var_partitions].to_vec()),
            num_free_partitions,
            num_constraints,
            kkt,
            s_min_norm: None,
            s_inv: None,
            cached_s_pattern,
        })
    }

    /// Solve the reduced system and back-substitute to get the full step vector.
    ///
    /// Returns `[dx_f; dx_m; dlambda]` where:
    /// - `dx_f` (size `nf`): free-variable update
    /// - `dx_m` (size `nm`): marginalized-variable update via back-substitution
    /// - `dlambda` (size `num_constraints`): Lagrange-multiplier update (empty when no constraints)
    pub fn solve(&self) -> Result<DVector<f64>, LinearSolverError> {
        let num_free_scalars = self.free_part_set.scalar_dim();
        let num_marg_scalars = self.g_marg.len();
        let num_constraints = self.num_constraints;

        // Solve for dx_f (and dlambda when equality constraints are present).
        let (dx_f, dlambda) = if let Some(kkt) = &self.kkt {
            // Range-space KKT solve:
            //   u       = S_ff⁻¹ rhs_bar_f
            //   dλ      = M⁻¹ (G_f u − rhs_c)
            //   dx_f    = S_ff⁻¹ (rhs_bar_f − G_fᵀ dλ)
            let u = self.s_factor.solve(&self.rhs_bar)?;
            let w = &kkt.g_f * &u - &kkt.rhs_c;
            let dlambda = &kkt.m_inv * w;
            let rhs_corrected = &self.rhs_bar - kkt.g_f.transpose() * &dlambda;
            let dx_f = self.s_factor.solve(&rhs_corrected)?;
            (dx_f, dlambda)
        } else {
            // No equality constraints: plain S_ff solve.
            let dx_f = self.s_factor.solve(&self.rhs_bar)?;
            (dx_f, DVector::zeros(0))
        };

        let mut dx = DVector::<f64>::zeros(num_free_scalars + num_marg_scalars + num_constraints);
        dx.rows_mut(0, num_free_scalars).copy_from(&dx_f);

        // Back-substitute: dx_m[b] = H_mm⁻¹[b] (g_m[b] − Σ H_mf[b,f] dx_f[f])
        let mut marg_scalar = 0usize;
        for (b, non_zero_rows) in self.h_mf_rows.iter().enumerate() {
            let h_mm_inv = &self.h_mm_inv_cache[b];
            let d_m = h_mm_inv.nrows();

            let mut marg_vec = self.g_marg.rows(marg_scalar, d_m).into_owned();

            for (free_idx, h_mf_block) in non_zero_rows {
                let free_range = self.free_part_set.block_range(*free_idx);
                marg_vec.gemv(
                    -1.0,
                    h_mf_block,
                    &dx_f.rows(free_range.start_idx, free_range.block_dim),
                    1.0,
                );
            }

            dx.rows_mut(num_free_scalars + marg_scalar, d_m)
                .gemv(1.0, h_mm_inv, &marg_vec, 0.0);
            marg_scalar += d_m;
        }

        // Append lambda update at offset num_free_scalars + num_marg_scalars.
        if num_constraints > 0 {
            dx.rows_mut(num_free_scalars + num_marg_scalars, num_constraints)
                .copy_from(&dlambda);
        }

        Ok(dx)
    }

    /// Extract a block of `H⁻¹` (covariance matrix).
    ///
    /// Handles all combinations of free and marginalized partition indices:
    ///
    /// | row  | col  | formula                                                        |
    /// |------|------|----------------------------------------------------------------|
    /// | free | free | `S⁻¹[row, col]`                                                |
    /// | free | marg | `-(S⁻¹ H_fm H_mm⁻¹)[row, col]`                                 |
    /// | marg | free | transpose of free x marg                                       |
    /// | marg | marg | `H_mm⁻¹ δ_{b_i,b_j} + H_mm[i]⁻¹ H_mf_i S⁻¹ H_mf_j^T H_mm[j]⁻¹` |
    pub fn inverse_block(
        &mut self,
        row: PartitionBlockIndex,
        col: PartitionBlockIndex,
    ) -> DMatrix<f64> {
        let row_is_free = row.partition < self.num_free_partitions;
        let col_is_free = col.partition < self.num_free_partitions;

        match (row_is_free, col_is_free) {
            (true, true) => self.cov_free_free(row, col),
            (true, false) => self.cov_free_marg(row, col),
            (false, true) => self.cov_free_marg(col, row).transpose(),
            (false, false) => self.cov_marg_marg(row, col),
        }
    }

    /// Take the cached S sparsity pattern (for reuse in the next optimizer iteration).
    pub fn take_cached_s_pattern(&mut self) -> Option<BlockSparseSymmetricMatrixPattern> {
        self.cached_s_pattern.take()
    }

    // ── Private covariance helpers ────────────────────────────────────────────

    /// Ensure the min-norm factor for S is computed (lazy initialization).
    fn ensure_s_min_norm(&mut self) {
        if self.s_min_norm.is_none() {
            let s_sym = SymmetricMatrixEnum::from_block_sparse_lower(
                self.s_block.clone(),
                LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default()),
            );
            let factor = LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default())
                .factorize(&s_sym)
                .expect("S re-factorization for min-norm failed");
            self.s_min_norm = factor.into_invertible();
        }
    }

    /// Ensure the full S⁻¹ matrix is computed (lazy initialization).
    fn ensure_s_inv(&mut self) {
        if self.s_inv.is_none() {
            self.ensure_s_min_norm();
            self.s_inv = Some(self.s_min_norm.as_mut().unwrap().pseudo_inverse());
        }
    }

    fn cov_free_free(
        &mut self,
        row: PartitionBlockIndex,
        col: PartitionBlockIndex,
    ) -> DMatrix<f64> {
        self.ensure_s_min_norm();
        self.s_min_norm
            .as_mut()
            .unwrap()
            .pseudo_inverse_block(row, col)
    }

    fn cov_free_marg(
        &mut self,
        free_idx: PartitionBlockIndex,
        marg_idx: PartitionBlockIndex,
    ) -> DMatrix<f64> {
        self.ensure_s_inv();
        let b = self.marg_cache_idx(marg_idx);
        let d_m = self.h_mm_inv_cache[b].nrows();
        let num_free_scalars = self.free_part_set.scalar_dim();

        let h_mf_dense = self.build_h_mf_dense(b, d_m, num_free_scalars);

        let free_range = self.free_part_set.block_range(free_idx);
        let d_f = free_range.block_dim;

        // -(S⁻¹[free_rows, :] · H_mf_b^T) · H_mm[b]⁻¹  (d_f × d_m)
        let tmp = self
            .s_inv
            .as_ref()
            .unwrap()
            .rows(free_range.start_idx, d_f)
            .into_owned()
            * h_mf_dense.transpose();
        -tmp * &self.h_mm_inv_cache[b]
    }

    fn cov_marg_marg(
        &mut self,
        row_marg: PartitionBlockIndex,
        col_marg: PartitionBlockIndex,
    ) -> DMatrix<f64> {
        self.ensure_s_inv();
        let bi = self.marg_cache_idx(row_marg);
        let bj = self.marg_cache_idx(col_marg);
        let num_free_scalars = self.free_part_set.scalar_dim();

        let d_mi = self.h_mm_inv_cache[bi].nrows();
        let d_mj = self.h_mm_inv_cache[bj].nrows();

        let h_mf_i = self.build_h_mf_dense(bi, d_mi, num_free_scalars);
        let h_mf_j = self.build_h_mf_dense(bj, d_mj, num_free_scalars);

        // core = H_mf_i · S⁻¹ · H_mf_j^T  (d_mi × d_mj)
        let core = &h_mf_i * self.s_inv.as_ref().unwrap() * h_mf_j.transpose();

        // H_mm[i]⁻¹ · core · H_mm[j]⁻¹
        let result = &self.h_mm_inv_cache[bi] * core * &self.h_mm_inv_cache[bj];

        // Add H_mm[b]⁻¹ for the diagonal block (same variable).
        if bi == bj && row_marg.block == col_marg.block {
            &self.h_mm_inv_cache[bi] + result
        } else {
            result
        }
    }

    /// Linear index into `h_mm_inv_cache` / `h_mf_rows` for a marg partition block.
    fn marg_cache_idx(&self, marg_idx: PartitionBlockIndex) -> usize {
        let mut idx = 0;
        for p in self.num_free_partitions..marg_idx.partition {
            idx += self.full_partitions.specs()[p].block_count;
        }
        idx + marg_idx.block
    }

    /// Build the dense `d_m × nf` matrix H_mf[b] for marg-cache entry `b`.
    fn build_h_mf_dense(&self, b: usize, d_m: usize, num_free_scalars: usize) -> DMatrix<f64> {
        let mut h_mf_dense = DMatrix::<f64>::zeros(d_m, num_free_scalars);
        for (free_idx, block) in &self.h_mf_rows[b] {
            let range = self.free_part_set.block_range(*free_idx);
            h_mf_dense
                .columns_mut(range.start_idx, range.block_dim)
                .copy_from(block);
        }
        h_mf_dense
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{
        DMatrix,
        DVector,
    };

    use super::SchurFactor;
    use crate::{
        LinearSolverEnum,
        ldlt::BlockSparseLdlt,
        matrix::{
            IsSymmetricMatrixBuilder,
            PartitionBlockIndex,
            PartitionSet,
            PartitionSpec,
            block_sparse::BlockSparseSymmetricMatrixBuilder,
        },
    };

    /// Build a `BlockSparseSymmetricMatrix` from the lower triangle of a dense matrix.
    fn build_block_sparse_from_dense(
        partitions: PartitionSet,
        h_dense: &DMatrix<f64>,
    ) -> crate::matrix::block_sparse::BlockSparseSymmetricMatrix {
        let specs = partitions.specs().to_vec();
        let mut offsets = Vec::with_capacity(specs.len());
        let mut off = 0usize;
        for p in &specs {
            offsets.push(off);
            off += p.block_count * p.block_dim;
        }
        assert_eq!(off, h_dense.nrows());

        let mut builder = BlockSparseSymmetricMatrixBuilder::zero(partitions);

        for (pi_idx, pi) in specs.iter().enumerate() {
            let oi = offsets[pi_idx];
            for bi in 0..pi.block_count {
                let si = oi + bi * pi.block_dim;
                let row_idx = PartitionBlockIndex {
                    partition: pi_idx,
                    block: bi,
                };
                builder.add_lower_block(
                    row_idx,
                    row_idx,
                    &h_dense.view((si, si), (pi.block_dim, pi.block_dim)),
                );
                for bj in 0..bi {
                    let sj = oi + bj * pi.block_dim;
                    let col_idx = PartitionBlockIndex {
                        partition: pi_idx,
                        block: bj,
                    };
                    builder.add_lower_block(
                        row_idx,
                        col_idx,
                        &h_dense.view((si, sj), (pi.block_dim, pi.block_dim)),
                    );
                }
                for (pj_idx, pj) in specs.iter().enumerate().take(pi_idx) {
                    let oj = offsets[pj_idx];
                    for bj in 0..pj.block_count {
                        let sj = oj + bj * pj.block_dim;
                        let col_idx = PartitionBlockIndex {
                            partition: pj_idx,
                            block: bj,
                        };
                        builder.add_lower_block(
                            row_idx,
                            col_idx,
                            &h_dense.view((si, sj), (pi.block_dim, pj.block_dim)),
                        );
                    }
                }
            }
        }
        builder.build()
    }

    /// Build a diagonally-dominant SPD matrix compatible with Schur complement.
    ///
    /// H_mm (marg block) is block-diagonal with `marg_block_dim`-sized blocks, as
    /// the Schur complement code assumes.
    fn build_schur_spd_matrix(n_free: usize, n_marg: usize, marg_block_dim: usize) -> DMatrix<f64> {
        let n = n_free + n_marg;
        let mut h = DMatrix::<f64>::zeros(n, n);

        // H_ff: diagonally dominant with tridiagonal coupling
        for i in 0..n_free {
            h[(i, i)] = 10.0 + i as f64;
            if i + 1 < n_free {
                h[(i, i + 1)] = 1.0;
                h[(i + 1, i)] = 1.0;
            }
        }

        // H_mm: block-diagonal (each block is diagonally dominant)
        let num_marg_blocks = n_marg / marg_block_dim;
        for b in 0..num_marg_blocks {
            let off = n_free + b * marg_block_dim;
            for i in 0..marg_block_dim {
                h[(off + i, off + i)] = 10.0 + (off + i) as f64;
                if i + 1 < marg_block_dim {
                    h[(off + i, off + i + 1)] = 0.5;
                    h[(off + i + 1, off + i)] = 0.5;
                }
            }
        }

        // H_fm: small coupling between free and marg (preserves SPD via diagonal dominance)
        for i in 0..n_free.min(n_marg) {
            h[(i, n_free + i)] = 0.3;
            h[(n_free + i, i)] = 0.3;
        }

        h
    }

    /// Schur complement solve should match dense solve for an SPD system.
    ///
    /// Partitions: 2 free blocks (dim 2) + 3 marg blocks (dim 2) = 10×10 SPD.
    #[test]
    fn schur_vs_dense_equivalence() {
        let n_free = 4;
        let n_marg = 6;
        let n = n_free + n_marg;
        let h_dense = build_schur_spd_matrix(n_free, n_marg, 2);

        // Partition: [1×2, 1×2] free + [3×2] marg
        let parts = PartitionSet::new(vec![
            PartitionSpec {
                block_count: 1,
                block_dim: 2,
            eliminate_last: false,
                },
            PartitionSpec {
                block_count: 1,
                block_dim: 2,
            eliminate_last: false,
                },
            PartitionSpec {
                block_count: 3,
                block_dim: 2,
            eliminate_last: false,
                },
        ]);

        let g_all = DVector::<f64>::from_fn(n, |i, _| (i as f64 + 1.0) * 0.3);

        let h_block = build_block_sparse_from_dense(parts, &h_dense);

        let solver = LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default());
        let sf = SchurFactor::factorize(
            &h_block, 2, // num_free_partitions
            3, // total_var_partitions (2 free + 1 marg)
            solver, None, &g_all, false,
        )
        .unwrap();

        let dx_schur = sf.solve().unwrap();

        // Reference: dense solve H \ g
        let dx_dense = h_dense.try_inverse().unwrap() * &g_all;

        assert_relative_eq!(dx_schur, dx_dense, epsilon = 1e-9, max_relative = 1e-9);
    }

    /// KKT range-space method: verify G * dx_f = rhs_c (constraint satisfaction)
    /// and solution matches dense KKT solve.
    ///
    /// Layout: 2 free (dim 2 each) + 2 marg (dim 2 each) + 1 constraint (dim 1).
    #[test]
    fn kkt_range_space_constraint() {
        let n_free = 4;
        let n_marg = 4;
        let n_constr = 1;
        let n_var = n_free + n_marg;
        let n_total = n_var + n_constr;

        // Build SPD H for the variable block (marg is block-diagonal).
        let h_var = build_schur_spd_matrix(n_free, n_marg, 2);

        // Constraint row G_f linking to first two free scalars: x0 + x1 = c
        let g_row = DVector::<f64>::from_row_slice(&[1.0, 1.0, 0.0, 0.0]);

        // Build the full matrix including constraint partition.
        // The block-sparse matrix stores the constraint as lower-triangle blocks.
        let mut h_full = DMatrix::<f64>::zeros(n_total, n_total);
        h_full.view_mut((0, 0), (n_var, n_var)).copy_from(&h_var);
        // Constraint row/col: H[constraint, free] in lower triangle
        for j in 0..n_free {
            h_full[(n_var, j)] = g_row[j];
            h_full[(j, n_var)] = g_row[j];
        }

        // Partitions: [1×2, 1×2] free, [2×2] marg, [1×1] constraint
        let parts = PartitionSet::new(vec![
            PartitionSpec {
                block_count: 1,
                block_dim: 2,
            eliminate_last: false,
                },
            PartitionSpec {
                block_count: 1,
                block_dim: 2,
            eliminate_last: false,
                },
            PartitionSpec {
                block_count: 2,
                block_dim: 2,
            eliminate_last: false,
                },
            PartitionSpec {
                block_count: 1,
                block_dim: 1,
            eliminate_last: false,
                },
        ]);

        let g_all = DVector::<f64>::from_fn(n_total, |i, _| (i as f64 + 1.0) * 0.2);

        let h_block = build_block_sparse_from_dense(parts, &h_full);

        let solver = LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default());
        let sf = SchurFactor::factorize(
            &h_block, 2, // num_free_partitions
            3, // total_var_partitions (2 free + 1 marg)
            solver, None, &g_all, false,
        )
        .unwrap();

        let dx = sf.solve().unwrap();
        let dx_f = dx.rows(0, n_free).into_owned();

        // Verify constraint satisfaction: G_f * dx_f ≈ rhs_c
        let rhs_c = g_all[n_var]; // scalar constraint RHS
        let g_f_dense = DMatrix::<f64>::from_row_slice(1, n_free, &[1.0, 1.0, 0.0, 0.0]);
        let constraint_check = (&g_f_dense * &dx_f)[(0, 0)];
        assert_relative_eq!(constraint_check, rhs_c, epsilon = 1e-9, max_relative = 1e-9);

        // Verify against dense KKT solve: [H G^T; G 0]^{-1} [g_var; c]
        let mut kkt_mat = DMatrix::<f64>::zeros(n_total, n_total);
        kkt_mat.view_mut((0, 0), (n_var, n_var)).copy_from(&h_var);
        for j in 0..n_free {
            kkt_mat[(n_var, j)] = g_row[j];
            kkt_mat[(j, n_var)] = g_row[j];
        }
        let dx_ref = kkt_mat.try_inverse().unwrap() * &g_all;
        assert_relative_eq!(dx, dx_ref, epsilon = 1e-9, max_relative = 1e-9);
    }

    /// Parallel and sequential Schur forward passes should produce identical results.
    #[test]
    fn parallel_vs_sequential_schur() {
        let n_free = 4;
        let n_marg = 6;
        let n = n_free + n_marg;
        let h_dense = build_schur_spd_matrix(n_free, n_marg, 2);

        let parts = PartitionSet::new(vec![
            PartitionSpec {
                block_count: 1,
                block_dim: 2,
            eliminate_last: false,
                },
            PartitionSpec {
                block_count: 1,
                block_dim: 2,
            eliminate_last: false,
                },
            PartitionSpec {
                block_count: 3,
                block_dim: 2,
            eliminate_last: false,
                },
        ]);

        let g_all = DVector::<f64>::from_fn(n, |i, _| (i as f64 + 1.0) * 0.3);
        let solver = LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default());

        let h_block_seq = build_block_sparse_from_dense(parts.clone(), &h_dense);
        let sf_seq =
            SchurFactor::factorize(&h_block_seq, 2, 3, solver, None, &g_all, false).unwrap();
        let dx_seq = sf_seq.solve().unwrap();

        let h_block_par = build_block_sparse_from_dense(parts, &h_dense);
        let sf_par =
            SchurFactor::factorize(&h_block_par, 2, 3, solver, None, &g_all, true).unwrap();
        let dx_par = sf_par.solve().unwrap();

        assert_relative_eq!(dx_seq, dx_par, epsilon = 1e-12, max_relative = 1e-12);
    }
}
