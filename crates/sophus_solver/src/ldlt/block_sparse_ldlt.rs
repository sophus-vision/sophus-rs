use nalgebra::{
    DMatrix,
    DMatrixView,
    DVector,
};
use snafu::ResultExt;
use sophus_assert::assert_gt;

use crate::{
    BlockSparseLdltError,
    BlockSparseLdltSnafu,
    IsFactor,
    LinearSolverEnum,
    LinearSolverError,
    kernel::{
        sub_mat_diag_mat_t_inplace,
        sub_mat_plus_vec_inplace,
        sub_mat_transpose_plus_vec_in_place,
    },
    ldlt::{
        EliminationTree,
        IsLMatBuilder,
        IsLdltTracer,
        IsLdltWorkspace,
        NoopLdltTracer,
        block_diag_ldlt::BlockDiagonalLdltSystem,
    },
    matrix::{
        PartitionSet,
        PartitionSpec,
        SymmetricMatrixBuilderEnum,
        block::BlockColumn,
        block_sparse::{
            BlockMatrixSubdivision,
            BlockRegion,
            BlockSparseMatrix,
            BlockSparsePattern,
            BlockSparseSymmetricMatrixBuilder,
            NonZeroBlock,
            block_sparse_symmetric_matrix::BlockSparseSymmetricMatrix,
        },
        grid::Grid,
    },
    prelude::*,
};

/// Block sparse LDLᵀ.
#[derive(Copy, Clone, Debug)]
pub struct BlockSparseLdlt {
    tol_rel: f64,
}

impl Default for BlockSparseLdlt {
    fn default() -> Self {
        BlockSparseLdlt { tol_rel: 1e-12_f64 }
    }
}

impl IsLinearSolver for BlockSparseLdlt {
    type SymmetricMatrixBuilder = BlockSparseSymmetricMatrixBuilder;

    const NAME: &'static str = "block sparse LDLᵀ";

    fn name(&self) -> String {
        Self::NAME.into()
    }

    fn zero(&self, partitions: PartitionSet) -> SymmetricMatrixBuilderEnum {
        SymmetricMatrixBuilderEnum::BlockSparseLower(
            BlockSparseSymmetricMatrixBuilder::zero(partitions),
            LinearSolverEnum::BlockSparseLdlt(*self),
        )
    }

    type Factor = BlockSparseLdltFactor;

    fn factorize(
        &self,
        mat_a: &BlockSparseSymmetricMatrix,
    ) -> Result<Self::Factor, LinearSolverError> {
        let mut tracer = NoopLdltTracer::new();
        self.factorize_impl(mat_a, &mut tracer)
            .context(BlockSparseLdltSnafu)
    }

    /// Does not support parallel execution.
    fn set_parallelize(&mut self, _parallelize: bool) {
        // no-op
    }
}

impl IsFactor for BlockSparseLdltFactor {
    type Matrix = BlockSparseSymmetricMatrix;

    fn solve_inplace(
        &self,
        b_in_x_out: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        profile_scope!("ldlt solve");

        assert_eq!(b_in_x_out.len(), self.mat_l.subdivision.scalar_dim());

        // Permute b → b_perm (scalar_perm[new] = old).
        let mut b_perm = DVector::zeros(b_in_x_out.len());
        if let Some(ref sp) = self.scalar_perm {
            for (new_pos, &old_pos) in sp.iter().enumerate() {
                b_perm[new_pos] = b_in_x_out[old_pos];
            }
        } else {
            b_perm.copy_from(b_in_x_out);
        }

        let mat_l = &self.mat_l;
        let col_block_count = mat_l.subdivision.block_count();

        // Solve: L y = b_perm.
        for col_j in 0..col_block_count {
            let offset_col_j = mat_l.subdivision.scalar_offset(col_j);

            for e in mat_l.col(col_j).iter() {
                let row_i = e.global_block_row_idx;
                assert_gt!(row_i, col_j);

                let mat_l_ij = e.view;
                let scalar_offset_i = mat_l.subdivision.scalar_offset(row_i);

                let (head, tail) = b_perm.as_mut_slice().split_at_mut(scalar_offset_i);
                let b_j: &[f64] = &head[offset_col_j..offset_col_j + mat_l_ij.ncols()];
                let y_i: &mut [f64] = &mut tail[..mat_l_ij.nrows()];

                sub_mat_plus_vec_inplace(y_i, mat_l_ij, b_j);
            }
        }

        // Solve: D z = y.
        for col_j in 0..col_block_count {
            let offset_col_j = mat_l.subdivision.scalar_offset(col_j);
            let col_j_idx = mat_l.subdivision.idx(col_j);
            let block_width_j = mat_l.subdivision.block_dim(col_j_idx.partition);

            let z_j: nalgebra::Matrix<
                f64,
                nalgebra::Dyn,
                nalgebra::Const<1>,
                nalgebra::ViewStorageMut<
                    '_,
                    f64,
                    nalgebra::Dyn,
                    nalgebra::Const<1>,
                    nalgebra::Const<1>,
                    nalgebra::Dyn,
                >,
            > = b_perm.rows_mut(offset_col_j, block_width_j);
            self.block_diag.solve_inplace_vec(z_j, col_j_idx);
        }

        // Solve: Lᵀ x_perm = z.
        for col_j in (0..col_block_count).rev() {
            let offset_col_j = mat_l.subdivision.scalar_offset(col_j);

            for e in mat_l.col(col_j).iter() {
                let row_i = e.global_block_row_idx;
                assert_gt!(row_i, col_j);

                let mat_l_ij = e.view;
                let scalar_offset_i = mat_l.subdivision.scalar_offset(row_i);

                let (head, tail) = b_perm.as_mut_slice().split_at_mut(scalar_offset_i);
                let x_j: &mut [f64] = &mut head[offset_col_j..offset_col_j + mat_l_ij.ncols()];
                let z_i: &[f64] = &tail[..mat_l_ij.nrows()];

                sub_mat_transpose_plus_vec_in_place(x_j, mat_l_ij, z_i);
            }
        }

        // Unpermute x_perm → x (scalar_perm[new] = old).
        if let Some(ref sp) = self.scalar_perm {
            for (new_pos, &old_pos) in sp.iter().enumerate() {
                b_in_x_out[old_pos] = b_perm[new_pos];
            }
        } else {
            b_in_x_out.copy_from(&b_perm);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AMD + symbolic helpers
// ---------------------------------------------------------------------------

/// Run block-level AMD on the lower block-sparse matrix.
///
/// Returns `(perm, perm_inv)` where `perm[new_block] = old_block`.
fn block_amd_ordering(a_lower: &BlockSparseMatrix) -> (Vec<usize>, Vec<usize>) {
    let nb = a_lower.block_count();
    let upper = a_lower.build_pattern_upper();

    // SAFETY: `upper` was built by `build_pattern_upper` which guarantees valid CSC.
    let symbolic = unsafe {
        faer::sparse::SymbolicSparseColMat::<usize>::new_unchecked(
            nb,
            nb,
            upper.storage_idx_by_col().to_vec(),
            None,
            upper.row_idx_storage().to_vec(),
        )
    };

    crate::ldlt::amd_order(symbolic.as_ref())
}

/// Build a flat `BlockMatrixSubdivision` (one block per partition) for the
/// AMD-permuted matrix.
fn make_flat_subdivision(
    orig: &BlockMatrixSubdivision,
    block_perm: &[usize],
) -> BlockMatrixSubdivision {
    let nb = block_perm.len();

    // Each permuted block k gets its own partition with block_count=1.
    let partition_specs: Vec<PartitionSpec> = block_perm
        .iter()
        .map(|&old_k| {
            let old_part = orig.idx(old_k).partition;
            PartitionSpec {
                block_count: 1,
                block_dim: orig.block_dim(old_part),
            }
        })
        .collect();

    let flat_partitions = PartitionSet::new(partition_specs);

    // In the flat subdivision each partition has exactly one block.
    let scalar_offset_by_block: Vec<usize> = {
        let mut offsets = Vec::with_capacity(nb + 1);
        let mut off = 0usize;
        for &old_k in block_perm {
            offsets.push(off);
            let old_part = orig.idx(old_k).partition;
            off += orig.block_dim(old_part);
        }
        offsets.push(off);
        offsets
    };

    // partition_idx_by_block[k] = k  (flat)
    let partition_idx_by_block: Vec<u16> = (0..nb as u16).collect();

    // block_offset_by_partition[k] = k  (each partition has 1 block)
    let block_offset_by_partition: Vec<usize> = (0..=nb).collect();

    BlockMatrixSubdivision::new(
        scalar_offset_by_block,
        partition_idx_by_block,
        block_offset_by_partition,
        flat_partitions,
    )
}

/// Permute the lower block-sparse matrix: entry `(old_i, old_j)` → `(perm_inv[old_i],
/// perm_inv[old_j])`.
///
/// Returns a new `BlockSparseMatrix` in the flat subdivision.
fn permute_block_sparse_lower(
    a_lower: &BlockSparseMatrix,
    block_perm_inv: &[usize],
    flat_sub: &BlockMatrixSubdivision,
) -> BlockSparseMatrix {
    let nb = a_lower.block_count();

    // Collect (new_col, new_row, block_data) triples, then sort by col.
    let mut entries: Vec<(usize, usize, Vec<f64>)> = Vec::new();

    for old_col in 0..nb {
        for entry in a_lower.col(old_col).iter() {
            let old_row = entry.global_block_row_idx;
            let new_col_raw = block_perm_inv[old_col];
            let new_row_raw = block_perm_inv[old_row];

            // Store in lower triangle: final_row >= final_col.
            // If the permuted position is in the upper triangle, transpose the block.
            let (final_row, final_col, transposed) = if new_row_raw >= new_col_raw {
                (new_row_raw, new_col_raw, false)
            } else {
                (new_col_raw, new_row_raw, true)
            };

            let (h, w) = if transposed {
                (entry.view.ncols(), entry.view.nrows())
            } else {
                (entry.view.nrows(), entry.view.ncols())
            };

            let mut data = vec![0.0f64; h * w];
            if transposed {
                // Store blockᵀ in column-major: col c of transposed = row c of original.
                for c in 0..w {
                    let orig_row = entry.view.row(c);
                    for r in 0..h {
                        data[c * h + r] = orig_row[r];
                    }
                }
            } else {
                // Column-major copy.
                for c in 0..w {
                    let col = entry.view.column(c);
                    data[c * h..(c + 1) * h].copy_from_slice(col.as_slice());
                }
            }
            entries.push((final_col, final_row, data));
        }
    }

    // Sort by (new_col ASC, new_row ASC).
    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    // Build region storage: in flat subdivision, region (new_row, new_col) has exactly one block.
    let mut regions = Grid::new(
        [nb, nb],
        BlockRegion {
            storage: Vec::new(),
        },
    );
    let mut block_columns: Vec<Vec<NonZeroBlock>> = vec![Vec::new(); nb];

    for (new_col, new_row, data) in &entries {
        let new_col = *new_col;
        let new_row = *new_row;

        // In flat subdivision, region = (new_row_partition, new_col_partition) = (new_row,
        // new_col).
        let new_col_part = flat_sub.idx(new_col).partition;
        let new_row_part = flat_sub.idx(new_row).partition;
        let region = regions.get_mut(&[new_row_part, new_col_part]);
        let storage_base = region.storage.len() as u32;
        region.storage.extend_from_slice(data);

        block_columns[new_col].push(NonZeroBlock {
            block_row_idx: new_row as u32,
            storage_base,
        });
    }

    // Build CSC pattern.
    let mut nonzero_idx_by_block_col = Vec::with_capacity(nb + 1);
    nonzero_idx_by_block_col.push(0);
    for col in 0..nb {
        nonzero_idx_by_block_col.push(nonzero_idx_by_block_col[col] + block_columns[col].len());
    }
    let total_nz = nonzero_idx_by_block_col[nb];
    let mut nonzero_blocks = Vec::with_capacity(total_nz);
    for col in &block_columns {
        nonzero_blocks.extend_from_slice(col);
    }

    BlockSparseMatrix {
        block_col_pattern: BlockSparsePattern {
            nonzero_idx_by_block_col,
            nonzero_blocks,
        },
        subdivision: flat_sub.clone(),
        regions,
    }
}

/// Symbolic LDLᵀ factorization: compute the exact sparsity of L without floats.
///
/// Returns `(l_col_rows, l_region_float_counts)`:
/// - `l_col_rows[j]` — sorted block-row indices > j that appear in column j of L
/// - `l_region_nf[region]` — total float count for each region
fn symbolic_factorize_l(
    a_lower_perm: &BlockSparseMatrix,
    flat_sub: &BlockMatrixSubdivision,
) -> (Vec<Vec<usize>>, Grid<usize>) {
    let nb = a_lower_perm.block_count();
    let mut etree = EliminationTree::new(a_lower_perm.build_pattern_upper());

    let mut was_touched = vec![false; nb];
    let mut touched_rows: Vec<usize> = Vec::with_capacity(nb);
    let mut l_col_rows: Vec<Vec<usize>> = vec![Vec::new(); nb];

    for col_j in 0..nb {
        // Rows from A[:,j] (strictly below diagonal).
        for entry in a_lower_perm.col(col_j).iter() {
            let row_i = entry.global_block_row_idx;
            if row_i > col_j && !was_touched[row_i] {
                was_touched[row_i] = true;
                touched_rows.push(row_i);
            }
        }

        // Fill from previous L columns in reach.
        let reach = etree.reach(col_j).to_vec(); // clone to release borrow on etree
        for col_k in reach {
            for &row_i in &l_col_rows[col_k] {
                if row_i > col_j && !was_touched[row_i] {
                    was_touched[row_i] = true;
                    touched_rows.push(row_i);
                }
            }
        }

        touched_rows.sort_unstable();
        l_col_rows[col_j] = touched_rows.clone();

        for &r in &touched_rows {
            was_touched[r] = false;
        }
        touched_rows.clear();
    }

    // Per-region float counts.
    let mut l_region_nf = Grid::new([nb, nb], 0usize);
    for col_j in 0..nb {
        let col_dim = flat_sub.block_dim(col_j); // partition == block in flat sub
        for &row_i in &l_col_rows[col_j] {
            let row_dim = flat_sub.block_dim(row_i);
            *l_region_nf.get_mut(&[row_i, col_j]) += row_dim * col_dim;
        }
    }

    (l_col_rows, l_region_nf)
}

/// Scalar permutation from flat (permuted) ordering back to original ordering.
///
/// `scalar_perm[new_scalar_pos] = old_scalar_pos`.
fn make_scalar_perm(
    orig_sub: &BlockMatrixSubdivision,
    block_perm: &[usize],
    flat_sub: &BlockMatrixSubdivision,
) -> Vec<usize> {
    let total = orig_sub.scalar_dim();
    let mut scalar_perm = vec![0usize; total];

    for (new_k, &old_k) in block_perm.iter().enumerate() {
        let new_scalar_start = flat_sub.scalar_offset(new_k);
        let old_scalar_start = orig_sub.scalar_offset(old_k);
        let block_d = flat_sub.block_dim(new_k);
        for s in 0..block_d {
            scalar_perm[new_scalar_start + s] = old_scalar_start + s;
        }
    }

    scalar_perm
}

// ---------------------------------------------------------------------------
// BlockSparseLdlt factorization
// ---------------------------------------------------------------------------

impl BlockSparseLdlt {
    /// Factorize `a_lower` (lower triangle of `A`) into `L` and `D`.
    pub fn factorize_impl(
        &self,
        a_lower: &BlockSparseSymmetricMatrix,
        tracer: &mut impl IsLdltTracer<BlockLdltWorkspace>,
    ) -> Result<BlockSparseLdltFactor, BlockSparseLdltError> {
        profile_scope!("ldlt fact");

        // --- AMD ordering (block level) ------------------------------------
        let (block_perm, block_perm_inv) = {
            profile_scope!("block_amd");
            block_amd_ordering(&a_lower.lower)
        };

        // --- Flat subdivision + permuted matrix ----------------------------
        let (flat_sub, a_lower_perm) = {
            profile_scope!("permute");
            let flat_sub = make_flat_subdivision(a_lower.subdivision(), &block_perm);
            let a_lower_perm =
                permute_block_sparse_lower(&a_lower.lower, &block_perm_inv, &flat_sub);
            (flat_sub, a_lower_perm)
        };

        // --- Symbolic factorization (L sparsity) ---------------------------
        let (l_col_rows, l_region_nf) = {
            profile_scope!("symbolic");
            symbolic_factorize_l(&a_lower_perm, &flat_sub)
        };

        // --- Scalar permutation for solve ----------------------------------
        let scalar_perm = make_scalar_perm(a_lower.subdivision(), &block_perm, &flat_sub);

        // --- Numeric LDLᵀ on permuted matrix ------------------------------
        let nb = a_lower_perm.block_count();
        let mut etree = BlockLdltWorkspace::calc_etree(&a_lower_perm);
        let mut ws = BlockLdltWorkspace::new(nb, &flat_sub);
        let mut mat_l =
            BlockSparseLFactorBuilder::with_symbolic(&flat_sub, &l_col_rows, &l_region_nf);
        let mut diag_d = BlockDiagonalLdltSystem::zero(flat_sub.partitions().specs(), self.tol_rel);

        for col_j in 0..nb {
            ws.activate_col(col_j);
            ws.load_column(&a_lower_perm);
            let reach = etree.reach(col_j);
            for &col_k in reach {
                ws.apply_to_col_k_in_reach(col_k, &mat_l, &diag_d, tracer);
            }
            ws.append_to_ldlt(&mut mat_l, &mut diag_d)?;
            ws.clear();
        }

        Ok(BlockSparseLdltFactor {
            mat_l: mat_l.compress(),
            block_diag: diag_d,
            scalar_perm: Some(scalar_perm),
            _original_partitions: a_lower.subdivision().partitions().clone(),
        })
    }
}

/// Factorization product `A = L D Lᵀ`.
#[derive(Clone, Debug)]
pub struct BlockSparseLdltFactor {
    /// Off-diagonal lower block of L (in AMD-permuted flat subdivision).
    pub mat_l: BlockSparseMatrix,
    /// Diagonal blocks: L[j,j] d[j].
    pub(crate) block_diag: BlockDiagonalLdltSystem,
    /// Scalar permutation: `scalar_perm[new_pos] = old_pos`.
    /// `None` if no reordering was applied (identity).
    pub(crate) scalar_perm: Option<Vec<usize>>,
    /// Original (pre-AMD) partition set — used for external block-range queries.
    pub(crate) _original_partitions: PartitionSet,
}

/// LDLᵀ workspace
#[derive(Debug)]
pub struct BlockLdltWorkspace {
    // Active column j.
    col_j: usize,
    // Height of blocks in column j.
    block_height_col_j: usize,
    // Mark per row whether it was touched.
    was_row_touched: Vec<bool>,
    // List of touched rows for the current column `j`.
    touched_rows: Vec<usize>,
    // Q[j,k] = L[j,k] * L[k,k].
    mat_q_jk: nalgebra::DMatrix<f64>,
    // Q[i,k] = L[i,k] * L[k,k].
    mat_q_ik: nalgebra::DMatrix<f64>,
    // Column accumulator C(:,j), C(i,j) = ∑ᵢ A[i,j] - ∑ₖ Q[i,k] * diag(d[k]) * Q[j,k]ᵀ.
    mat_c: BlockColumn,
    // The block-matrix subdivision structure.
    mat_subdivision: BlockMatrixSubdivision,
}

impl IsLdltWorkspace for BlockLdltWorkspace {
    type Error = BlockSparseLdltError;

    type Matrix = BlockSparseMatrix;
    type Diag = BlockDiagonalLdltSystem;

    type MatrixEntry = DMatrix<f64>;
    type DiagnalEntry = (DMatrix<f64>, DVector<f64>);

    type MatLBuilder = BlockSparseLFactorBuilder;

    fn calc_etree(a_lower: &Self::Matrix) -> EliminationTree {
        EliminationTree::new(a_lower.build_pattern_upper())
    }

    fn activate_col(&mut self, col_j: usize) {
        let block_dim = self
            .mat_subdivision
            .block_dim(self.mat_subdivision.idx(col_j).partition);

        self.mat_c.set_active_width(block_dim);
        self.col_j = col_j;
        self.block_height_col_j = block_dim;
    }

    #[inline(always)]
    fn load_column(&mut self, a_lower: &BlockSparseMatrix) {
        for entry in a_lower.col(self.col_j).iter() {
            let row_i = entry.global_block_row_idx;
            let mat_a_ij = entry.view;

            let row_idx_i = a_lower.subdivision.idx(row_i);
            self.mat_c.add_block(row_idx_i, mat_a_ij);

            if !self.was_row_touched[row_i] {
                self.was_row_touched[row_i] = true;
                self.touched_rows.push(row_i);
            }
        }
    }

    fn apply_to_col_k_in_reach(
        &mut self,
        col_k: usize,
        l_mat_builder: &Self::MatLBuilder,
        diag: &Self::Diag,
        _tracer: &mut impl IsLdltTracer<Self>,
    ) {
        // Get L[j,k] if present.
        let Some(mat_l_jk) = l_mat_builder.get_block(self.col_j, col_k) else {
            return;
        };

        let col_k_idx = self.mat_subdivision.idx(col_k);
        let block_width_col_k = self.mat_subdivision.block_dim(col_k_idx.partition);

        // Block diagonal LDLᵀ factors: L[k,k] and diag(d[k]).
        let mat_l_kk = diag.mat_l.get_block(col_k_idx);
        let diag_d_k = diag.d.get_block(col_k_idx);

        // Q[j,k] = L[j,k] * L[k,k].
        let mut mat_q_jk = self
            .mat_q_jk
            .view_range_mut(0..self.block_height_col_j, 0..block_width_col_k);
        mat_q_jk.gemm(1.0, &mat_l_jk, &mat_l_kk, 0.0);

        // C[j,j] -= Q[j,k] * Diag(d[k]) * Q[j,k]ᵀ
        let col_j_idx = self.mat_subdivision.idx(self.col_j);
        let mut mat_c_jj = self.mat_c.get_block_mut(col_j_idx);
        sub_mat_diag_mat_t_inplace(
            &mut mat_c_jj,
            mat_q_jk.as_view(),
            diag_d_k.as_view(),
            mat_q_jk.as_view(),
        );

        let column = &l_mat_builder.columns[col_k];
        for block in column.non_zero_blocks.iter() {
            let row_i = block.block_row_idx as usize;
            let storage_offset_of_block_ik = block.storage_base as usize;

            if row_i <= self.col_j {
                continue;
            }
            let row_idx_i = self.mat_subdivision.idx(row_i);

            let block_height_row_i = self.mat_subdivision.block_dim(row_idx_i.partition);
            let size_of_block_ik = block_height_row_i * block_width_col_k;
            let region_of_block_ik = l_mat_builder
                .regions
                .get(&[row_idx_i.partition, col_k_idx.partition]);
            debug_assert!(
                storage_offset_of_block_ik + size_of_block_ik <= region_of_block_ik.storage.len()
            );
            let data_of_block_ik = &region_of_block_ik.storage
                [storage_offset_of_block_ik..storage_offset_of_block_ik + size_of_block_ik];

            if !self.was_row_touched[row_i] {
                self.was_row_touched[row_i] = true;
                self.touched_rows.push(row_i);
            }

            // Q[i,k] = L[i,k] * L[k,k].
            let mut mat_q_ik = self
                .mat_q_ik
                .view_range_mut(0..block_height_row_i, 0..block_width_col_k);
            let mat_l_ik: nalgebra::Matrix<
                f64,
                nalgebra::Dyn,
                nalgebra::Dyn,
                nalgebra::ViewStorage<
                    '_,
                    f64,
                    nalgebra::Dyn,
                    nalgebra::Dyn,
                    nalgebra::Const<1>,
                    nalgebra::Dyn,
                >,
            > = DMatrixView::from_slice(data_of_block_ik, block_height_row_i, block_width_col_k);
            mat_q_ik.gemm(1.0, &mat_l_ik, &mat_l_kk, 0.0);

            // C(i,j) -= Q[i,k] * diag(d[k]) * Q[j,k]ᵀ
            let mut mat_c_ij = self.mat_c.get_block_mut(row_idx_i);
            sub_mat_diag_mat_t_inplace(
                &mut mat_c_ij,
                mat_q_ik.as_view(),
                diag_d_k.as_view(),
                mat_q_jk.as_view(),
            );

            #[cfg(debug_assertions)]
            {
                use crate::ldlt::LdltIndices;

                _tracer.after_update(
                    LdltIndices {
                        row_i,
                        col_j: self.col_j,
                        col_k,
                    },
                    (mat_l_kk.into_owned(), diag_d_k.into_owned()),
                    mat_l_ik.into_owned(),
                    mat_l_jk.into_owned(),
                    mat_c_ij.into_owned(),
                );
            }
        }
    }

    fn append_to_ldlt(
        &mut self,
        l_mat_builder: &mut Self::MatLBuilder,
        diag: &mut Self::Diag,
    ) -> Result<(), BlockSparseLdltError> {
        let col_j_idx = self.mat_subdivision.idx(self.col_j);

        let mat_a_jj = self.mat_c.get_block(col_j_idx);
        diag.decompose(col_j_idx, mat_a_jj.as_view())?;

        self.touched_rows.sort_unstable();
        for &row_i in &self.touched_rows {
            if row_i <= self.col_j {
                continue;
            }

            let row_idx_i = self.mat_subdivision.idx(row_i);
            let mut mat_c_ij = self.mat_c.get_block_mut(row_idx_i);

            diag.right_solve_inplace(mat_c_ij.as_view_mut(), col_j_idx);

            l_mat_builder.append_to(row_i, self.col_j, mat_c_ij.as_view());
        }

        Ok(())
    }

    #[inline(always)]
    fn clear(&mut self) {
        for row_i in self.touched_rows.drain(..) {
            if self.was_row_touched[row_i] {
                let row_idx_i = self.mat_subdivision.idx(row_i);
                self.mat_c.zero_block(row_idx_i);
                self.was_row_touched[row_i] = false;
            }
        }
    }
}

impl BlockLdltWorkspace {
    /// Create workspace for N x N matrix.
    pub fn new(n: usize, mat_subdivision: &BlockMatrixSubdivision) -> Self {
        let max_block_dim = mat_subdivision.max_block_dim();
        let mat_q_jk = nalgebra::DMatrix::<f64>::zeros(max_block_dim, max_block_dim);
        let mat_q_ik = nalgebra::DMatrix::<f64>::zeros(max_block_dim, max_block_dim);
        Self {
            mat_c: BlockColumn::zero(mat_subdivision.partitions().specs(), max_block_dim),
            was_row_touched: vec![false; n],
            touched_rows: Vec::with_capacity(n),
            mat_q_ik,
            mat_q_jk,
            col_j: 0,
            block_height_col_j: 0,
            mat_subdivision: mat_subdivision.clone(),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct BlockSparseLFactorColumn {
    non_zero_blocks: Vec<NonZeroBlock>,
}

/// Block-sparse factor L of LDLᵀ decomposition.
#[derive(Debug)]
pub struct BlockSparseLFactorBuilder {
    columns: Vec<BlockSparseLFactorColumn>,
    regions: Grid<BlockRegion>,
    mat_subdivision: BlockMatrixSubdivision,
}

impl IsLMatBuilder for BlockSparseLFactorBuilder {
    type Matrix = BlockSparseMatrix;

    fn compress(self) -> BlockSparseMatrix {
        let block_col_count = self.mat_subdivision.block_count();

        let mut nonzero_idx_by_block_col = Vec::with_capacity(block_col_count + 1);
        nonzero_idx_by_block_col.push(0);
        for col_j in 0..block_col_count {
            nonzero_idx_by_block_col
                .push(nonzero_idx_by_block_col[col_j] + self.columns[col_j].non_zero_blocks.len());
        }
        let nonzero_block_count = nonzero_idx_by_block_col[block_col_count];

        let mut nonzero_blocks = vec![
            NonZeroBlock {
                block_row_idx: 0,
                storage_base: 0
            };
            nonzero_block_count
        ];
        for col_j in 0..block_col_count {
            let start = nonzero_idx_by_block_col[col_j];

            for (i, block) in self.columns[col_j].non_zero_blocks.iter().enumerate() {
                nonzero_blocks[start + i] = *block;
            }
        }

        BlockSparseMatrix {
            block_col_pattern: BlockSparsePattern {
                nonzero_idx_by_block_col,
                nonzero_blocks,
            },
            subdivision: self.mat_subdivision,
            regions: self.regions,
        }
    }
}

impl BlockSparseLFactorBuilder {
    /// Create a pre-allocated block-sparse factor L from symbolic analysis.
    ///
    /// Pre-reserves exact storage for all regions and columns based on the
    /// symbolic factorization result, eliminating reallocation during numeric pass.
    pub fn with_symbolic(
        flat_sub: &BlockMatrixSubdivision,
        l_col_rows: &[Vec<usize>],
        l_region_nf: &Grid<usize>,
    ) -> Self {
        let nb = flat_sub.block_count();

        let columns: Vec<BlockSparseLFactorColumn> = (0..nb)
            .map(|j| BlockSparseLFactorColumn {
                non_zero_blocks: Vec::with_capacity(l_col_rows[j].len()),
            })
            .collect();

        // Pre-allocate region storage.
        let mut regions = Grid::new(
            [nb, nb],
            BlockRegion {
                storage: Vec::new(),
            },
        );
        for row_part in 0..nb {
            for col_part in 0..nb {
                let nf = *l_region_nf.get(&[row_part, col_part]);
                if nf > 0 {
                    regions.get_mut(&[row_part, col_part]).storage = vec![0.0f64; nf];
                    // Reset to 0 length so append_to can extend_from_slice normally.
                    regions.get_mut(&[row_part, col_part]).storage.clear();
                    regions
                        .get_mut(&[row_part, col_part])
                        .storage
                        .reserve_exact(nf);
                }
            }
        }

        Self {
            columns,
            regions,
            mat_subdivision: flat_sub.clone(),
        }
    }

    /// Append `L[i,j]` (with i>j).
    ///
    /// Precondition:
    ///   * Calls for the same column `j` arrive with ascending `i`.
    pub fn append_to(&mut self, row_i: usize, col_j: usize, mat_l_ij: DMatrixView<f64>) {
        debug_assert!(row_i > col_j, "strictly lower only");

        let row_idx = self.mat_subdivision.idx(row_i);
        let col_idx = self.mat_subdivision.idx(col_j);

        let region_ij = self
            .regions
            .get_mut(&[row_idx.partition, col_idx.partition]);
        let start = region_ij.storage.len();

        for c in 0..mat_l_ij.ncols() {
            region_ij
                .storage
                .extend_from_slice(mat_l_ij.column(c).as_slice());
        }

        self.columns[col_j].non_zero_blocks.push(NonZeroBlock {
            block_row_idx: row_i as u32,
            storage_base: start as u32,
        });
    }

    /// Lookup `L[j,k]` in column k, returning a view, or None if absent.
    pub fn get_block<'a>(&'a self, row_j: usize, col_k: usize) -> Option<DMatrixView<'a, f64>> {
        let column = &self.columns[col_k];
        let storage_idx = column
            .non_zero_blocks
            .binary_search_by(|block| block.block_row_idx.cmp(&(row_j as u32)))
            .ok()?;
        let base = column.non_zero_blocks[storage_idx].storage_base as usize;

        let row_idx = self.mat_subdivision.idx(row_j);
        let col_idx = self.mat_subdivision.idx(col_k);

        let row_partition_idx = row_idx.partition;
        let col_partition_idx = col_idx.partition;
        let block_height = self.mat_subdivision.block_dim(row_partition_idx);
        let block_width = self.mat_subdivision.block_dim(col_partition_idx);

        let block_area = block_height * block_width;
        let region = self.regions.get(&[row_partition_idx, col_partition_idx]);
        debug_assert!(base + block_area <= region.storage.len());

        Some(DMatrixView::from_slice(
            &region.storage[base..base + block_area],
            block_height,
            block_width,
        ))
    }
}
