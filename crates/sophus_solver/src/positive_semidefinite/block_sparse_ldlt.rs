use nalgebra::{
    DMatrix,
    DMatrixView,
    DVector,
};
use snafu::ResultExt;

use crate::{
    BlockSparseLdltError,
    BlockSparseLdltSnafu,
    IsLinearSolver,
    LinearSolverError,
    assert_gt,
    kernel::{
        sub_mat_diag_mat_t_in_place,
        sub_mat_plus_vec_in_place,
        sub_mat_transpose_plus_vec_in_place,
    },
    matrix::{
        BlockColCompressedMatrix,
        BlockColCompressedPattern,
        BlockColumn,
        BlockMatrixSubdivision,
        BlockRegion,
        BlockSparseTripletMatrix,
        IsCompressibleMatrix,
        LowerBlockSparseMatrixBuilder,
        NonZeroBlock,
        PartitionSpec,
        SymmetricMatrixBuilderEnum,
        grid::Grid,
    },
    positive_semidefinite::{
        EliminationTree,
        IsLMatBuilder,
        IsLdltTracer,
        IsLdltWorkspace,
        NoopLdltTracer,
        block_diag_ldlt::BlockDiagonalLdltFactors,
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
    type Matrix = BlockSparseTripletMatrix;

    const NAME: &'static str = "block sparse LDLᵀ";

    fn solve_in_place(
        &self,
        _parallelize: bool,
        a_lower: &BlockColCompressedMatrix,
        b_in_x_out: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        let mut tracer = NoopLdltTracer::new();

        let ldlt = self
            .factorize(a_lower, &mut tracer)
            .context(BlockSparseLdltSnafu)?;

        // Solve (L D Lᵀ) x = b.
        ldlt.solve_in_place(b_in_x_out);
        Ok(())
    }

    fn name(&self) -> String {
        Self::NAME.into()
    }

    fn matrix_builder(&self, partitions: &[PartitionSpec]) -> SymmetricMatrixBuilderEnum {
        SymmetricMatrixBuilderEnum::BlockSparseLower(LowerBlockSparseMatrixBuilder::zero(
            partitions,
        ))
    }

    fn solve(
        &self,
        parallelize: bool,
        matrix: &<Self::Matrix as IsCompressibleMatrix>::Compressed,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, crate::LinearSolverError> {
        let mut x = b.clone();
        self.solve_in_place(parallelize, matrix, &mut x)?;
        Ok(x)
    }
}

impl BlockSparseLdlt {
    /// Factorize `a_lower` (lower triangle of `A`) into `L` and `D`.
    pub fn factorize(
        &self,
        a_lower: &BlockColCompressedMatrix,
        tracer: &mut impl IsLdltTracer<BlockLdltWorkspace>,
    ) -> Result<BlockLdltFactor, BlockSparseLdltError> {
        puffin::profile_scope!("ldlt fact");

        let mut etree = BlockLdltWorkspace::calc_etree(a_lower);
        let nb = a_lower.subdivision.block_col_count();

        let mut ws = BlockLdltWorkspace::new(nb, &a_lower.subdivision);

        let mut mat_l = BlockSparseLFactorBuilder::new(&a_lower.subdivision);
        let mut diag_d =
            BlockDiagonalLdltFactors::zero(a_lower.subdivision.row_partitons(), self.tol_rel);

        for col_j in 0..nb {
            ws.activate_col(col_j);

            ws.load_column(a_lower);

            let reach = etree.reach(col_j);
            for &col_k in reach {
                ws.apply_to_col_k_in_reach(col_k, &mat_l, &diag_d, tracer);
            }

            ws.append_to_ldlt(&mut mat_l, &mut diag_d)?;

            ws.clear();
        }

        Ok(BlockLdltFactor {
            mat_l: mat_l.compress(),
            block_diag: diag_d,
        })
    }
}

/// Factorization product `A = L D Lᵀ`.
#[derive(Debug)]
pub struct BlockLdltFactor {
    /// Off-diagonal lower block of L.
    pub mat_l: BlockColCompressedMatrix,
    /// Diagonal blocks: L[j,j] d[j].
    pub(crate) block_diag: BlockDiagonalLdltFactors,
}

impl BlockLdltFactor {
    /// Solve `(L D Lᵀ) x = b` in-place.
    #[inline]
    pub fn solve_in_place(&self, b_in_x_out: &mut DVector<f64>) {
        puffin::profile_scope!("ldlt solve");

        let mat_l = &self.mat_l;
        let col_block_count = mat_l.subdivision.block_col_count();
        assert_eq!(
            mat_l.subdivision.scalar_shape()[0],
            mat_l.subdivision.scalar_shape()[1]
        );
        assert_eq!(b_in_x_out.len(), mat_l.subdivision.scalar_shape()[0]);

        // Solve: L y = b.
        for col_j in 0..col_block_count {
            let offset_col_j = mat_l.subdivision.scalar_col_offset(col_j);

            for e in mat_l.col(col_j).iter() {
                let row_i = e.block_row_idx;
                assert_gt!(row_i, col_j);

                let mat_l_ij = e.view;
                let scalar_offset_i = mat_l.subdivision.scalar_row_offset(row_i);

                // Split b into read-only b[j] and mutable y[i], avoiding aliasing:
                let (head, tail) = b_in_x_out.as_mut_slice().split_at_mut(scalar_offset_i);
                let b_j: &[f64] = &head[offset_col_j..offset_col_j + mat_l_ij.ncols()];
                let y_i: &mut [f64] = &mut tail[..mat_l_ij.nrows()];

                // y[i] -= L[i,j] * b[j]
                sub_mat_plus_vec_in_place(y_i, mat_l_ij, b_j);
            }
        }

        // Solve: D z = y.
        for col_j in 0..col_block_count {
            let offset_col_j = mat_l.subdivision.scalar_col_offset(col_j);

            let col_j_info = mat_l.subdivision.col_info(col_j);

            // z[j] := D⁻¹ * z[j]
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
            > = b_in_x_out.rows_mut(offset_col_j, col_j_info.block_width);
            self.block_diag.solve_inplace_vec(
                z_j,
                col_j_info.col_partition_idx,
                col_j_info.local_col_block_idx,
            );
        }

        // Solve: Lᵀ x = z.
        for col_j in (0..col_block_count).rev() {
            let offset_col_j = mat_l.subdivision.scalar_col_offset(col_j);

            for e in mat_l.col(col_j).iter() {
                let row_i = e.block_row_idx;
                assert_gt!(row_i, col_j);

                let mat_l_ij = e.view;

                let scalar_offset_i = mat_l.subdivision.scalar_row_offset(row_i);

                // Split b into mutable x[j] and read-only z[i], avoiding aliasing:
                let (head, tail) = b_in_x_out.as_mut_slice().split_at_mut(scalar_offset_i);
                let x_j: &mut [f64] = &mut head[offset_col_j..offset_col_j + mat_l_ij.ncols()];
                let z_i: &[f64] = &tail[..mat_l_ij.nrows()];

                // x[j] -= L[i,j]ᵀ * z[i]
                sub_mat_transpose_plus_vec_in_place(x_j, mat_l_ij, z_i);
            }
        }
    }
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

    type Matrix = BlockColCompressedMatrix;
    type Diag = BlockDiagonalLdltFactors;

    type MatrixEntry = DMatrix<f64>;
    type DiagnalEntry = (DMatrix<f64>, DVector<f64>);

    type MatLBuilder = BlockSparseLFactorBuilder;

    fn calc_etree(a_lower: &Self::Matrix) -> EliminationTree {
        EliminationTree::new(a_lower.build_pattern_upper())
    }

    fn activate_col(&mut self, col_j: usize) {
        let block_shape = self.mat_subdivision.block_shape([col_j, col_j]);

        self.mat_c.set_active_width(block_shape[1]);
        self.col_j = col_j;
        self.block_height_col_j = block_shape[0];
    }

    #[inline(always)]
    fn load_column(&mut self, a_lower: &BlockColCompressedMatrix) {
        for entry in a_lower.col(self.col_j).iter() {
            let row_i = entry.block_row_idx;
            let mat_a_ij = entry.view;

            let row_info_i = a_lower.subdivision.row_info(row_i);
            self.mat_c.add_block(
                row_info_i.row_partition_idx,
                row_info_i.local_block_row_idx,
                mat_a_ij,
            );

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

        let info_col_k = self.mat_subdivision.col_info(col_k);

        // Block diagonal LDLᵀ factors: L[k,k] and diag(d[k]).
        let mat_l_kk = diag
            .mat_l
            .get_block(info_col_k.col_partition_idx, info_col_k.local_col_block_idx);
        let diag_d_k = diag
            .d
            .get_block(info_col_k.col_partition_idx, info_col_k.local_col_block_idx);

        // Q[j,k] = L[j,k] * L[k,k].
        let mut mat_q_jk = self
            .mat_q_jk
            .view_range_mut(0..self.block_height_col_j, 0..info_col_k.block_width);
        mat_q_jk.gemm(1.0, &mat_l_jk, &mat_l_kk, 0.0);

        // C[j,j] -= Q[j,k] * Diag(d[k]) * Q[j,k]ᵀ
        let block_jj = self.mat_subdivision.col_info(self.col_j);
        let mut mat_c_jj = self
            .mat_c
            .get_block_mut(block_jj.col_partition_idx, block_jj.local_col_block_idx);
        sub_mat_diag_mat_t_in_place(
            &mut mat_c_jj,
            mat_q_jk.as_view(),
            diag_d_k.as_view(),
            mat_q_jk.as_view(),
        );

        let col_info_col_k = self.mat_subdivision.col_info(col_k);

        let column = &l_mat_builder.columns[col_k];
        for block in column.non_zero_blocks.iter() {
            let row_i = block.block_row_idx as usize;
            let storage_offset_of_block_ik = block.storage_base as usize;

            if row_i <= self.col_j {
                continue;
            }
            let info_row_i = self.mat_subdivision.row_info(row_i);

            let size_of_block_ik = info_row_i.block_height * col_info_col_k.block_width;
            let region_of_block_ik = l_mat_builder.regions.get(&[
                info_row_i.row_partition_idx,
                col_info_col_k.col_partition_idx,
            ]);
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
                .view_range_mut(0..info_row_i.block_height, 0..col_info_col_k.block_width);
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
            > = DMatrixView::from_slice(
                data_of_block_ik,
                info_row_i.block_height,
                col_info_col_k.block_width,
            );
            mat_q_ik.gemm(1.0, &mat_l_ik, &mat_l_kk, 0.0);

            // C(i,j) -= Q[i,k] * diag(d[k]) * Q[j,k]ᵀ
            let mut mat_c_ij = self
                .mat_c
                .get_block_mut(info_row_i.row_partition_idx, info_row_i.local_block_row_idx);
            sub_mat_diag_mat_t_in_place(
                &mut mat_c_ij,
                mat_q_ik.as_view(),
                diag_d_k.as_view(),
                mat_q_jk.as_view(),
            );

            #[cfg(debug_assertions)]
            {
                use crate::positive_semidefinite::LdltIndices;

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
        let col_info_col_j = self.mat_subdivision.col_info(self.col_j); // diagonal block info

        // Factor the diagonal: A[j,j] = L[j,j]ᵀ diag(d[j]) L[j,j]ᵀ.
        let mat_a_jj = self.mat_c.get_block(
            col_info_col_j.col_partition_idx,
            col_info_col_j.local_col_block_idx,
        );
        diag.decompose(
            col_info_col_j.col_partition_idx,
            col_info_col_j.local_col_block_idx,
            mat_a_jj.as_view(),
        )?;

        // Calculate off-diagonal: L[i,j].
        self.touched_rows.sort_unstable();
        for &row_i in &self.touched_rows {
            if row_i <= self.col_j {
                continue;
            }

            let row_info_i = self.mat_subdivision.row_info(row_i);
            let mut mat_c_ij = self
                .mat_c
                .get_block_mut(row_info_i.row_partition_idx, row_info_i.local_block_row_idx);

            // Right-solve in-place: C[i,j] := C[i,j] * L[j,j]⁻ᵀ diag(d[j])⁻¹ L[j,j]⁻¹.
            diag.right_solve_inplace(
                mat_c_ij.as_view_mut(),
                col_info_col_j.col_partition_idx,
                col_info_col_j.local_col_block_idx,
            );

            debug_assert_eq!(mat_c_ij.ncols(), col_info_col_j.block_width);
            l_mat_builder.append_to(row_i, self.col_j, mat_c_ij.as_view());
        }

        Ok(())
    }

    #[inline(always)]
    fn clear(&mut self) {
        for row_i in self.touched_rows.drain(..) {
            if self.was_row_touched[row_i] {
                let info_row_i = self.mat_subdivision.row_info(row_i);
                self.mat_c
                    .zero_block(info_row_i.row_partition_idx, info_row_i.local_block_row_idx);
                self.was_row_touched[row_i] = false;
            }
        }
    }
}

impl BlockLdltWorkspace {
    /// Create workspace for N x N matrix.
    pub fn new(n: usize, mat_subdivision: &BlockMatrixSubdivision) -> Self {
        let max_block_height = mat_subdivision.max_block_height();
        let max_block_width = mat_subdivision.max_block_width();
        let mat_q_jk = nalgebra::DMatrix::<f64>::zeros(max_block_height, max_block_width);
        let mat_q_ik = nalgebra::DMatrix::<f64>::zeros(max_block_height, max_block_width);
        Self {
            mat_c: BlockColumn::zero(mat_subdivision.row_partitons(), max_block_width),
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
    type Matrix = BlockColCompressedMatrix;

    fn compress(self) -> BlockColCompressedMatrix {
        let block_col_count = self.mat_subdivision.block_col_count();

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

        BlockColCompressedMatrix {
            block_col_pattern: BlockColCompressedPattern {
                nonzero_idx_by_block_col,
                nonzero_blocks,
            },
            subdivision: self.mat_subdivision,
            regions: self.regions,
        }
    }
}

impl BlockSparseLFactorBuilder {
    /// Crate a empty block-sparse factor L from matrix subdivision structure.
    pub fn new(mat_subdivision: &BlockMatrixSubdivision) -> Self {
        Self {
            columns: vec![BlockSparseLFactorColumn::default(); mat_subdivision.block_col_count()],
            regions: Grid::new(
                mat_subdivision.region_grid_shape(),
                BlockRegion {
                    storage: Vec::new(),
                },
            ),
            mat_subdivision: mat_subdivision.clone(),
        }
    }

    /// Append `L[i,j]` (with i>j).
    ///
    /// Precondition:
    ///   * Calls for the same column `j` arrive with ascending `i`.
    pub fn append_to(&mut self, row_i: usize, col_j: usize, mat_l_ij: DMatrixView<f64>) {
        debug_assert!(row_i > col_j, "strictly lower only");

        let row_info: crate::matrix::block_col_compressed_matrix::BlockRowInfo =
            self.mat_subdivision.row_info(row_i);
        let col_info = self.mat_subdivision.col_info(col_j);

        debug_assert_eq!(mat_l_ij.nrows(), row_info.block_height);
        debug_assert_eq!(mat_l_ij.ncols(), col_info.block_width);

        let region_ij = self
            .regions
            .get_mut(&[row_info.row_partition_idx, col_info.col_partition_idx]);
        let start = region_ij.storage.len();

        debug_assert_eq!(
            (region_ij.storage.len() - start) % (row_info.block_height * col_info.block_width),
            0,
        );

        for c in 0..col_info.block_width {
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

        let row_info = self.mat_subdivision.row_info(row_j);
        let col_info = self.mat_subdivision.col_info(col_k);

        let row_partition_idx = row_info.row_partition_idx;
        let col_partition_idx = col_info.col_partition_idx;
        let block_height = row_info.block_height;
        let block_width = col_info.block_width;
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
