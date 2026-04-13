use crate::matrix::{
    IsSymmetricMatrix,
    IsSymmetricMatrixBuilder,
    PartitionSet,
    block_sparse::{
        BlockMatrixSubdivision,
        BlockSparseMatrix,
        block_sparse_matrix_pattern::BlockSparseSymbolicBuilder,
        block_sparse_symmetric_matrix_builder::BlockSparseSymmetricMatrixPattern,
    },
    dense::DenseSymmetricMatrixBuilder,
    sparse::{
        FaerSparseMatrix,
        FaerSparseMatrixBuilder,
        FaerSparseSymmetricMatrix,
        FaerSparseSymmetricMatrixBuilder,
        SparseSymmetricMatrixBuilder,
        sparse_symmetric_matrix::SparseSymmetricMatrix,
    },
};

/// Symmetric `N x N` matrix in column compressed block-sparse form.
///
/// It is a newtype over [BlockSparseMatrix]. The symmetric matrix is represented by
/// a lower block-triangular matrix.
#[derive(Clone, Debug)]
pub struct BlockSparseSymmetricMatrix {
    pub(crate) lower: BlockSparseMatrix,
}

impl BlockSparseSymmetricMatrix {
    /// Return the matrix subdivision.
    pub fn subdivision(&self) -> &BlockMatrixSubdivision {
        &self.lower.subdivision
    }

    /// Return the internal lower block-triangular matrix.
    pub fn into_lower(self) -> BlockSparseMatrix {
        self.lower
    }

    /// Extract the sparsity pattern from this matrix and return a pre-allocated
    /// [`BlockSparseSymmetricMatrixPattern`] ready for reuse across optimizer iterations.
    ///
    /// The returned pattern has zeroed storage; call `add_lower_block` to accumulate, then
    /// `build()` to produce the next iteration's matrix without any HashMap allocation.
    pub fn into_pattern(self) -> BlockSparseSymmetricMatrixPattern {
        let nb = self.lower.block_count();
        let mut sym_builder =
            BlockSparseSymbolicBuilder::new(self.lower.subdivision.partitions().clone());

        for global_col in 0..nb {
            let col_idx = self.lower.subdivision.idx(global_col);
            for entry in self.lower.non_zero_block(global_col) {
                let global_row = entry.block_row_idx as usize;
                let row_idx = self.lower.subdivision.idx(global_row);
                sym_builder.add_block(row_idx, col_idx);
            }
        }

        BlockSparseSymmetricMatrixPattern {
            inner: sym_builder.into_pattern(),
            workers: Vec::new(),
        }
    }

    /// Convert this block-sparse matrix to a scalar-sparse lower-triangular matrix.
    ///
    /// Iterates over all non-zero lower-triangular blocks and accumulates them into a
    /// [`SparseSymmetricMatrix`].  Used to feed the result of the `BlockSparsePattern`
    /// fast-accumulation path into scalar-sparse solvers (`SparseLdlt`).
    pub fn to_sparse_symmetric(&self) -> SparseSymmetricMatrix {
        let mut builder = SparseSymmetricMatrixBuilder::zero(self.partitions().clone());
        let col_count = self.lower.block_count();
        for col_g in 0..col_count {
            let col_idx = self.lower.subdivision.idx(col_g);
            for entry in self.lower.col(col_g).iter() {
                let row_idx = self.lower.subdivision.idx(entry.global_block_row_idx);
                builder.add_lower_block(row_idx, col_idx, &entry.view);
            }
        }
        builder.build()
    }

    /// Convert this block-sparse matrix to a faer upper-triangular sparse matrix.
    ///
    /// Used to feed the result of the `BlockSparsePattern` fast-accumulation path into
    /// [`crate::ldlt::FaerSparseLdlt`].
    pub fn to_faer_sparse_symmetric(&self) -> FaerSparseSymmetricMatrix {
        let mut builder = FaerSparseSymmetricMatrixBuilder::zero(self.partitions().clone());
        let col_count = self.lower.block_count();
        for col_g in 0..col_count {
            let col_idx = self.lower.subdivision.idx(col_g);
            for entry in self.lower.col(col_g).iter() {
                let row_idx = self.lower.subdivision.idx(entry.global_block_row_idx);
                builder.add_lower_block(row_idx, col_idx, &entry.view);
            }
        }
        builder.build()
    }

    /// Convert this block-sparse matrix to a faer upper-triangular sparse matrix,
    /// preserving ALL scalar positions within existing blocks (including zeros).
    ///
    /// Use this when a cached symbolic factorization will be reused across iterations.
    /// The standard [`to_faer_sparse_symmetric`] skips scalar zeros, which can change
    /// the sparsity pattern between iterations and invalidate a pre-built symbolic factor.
    pub(crate) fn to_faer_sparse_symmetric_structural(&self) -> FaerSparseSymmetricMatrix {
        use faer::sparse::Triplet;

        let partitions = self.partitions().clone();
        let n = partitions.scalar_dim();
        let mut upper_triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();

        let col_count = self.lower.block_count();
        for col_g in 0..col_count {
            let col_idx = self.lower.subdivision.idx(col_g);
            let col_range = partitions.block_range(col_idx);
            for entry in self.lower.col(col_g).iter() {
                let row_idx = self.lower.subdivision.idx(entry.global_block_row_idx);
                let row_range = partitions.block_range(row_idx);
                let v = &entry.view;
                let is_diag_block =
                    col_idx.partition == row_idx.partition && col_idx.block == row_idx.block;
                if is_diag_block {
                    // Diagonal block: only upper triangle (r <= c in scalar coords).
                    for c in 0..col_range.block_dim {
                        for r in 0..=c {
                            upper_triplets.push(Triplet {
                                row: row_range.start_idx + r,
                                col: col_range.start_idx + c,
                                val: v[(r, c)],
                            });
                        }
                    }
                } else {
                    // Off-diagonal lower block (row_g > col_g): store as upper (swap r,c).
                    for c in 0..col_range.block_dim {
                        for r in 0..row_range.block_dim {
                            upper_triplets.push(Triplet {
                                row: col_range.start_idx + c,
                                col: row_range.start_idx + r,
                                val: v[(r, c)],
                            });
                        }
                    }
                }
            }
        }

        FaerSparseSymmetricMatrix {
            upper: faer::sparse::SparseColMat::try_new_from_triplets(n, n, &upper_triplets)
                .unwrap(),
            partitions,
        }
    }

    /// Convert this block-sparse matrix to a faer sparse matrix (for QR/LU solvers).
    pub fn to_faer_sparse(&self) -> FaerSparseMatrix {
        let mut builder = FaerSparseMatrixBuilder::zero(self.partitions().clone());
        let col_count = self.lower.block_count();
        for col_g in 0..col_count {
            let col_idx = self.lower.subdivision.idx(col_g);
            for entry in self.lower.col(col_g).iter() {
                let row_idx = self.lower.subdivision.idx(entry.global_block_row_idx);
                builder.add_lower_block(row_idx, col_idx, &entry.view);
            }
        }
        builder.build()
    }

    /// Convert this block-sparse matrix to a dense symmetric matrix.
    ///
    /// Only suitable for small matrices (e.g. test problems); large bundle adjustment problems
    /// should not use dense solvers.
    pub fn to_dense_symmetric(&self) -> crate::matrix::dense::DenseSymmetricMatrix {
        let mut builder = DenseSymmetricMatrixBuilder::zero(self.partitions().clone());
        let col_count = self.lower.block_count();
        for col_g in 0..col_count {
            let col_idx = self.lower.subdivision.idx(col_g);
            for entry in self.lower.col(col_g).iter() {
                let row_idx = self.lower.subdivision.idx(entry.global_block_row_idx);
                builder.add_lower_block(row_idx, col_idx, &entry.view);
            }
        }
        builder.build()
    }

    /// Visit every non-zero lower-triangular H_mf block across all free columns.
    ///
    /// Scans global block columns `0..total_free_blocks` in order. For each entry
    /// whose row scalar start is ≥ `nf` (marg partition), calls
    /// `f(free_scalar_start, free_dim, marg_offset, view)`.
    #[inline]
    pub fn visit_lower_hff_hmf<F>(&self, total_free_blocks: usize, nf: usize, mut f: F)
    where
        F: FnMut(usize, usize, usize, nalgebra::DMatrixView<'_, f64>),
    {
        for free_global in 0..total_free_blocks {
            let free_ss = self.lower.subdivision.scalar_offset(free_global);
            let free_dim = self.lower.subdivision.scalar_offset(free_global + 1) - free_ss;
            for entry in self.lower.col(free_global).iter() {
                let marg_ss = self
                    .lower
                    .subdivision
                    .scalar_offset(entry.global_block_row_idx);
                if marg_ss >= nf {
                    f(free_ss, free_dim, marg_ss - nf, entry.view);
                }
            }
        }
    }

    /// Zero-copy view of a lower-triangular block (row >= col in partition order).
    ///
    /// Returns `None` when the block is structurally zero.
    /// Panics in debug mode if called with an upper-triangular (row < col) index pair.
    pub fn try_get_lower_block_view(
        &self,
        row_idx: crate::matrix::PartitionBlockIndex,
        col_idx: crate::matrix::PartitionBlockIndex,
    ) -> Option<nalgebra::DMatrixView<'_, f64>> {
        debug_assert!(
            (row_idx.partition > col_idx.partition)
                || (row_idx.partition == col_idx.partition && row_idx.block >= col_idx.block),
            "try_get_lower_block_view requires row >= col (lower triangular)"
        );
        self.lower.try_get_block_view(row_idx, col_idx)
    }
}

impl BlockSparseSymmetricMatrix {
    /// Subtract `nu` from every scalar diagonal entry `M[i,i]` in-place.
    ///
    /// Only diagonal blocks (where block_row == block_col) are touched.
    /// The sparsity structure is unchanged.
    pub fn subtract_scalar_diagonal(&mut self, nu: f64) {
        let block_count = self.lower.block_count();
        for global_col in 0..block_count {
            let col_slice = self.lower.non_zero_block(global_col);
            // Find the diagonal entry (block_row_idx == global_col).
            let diag_pos =
                col_slice.binary_search_by(|e| (e.block_row_idx as usize).cmp(&global_col));
            if let Ok(pos) = diag_pos {
                let entry = col_slice[pos];
                let idx = self.lower.subdivision.idx(global_col);
                let block_dim = self.lower.subdivision.block_dim(idx.partition);
                let base = entry.storage_base as usize;
                let storage = &mut self
                    .lower
                    .regions
                    .get_mut(&[idx.partition, idx.partition])
                    .storage;
                // Subtract nu from each scalar diagonal within this block.
                // Block is stored column-major with dimensions block_dim x block_dim.
                for k in 0..block_dim {
                    storage[base + k * block_dim + k] -= nu;
                }
            }
        }
    }
}

impl IsSymmetricMatrix for BlockSparseSymmetricMatrix {
    fn has_block(
        &self,
        row_idx: crate::matrix::PartitionBlockIndex,
        col_idx: crate::matrix::PartitionBlockIndex,
    ) -> bool {
        let is_lower = (row_idx.partition > col_idx.partition)
            || (row_idx.partition == col_idx.partition && row_idx.block >= col_idx.block);
        if is_lower {
            self.lower.has_block(row_idx, col_idx)
        } else {
            self.lower.has_block(col_idx, row_idx)
        }
    }

    fn get_block(
        &self,
        row_idx: crate::matrix::PartitionBlockIndex,
        col_idx: crate::matrix::PartitionBlockIndex,
    ) -> nalgebra::DMatrix<f64> {
        let is_lower = (row_idx.partition > col_idx.partition)
            || (row_idx.partition == col_idx.partition && row_idx.block >= col_idx.block);

        if is_lower {
            self.lower.get_block(row_idx, col_idx)
        } else {
            self.lower.get_block(col_idx, row_idx).transpose()
        }
    }

    fn partitions(&self) -> &PartitionSet {
        self.lower.subdivision.partitions()
    }
}
