use nalgebra::{
    DMatrix,
    DVector,
};

use crate::{
    LinearSolverEnum,
    error::LinearSolverError,
    matrix::{
        BlockRange,
        PartitionBlockIndex,
        PartitionSet,
        block_sparse::{
            BlockSparseSymmetricMatrixBuilder,
            block_sparse_symmetric_matrix::BlockSparseSymmetricMatrix,
            block_sparse_symmetric_matrix_builder::BlockSparseSymmetricMatrixPattern,
        },
        dense::{
            DenseSymmetricMatrix,
            DenseSymmetricMatrixBuilder,
        },
        direct_solve::{
            DirectSolve,
            DirectSolveMatrix,
        },
        sparse::{
            FaerSparseMatrix,
            FaerSparseMatrixBuilder,
            FaerSparseSymmetricMatrix,
            FaerSparseSymmetricMatrixBuilder,
            SparseSymmetricMatrixBuilder,
            sparse_symmetric_matrix::SparseSymmetricMatrix,
        },
    },
};

/// Builder trait for a symmetric `N x N` matrix.
pub trait IsSymmetricMatrixBuilder {
    /// Symmetric matrix type.
    type Matrix: IsSymmetricMatrix;

    /// Create a symmetric matrix "filled" with zeros.
    ///
    /// The number and arrangement of regions and blocks, and scalar height (and width) of the
    /// matrix is determined by the partition set.
    fn zero(partitions: PartitionSet) -> Self;

    /// Scalar dimension of the matrix.
    fn scalar_dim(&self) -> usize;

    /// The row/column partition set.
    ///
    /// Since the matrix is symmetric, the row partition set equals the column partition set.
    fn partitions(&self) -> &PartitionSet;

    /// Add a block to the matrix.
    ///
    /// This is a += operation, i.e., the block is added to the existing block.
    ///
    /// How a block is saved is up to the individual implementation.
    ///
    /// Preconditions:
    ///  - Blocks must target the lower block-triangular area of the matrix (row_idx >= col_idx).
    ///  - Blocks on the diagonal must be self-symmetric.
    fn add_lower_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
        block: &nalgebra::DMatrixView<f64>,
    );

    /// Build the matrix and return it.
    fn build(self) -> Self::Matrix;
}

/// Symmetric `N x N` matrix trait.
pub trait IsSymmetricMatrix {
    /// Return true if a non-zero block exists at (row_idx, col_idx) — no allocation.
    ///
    /// Default implementation materializes the block; override for O(1)/O(log n) performance.
    fn has_block(&self, row_idx: PartitionBlockIndex, col_idx: PartitionBlockIndex) -> bool {
        self.get_block(row_idx, col_idx).iter().any(|&x| x != 0.0)
    }

    /// Extract block at index `row_idx`, `col_idx`.
    fn get_block(
        &self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> nalgebra::DMatrix<f64>;

    /// Returns the row/column partition set.
    ///
    /// Since this is a square matrix, the row partition set equals the column partition set.
    fn partitions(&self) -> &PartitionSet;

    /// Block range for the block at index `idx`.
    #[inline]
    fn block_range(&self, idx: PartitionBlockIndex) -> BlockRange {
        self.partitions().block_range(idx)
    }

    /// Solve the linear system `H x = rhs` in-place, returning `x`.
    ///
    /// Implementations may cache internal state (symbolic factorization, Schur
    /// complement pattern) across calls.
    fn solve(&mut self, _rhs: &DVector<f64>) -> Result<DVector<f64>, LinearSolverError> {
        panic!(
            "solve() not supported on this IsSymmetricMatrix variant; use LinearSolverEnum::factorize directly"
        )
    }

    /// Extract a block from the matrix pseudo-inverse H⁺.
    ///
    /// The default implementation computes the full dense SVD pseudo-inverse and
    /// extracts the requested block. Schur-complement variants override this with
    /// a more efficient formula that avoids the full inverse.
    fn inverse_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> Result<DMatrix<f64>, LinearSolverError> {
        let dense = self.to_dense();
        let pinv = nalgebra::SVD::new(dense, true, true)
            .pseudo_inverse(1e-10)
            .expect("pseudo-inverse of symmetric matrix");
        let rr = self.block_range(row_idx);
        let cr = self.block_range(col_idx);
        Ok(pinv
            .view((rr.start_idx, cr.start_idx), (rr.block_dim, cr.block_dim))
            .into_owned())
    }

    /// Construct dense matrix and returns it.
    fn to_dense(&self) -> DMatrix<f64> {
        let partitions = self.partitions();
        let n = partitions.scalar_dim();
        let mut out = DMatrix::<f64>::zeros(n, n);

        for row_partition_idx in 0..partitions.len() {
            let block_row_count = partitions.specs()[row_partition_idx].block_count;
            for row in 0..block_row_count {
                let row_idx = PartitionBlockIndex {
                    partition: row_partition_idx,
                    block: row,
                };
                let row_range = partitions.block_range(row_idx);

                for col_partition_count in 0..partitions.len() {
                    let block_col_count = partitions.specs()[col_partition_count].block_count;
                    for col in 0..block_col_count {
                        let col_idx = PartitionBlockIndex {
                            partition: col_partition_count,
                            block: col,
                        };
                        let col_range = partitions.block_range(col_idx);

                        let block = self.get_block(row_idx, col_idx);
                        debug_assert_eq!(block.nrows(), row_range.block_dim);
                        debug_assert_eq!(block.ncols(), col_range.block_dim);

                        out.view_mut(
                            (row_range.start_idx, col_range.start_idx),
                            (row_range.block_dim, col_range.block_dim),
                        )
                        .copy_from(&block);
                    }
                }
            }
        }

        out
    }
}

#[derive(Debug, Clone)]
/// Builder enum for a symmetric `N x N` matrix.
pub enum SymmetricMatrixBuilderEnum {
    /// Builder for dense symmetric matrix.
    Dense(DenseSymmetricMatrixBuilder, LinearSolverEnum),
    /// Builder for sparse symmetric matrix (with lower-triangular storage).
    SparseLower(SparseSymmetricMatrixBuilder, LinearSolverEnum),
    /// Builder for block-sparse symmetric matrix (with lower block-triangular storage).
    BlockSparseLower(BlockSparseSymmetricMatrixBuilder, LinearSolverEnum),
    /// Builder for sparse matrix to interact with the faer crate.
    FaerSparse(FaerSparseMatrixBuilder, LinearSolverEnum),
    /// Builder for sparse symmetric matrix (with upper-triangular storage) to interact with the
    /// faer crate.
    FaerSparseUpper(FaerSparseSymmetricMatrixBuilder, LinearSolverEnum),
    /// Pre-computed-pattern block-sparse builder.  Avoids HashMap deduplication and
    /// per-iteration triplet allocation; use across repeated optimizer iterations.
    BlockSparsePattern(BlockSparseSymmetricMatrixPattern, LinearSolverEnum),
}

impl SymmetricMatrixBuilderEnum {
    /// Create a symmetric matrix "filled" with zeros - to be used with given solver.
    ///
    /// The shape of the matrix is determined by the provided partition set.
    pub fn zero(solver: LinearSolverEnum, partitions: PartitionSet) -> Self {
        match solver {
            LinearSolverEnum::DenseLdlt(_) | LinearSolverEnum::DenseLu(_) => {
                SymmetricMatrixBuilderEnum::Dense(
                    DenseSymmetricMatrixBuilder::zero(partitions),
                    solver,
                )
            }
            LinearSolverEnum::FaerSparseQr(_) => SymmetricMatrixBuilderEnum::FaerSparse(
                FaerSparseMatrixBuilder::zero(partitions),
                solver,
            ),
            LinearSolverEnum::FaerSparseLu(_) => SymmetricMatrixBuilderEnum::FaerSparse(
                FaerSparseMatrixBuilder::zero(partitions),
                solver,
            ),
            LinearSolverEnum::FaerSparseLdlt(_) => SymmetricMatrixBuilderEnum::FaerSparseUpper(
                FaerSparseSymmetricMatrixBuilder::zero(partitions),
                solver,
            ),
            LinearSolverEnum::SparseLdlt(_) => SymmetricMatrixBuilderEnum::SparseLower(
                SparseSymmetricMatrixBuilder::zero(partitions),
                solver,
            ),
            LinearSolverEnum::BlockSparseLdlt(_) => SymmetricMatrixBuilderEnum::BlockSparseLower(
                BlockSparseSymmetricMatrixBuilder::zero(partitions),
                solver,
            ),
        }
    }

    /// Create a pattern-based block-sparse builder from a pre-computed pattern.
    ///
    /// Resets the pattern storage to zero and returns a builder ready for accumulation.
    /// Prefer this over `zero(BlockSparseLdlt, ...)` when the sparsity pattern is stable
    /// across optimizer iterations.
    pub fn from_block_sparse_pattern(
        mut pat: BlockSparseSymmetricMatrixPattern,
        solver: LinearSolverEnum,
    ) -> Self {
        pat.reset();
        SymmetricMatrixBuilderEnum::BlockSparsePattern(pat, solver)
    }

    /// Add a block to the matrix.
    ///
    /// This is a += operation, i.e., the block is added to the existing block.
    ///
    /// Only blocks targeting the block lower-triangular area of the matrix are accepted.
    ///
    /// In release mode, upper triangular blocks are ignored. In debug mode,
    /// this function will panic if the block is added to the upper triangular part.
    pub fn add_lower_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
        block: &nalgebra::DMatrixView<f64>,
    ) {
        match self {
            SymmetricMatrixBuilderEnum::Dense(b, _) => b.add_lower_block(row_idx, col_idx, block),
            SymmetricMatrixBuilderEnum::FaerSparse(b, _) => {
                b.add_lower_block(row_idx, col_idx, block);
            }
            SymmetricMatrixBuilderEnum::FaerSparseUpper(b, _) => {
                b.add_lower_block(row_idx, col_idx, block);
            }
            SymmetricMatrixBuilderEnum::SparseLower(b, _) => {
                b.add_lower_block(row_idx, col_idx, block)
            }
            SymmetricMatrixBuilderEnum::BlockSparseLower(b, _) => {
                b.add_lower_block(row_idx, col_idx, block)
            }
            SymmetricMatrixBuilderEnum::BlockSparsePattern(pat, _) => {
                pat.add_lower_block(row_idx, col_idx, block)
            }
        }
    }

    /// Build the matrix and return it.
    pub fn build(self) -> SymmetricMatrixEnum {
        match self {
            SymmetricMatrixBuilderEnum::Dense(b, solver) => SymmetricMatrixEnum::Direct(
                DirectSolve::new(DirectSolveMatrix::Dense(b.build()), solver),
            ),
            SymmetricMatrixBuilderEnum::SparseLower(b, solver) => SymmetricMatrixEnum::Direct(
                DirectSolve::new(DirectSolveMatrix::SparseLower(b.build()), solver),
            ),
            SymmetricMatrixBuilderEnum::FaerSparse(b, solver) => SymmetricMatrixEnum::Direct(
                DirectSolve::new(DirectSolveMatrix::FaerSparse(b.build()), solver),
            ),
            SymmetricMatrixBuilderEnum::FaerSparseUpper(b, solver) => SymmetricMatrixEnum::Direct(
                DirectSolve::new(DirectSolveMatrix::FaerSparseUpper(b.build()), solver),
            ),
            SymmetricMatrixBuilderEnum::BlockSparseLower(b, solver) => SymmetricMatrixEnum::Direct(
                DirectSolve::new(DirectSolveMatrix::BlockSparseLower(b.build()), solver),
            ),
            SymmetricMatrixBuilderEnum::BlockSparsePattern(pat, solver) => {
                SymmetricMatrixEnum::Direct(DirectSolve::new(
                    DirectSolveMatrix::BlockSparseLower(pat.build()),
                    solver,
                ))
            }
        }
    }

    /// Build the matrix and return it together with the consumed pattern (if any).
    ///
    /// When a `BlockSparsePattern` or `BlockSparseLower` variant is used, returns the
    /// corresponding pattern for reuse in the next iteration.  For other variants returns
    /// `None`.
    pub fn build_with_pattern(
        self,
    ) -> (
        SymmetricMatrixEnum,
        Option<BlockSparseSymmetricMatrixPattern>,
    ) {
        match self {
            SymmetricMatrixBuilderEnum::BlockSparsePattern(mut pat, solver) => {
                // Build the matrix from the current accumulated values.
                let mat = pat.build();
                // Reuse the same pattern for the next iteration: just zero the storage.
                pat.reset();
                (
                    SymmetricMatrixEnum::Direct(DirectSolve::new(
                        DirectSolveMatrix::BlockSparseLower(mat),
                        solver,
                    )),
                    Some(pat),
                )
            }
            SymmetricMatrixBuilderEnum::BlockSparseLower(builder, solver) => {
                let mat = builder.build();
                let pat = mat.clone().into_pattern();
                (
                    SymmetricMatrixEnum::Direct(DirectSolve::new(
                        DirectSolveMatrix::BlockSparseLower(mat),
                        solver,
                    )),
                    Some(pat),
                )
            }
            other => (other.build(), None),
        }
    }
}

/// Symmetric `N x N` matrix enum.
#[derive(Debug)]
pub enum SymmetricMatrixEnum {
    /// Direct-solve path: any non-Schur matrix bundled with its solver.
    Direct(DirectSolve),
}

impl SymmetricMatrixEnum {
    /// Construct a `Direct(BlockSparseLower(...))` variant with the given solver.
    pub fn from_block_sparse_lower(
        mat: BlockSparseSymmetricMatrix,
        solver: LinearSolverEnum,
    ) -> Self {
        SymmetricMatrixEnum::Direct(DirectSolve::new(
            DirectSolveMatrix::BlockSparseLower(mat),
            solver,
        ))
    }

    /// Consume and return the inner `BlockSparseSymmetricMatrix` if this is a
    /// `Direct(BlockSparseLower(...))` variant, otherwise `None`.
    pub fn into_block_sparse_lower(self) -> Option<BlockSparseSymmetricMatrix> {
        match self {
            SymmetricMatrixEnum::Direct(ds) => match ds.inner {
                DirectSolveMatrix::BlockSparseLower(m) => Some(m),
                _ => None,
            },
        }
    }

    /// Return a reference to the `DirectSolve` wrapper, if this is the `Direct` variant.
    pub fn as_direct(&self) -> Option<&DirectSolve> {
        match self {
            SymmetricMatrixEnum::Direct(ds) => Some(ds),
        }
    }

    /// Return a mutable reference to the `DirectSolve` wrapper, if this is the `Direct` variant.
    pub fn as_direct_mut(&mut self) -> Option<&mut DirectSolve> {
        match self {
            SymmetricMatrixEnum::Direct(ds) => Some(ds),
        }
    }

    /// Return a reference to the inner `BlockSparseSymmetricMatrix` if `Direct(BlockSparseLower)`.
    pub fn as_block_sparse_lower(&self) -> Option<&BlockSparseSymmetricMatrix> {
        match self {
            SymmetricMatrixEnum::Direct(ds) => ds.inner.as_block_sparse_lower(),
        }
    }

    /// Return a reference to the inner `DenseSymmetricMatrix` if `Direct(Dense)`.
    pub fn as_dense(&self) -> Option<&DenseSymmetricMatrix> {
        match self {
            SymmetricMatrixEnum::Direct(ds) => ds.inner.as_dense(),
        }
    }

    /// Return a reference to the inner `SparseSymmetricMatrix` if `Direct(SparseLower)`.
    pub fn as_sparse_lower(&self) -> Option<&SparseSymmetricMatrix> {
        match self {
            SymmetricMatrixEnum::Direct(ds) => ds.inner.as_sparse_lower(),
        }
    }

    /// Return a reference to the inner `FaerSparseMatrix` if `Direct(FaerSparse)`.
    pub fn as_faer_sparse(&self) -> Option<&FaerSparseMatrix> {
        match self {
            SymmetricMatrixEnum::Direct(ds) => ds.inner.as_faer_sparse(),
        }
    }

    /// Return a reference to the inner `FaerSparseSymmetricMatrix` if `Direct(FaerSparseUpper)`.
    pub fn as_faer_sparse_upper(&self) -> Option<&FaerSparseSymmetricMatrix> {
        match self {
            SymmetricMatrixEnum::Direct(ds) => ds.inner.as_faer_sparse_upper(),
        }
    }
}

impl IsSymmetricMatrix for SymmetricMatrixEnum {
    fn has_block(&self, row_idx: PartitionBlockIndex, col_idx: PartitionBlockIndex) -> bool {
        match self {
            SymmetricMatrixEnum::Direct(ds) => ds.has_block(row_idx, col_idx),
        }
    }

    fn get_block(
        &self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> nalgebra::DMatrix<f64> {
        match self {
            SymmetricMatrixEnum::Direct(ds) => ds.get_block(row_idx, col_idx),
        }
    }

    #[inline]
    fn partitions(&self) -> &PartitionSet {
        match self {
            SymmetricMatrixEnum::Direct(ds) => ds.partitions(),
        }
    }

    fn to_dense(&self) -> DMatrix<f64> {
        match self {
            SymmetricMatrixEnum::Direct(ds) => ds.to_dense(),
        }
    }

    fn solve(&mut self, rhs: &DVector<f64>) -> Result<DVector<f64>, LinearSolverError> {
        match self {
            SymmetricMatrixEnum::Direct(ds) => ds.solve(rhs),
        }
    }

    fn inverse_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> Result<DMatrix<f64>, LinearSolverError> {
        match self {
            SymmetricMatrixEnum::Direct(ds) => ds.inverse_block(row_idx, col_idx),
        }
    }
}

impl SymmetricMatrixEnum {
    /// Return a lower-triangular block, or `None` if structurally zero.
    ///
    /// For `BlockSparseLower` this is a single binary search with no extra allocation.
    /// For other variants it falls back to `has_block` + `get_block`.
    #[inline]
    pub fn try_get_lower_block(
        &self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> Option<DMatrix<f64>> {
        match self {
            SymmetricMatrixEnum::Direct(ds) => match &ds.inner {
                DirectSolveMatrix::BlockSparseLower(m) => m
                    .try_get_lower_block_view(row_idx, col_idx)
                    .map(|v| v.clone_owned()),
                _ => {
                    if ds.has_block(row_idx, col_idx) {
                        Some(ds.get_block(row_idx, col_idx))
                    } else {
                        None
                    }
                }
            },
        }
    }

    /// Zero-copy view of a lower-triangular block, or `None` if structurally zero.
    ///
    /// Only works for `BlockSparseLower`; returns `None` for all other variants.
    #[inline]
    pub fn try_get_lower_block_view<'a>(
        &'a self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> Option<nalgebra::DMatrixView<'a, f64>> {
        match self {
            SymmetricMatrixEnum::Direct(ds) => match &ds.inner {
                DirectSolveMatrix::BlockSparseLower(m) => {
                    m.try_get_lower_block_view(row_idx, col_idx)
                }
                _ => None,
            },
        }
    }

    /// Visit every non-zero lower-triangular H_mf block across all free columns.
    ///
    /// Scans global block columns `0..total_free_blocks` in order.  For each entry
    /// whose row scalar start is >= `nf` (i.e. belongs to a marginalized partition),
    /// calls `f(free_scalar_start, free_dim, marg_offset, view)` where
    /// `marg_offset = row_scalar_start - nf`.
    ///
    /// Only active for `BlockSparseLower` matrices; no-op for all other variants.
    #[inline]
    pub fn visit_lower_hff_hmf<F>(&self, total_free_blocks: usize, nf: usize, f: F)
    where
        F: FnMut(usize, usize, usize, nalgebra::DMatrixView<'_, f64>),
    {
        match self {
            SymmetricMatrixEnum::Direct(ds) => {
                if let DirectSolveMatrix::BlockSparseLower(m) = &ds.inner {
                    m.visit_lower_hff_hmf(total_free_blocks, nf, f)
                }
            }
        }
    }
}
