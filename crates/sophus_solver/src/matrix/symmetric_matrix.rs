use enum_as_inner::EnumAsInner;
use nalgebra::DMatrix;

use crate::{
    LinearSolverEnum,
    matrix::{
        BlockRange,
        PartitionBlockIndex,
        PartitionSet,
        block_sparse::{
            BlockSparseSymmetricMatrixBuilder,
            block_sparse_symmetric_matrix::BlockSparseSymmetricMatrix,
        },
        dense::{
            DenseSymmetricMatrix,
            DenseSymmetricMatrixBuilder,
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
    /// mat
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
    Dense(DenseSymmetricMatrixBuilder),
    /// Builder for sparse symmetric matrix (with lower-triangular storage).
    SparseLower(SparseSymmetricMatrixBuilder),
    /// Builder for block-sparse symmetric matrix (with lower block-triangular storage).
    BlockSparseLower(BlockSparseSymmetricMatrixBuilder),
    /// Builder for sparse matrix to interact with the faer crate.
    FaerSparse(FaerSparseMatrixBuilder),
    /// Builder for sparse symmetric matrix (with upper-triangular storage) to interact with the
    /// faer crate.
    FaerSparseUpper(FaerSparseSymmetricMatrixBuilder),
}

impl SymmetricMatrixBuilderEnum {
    /// Create a symmetric matrix "filled" with zeros - to be used with given solver.
    ///
    /// The shape of the matrix is determined by the provided partition set.
    pub fn zero(solver: LinearSolverEnum, partitions: PartitionSet) -> Self {
        match solver {
            LinearSolverEnum::DenseLdlt(_) | LinearSolverEnum::DenseLu(_) => {
                SymmetricMatrixBuilderEnum::Dense(DenseSymmetricMatrixBuilder::zero(partitions))
            }
            LinearSolverEnum::FaerSparseQr(_) => {
                SymmetricMatrixBuilderEnum::FaerSparse(FaerSparseMatrixBuilder::zero(partitions))
            }
            LinearSolverEnum::FaerSparseLu(_) => {
                SymmetricMatrixBuilderEnum::FaerSparse(FaerSparseMatrixBuilder::zero(partitions))
            }
            LinearSolverEnum::FaerSparseLdlt(_) => SymmetricMatrixBuilderEnum::FaerSparseUpper(
                FaerSparseSymmetricMatrixBuilder::zero(partitions),
            ),
            LinearSolverEnum::SparseLdlt(_) => SymmetricMatrixBuilderEnum::SparseLower(
                SparseSymmetricMatrixBuilder::zero(partitions),
            ),
            LinearSolverEnum::BlockSparseLdlt(_) => SymmetricMatrixBuilderEnum::BlockSparseLower(
                BlockSparseSymmetricMatrixBuilder::zero(partitions),
            ),
        }
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
            SymmetricMatrixBuilderEnum::Dense(dense_symmetric_matrix_builder) => {
                dense_symmetric_matrix_builder.add_lower_block(row_idx, col_idx, block)
            }
            SymmetricMatrixBuilderEnum::FaerSparse(sparse_symmetric_matrix_builder) => {
                sparse_symmetric_matrix_builder.add_lower_block(row_idx, col_idx, block);
            }
            SymmetricMatrixBuilderEnum::FaerSparseUpper(sparse_symmetric_matrix_builder) => {
                sparse_symmetric_matrix_builder.add_lower_block(row_idx, col_idx, block);
            }
            SymmetricMatrixBuilderEnum::SparseLower(sparse_symmetric_matrix_builder) => {
                sparse_symmetric_matrix_builder.add_lower_block(row_idx, col_idx, block)
            }
            SymmetricMatrixBuilderEnum::BlockSparseLower(sparse_symmetric_matrix_builder) => {
                sparse_symmetric_matrix_builder.add_lower_block(row_idx, col_idx, block)
            }
        }
    }

    /// Build the matrix and return it.
    pub fn build(self) -> SymmetricMatrixEnum {
        match self {
            SymmetricMatrixBuilderEnum::Dense(dense_symmetric_matrix_builder) => {
                SymmetricMatrixEnum::Dense(dense_symmetric_matrix_builder.build())
            }
            SymmetricMatrixBuilderEnum::SparseLower(sparse_symmetric_matrix_builder) => {
                SymmetricMatrixEnum::SparseLower(sparse_symmetric_matrix_builder.build())
            }
            SymmetricMatrixBuilderEnum::FaerSparse(sparse_symmetric_matrix_builder) => {
                SymmetricMatrixEnum::FaerSparse(sparse_symmetric_matrix_builder.build())
            }
            SymmetricMatrixBuilderEnum::FaerSparseUpper(sparse_symmetric_matrix_builder) => {
                SymmetricMatrixEnum::FaerSparseUpper(sparse_symmetric_matrix_builder.build())
            }
            SymmetricMatrixBuilderEnum::BlockSparseLower(sparse_symmetric_matrix_builder) => {
                SymmetricMatrixEnum::BlockSparseLower(sparse_symmetric_matrix_builder.build())
            }
        }
    }
}

/// Symmetric `N x N` matrix enum.
#[derive(Debug, EnumAsInner)]
pub enum SymmetricMatrixEnum {
    /// Dense symmetric matrix.
    Dense(DenseSymmetricMatrix),
    /// Sparse symmetric matrix (with lower-triangular storage).
    SparseLower(SparseSymmetricMatrix),
    /// Block-sparse symmetric matrix (with lower block-triangular storage).
    BlockSparseLower(BlockSparseSymmetricMatrix),
    /// Sparse matrix to interact with the faer crate.
    FaerSparse(FaerSparseMatrix),
    /// Sparse symmetric matrix (with upper-triangular storage) to interact with the faer crate.
    FaerSparseUpper(FaerSparseSymmetricMatrix),
}

impl IsSymmetricMatrix for SymmetricMatrixEnum {
    fn get_block(
        &self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> nalgebra::DMatrix<f64> {
        match self {
            SymmetricMatrixEnum::Dense(dense_square_mat) => {
                dense_square_mat.get_block(row_idx, col_idx)
            }
            SymmetricMatrixEnum::SparseLower(column_compressed_matrix) => {
                column_compressed_matrix.get_block(row_idx, col_idx)
            }
            SymmetricMatrixEnum::BlockSparseLower(block_col_compressed_matrix) => {
                block_col_compressed_matrix.get_block(row_idx, col_idx)
            }
            SymmetricMatrixEnum::FaerSparse(faer_compressed_matrix) => {
                faer_compressed_matrix.get_block(row_idx, col_idx)
            }
            SymmetricMatrixEnum::FaerSparseUpper(faer_upper_compressed_matrix) => {
                faer_upper_compressed_matrix.get_block(row_idx, col_idx)
            }
        }
    }

    #[inline]
    fn partitions(&self) -> &PartitionSet {
        match self {
            SymmetricMatrixEnum::Dense(dense_square_mat) => dense_square_mat.partitions(),
            SymmetricMatrixEnum::SparseLower(lower_column_compressed_matrix) => {
                lower_column_compressed_matrix.partitions()
            }
            SymmetricMatrixEnum::BlockSparseLower(block_col_compressed_matrix) => {
                block_col_compressed_matrix.partitions()
            }
            SymmetricMatrixEnum::FaerSparse(faer_compressed_matrix) => {
                faer_compressed_matrix.partitions()
            }
            SymmetricMatrixEnum::FaerSparseUpper(faer_upper_compressed_matrix) => {
                faer_upper_compressed_matrix.partitions()
            }
        }
    }

    fn to_dense(&self) -> DMatrix<f64> {
        match self {
            SymmetricMatrixEnum::Dense(dense_square_mat) => dense_square_mat.to_dense(),
            SymmetricMatrixEnum::SparseLower(lower_column_compressed_matrix) => {
                lower_column_compressed_matrix.to_dense()
            }
            SymmetricMatrixEnum::BlockSparseLower(block_col_compressed_matrix) => {
                block_col_compressed_matrix.to_dense()
            }
            SymmetricMatrixEnum::FaerSparse(faer_compressed_matrix) => {
                faer_compressed_matrix.to_dense()
            }
            SymmetricMatrixEnum::FaerSparseUpper(faer_upper_compressed_matrix) => {
                faer_upper_compressed_matrix.to_dense()
            }
        }
    }
}
