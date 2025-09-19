use std::fmt::Debug;

use sophus_assert::debug_assert_ge;

use crate::matrix::{
    IsSymmetricMatrixBuilder,
    PartitionBlockIndex,
    PartitionSet,
    block_sparse::{
        BlockSparseMatrixBuilder,
        block_sparse_symmetric_matrix::BlockSparseSymmetricMatrix,
    },
};

/// A builder for a symmetric block sparse matrix.
///
/// Internally, the symmetric matrix is represented as a lower block-triangular matrix.
///
/// ```ascii
/// -------------------------------------------
/// | AxA         |             |             |
/// |  .  .       |             |             |
/// |  .     .    |             |             |
/// | AxA ... AxA |             |             |
/// -------------------------------------------
/// | BxA ... BxA | BxB         |             |
/// |  .       .  |  .  .       |             |
/// |  .       .  |  .     .    |             |
/// | BxA ... BxA | BxB ... BxB |             |
/// -------------------------------------------
/// |             |             |             |
/// |      *      |             |    *        |
/// |      *      |             |        *    |
/// |             |             |             |
/// -------------------------------------------
/// ```
#[derive(Debug, Clone)]
pub struct BlockSparseSymmetricMatrixBuilder {
    lower_triangular: BlockSparseMatrixBuilder,
}

impl BlockSparseSymmetricMatrixBuilder {
    /// Number of partitions horizontally (or vertically).
    #[inline]
    pub fn partition_count(&self) -> usize {
        self.lower_triangular.triplets.partition_count()
    }
}

impl IsSymmetricMatrixBuilder for BlockSparseSymmetricMatrixBuilder {
    type Matrix = BlockSparseSymmetricMatrix;

    fn zero(partitions: PartitionSet) -> Self {
        Self {
            lower_triangular: BlockSparseMatrixBuilder::zero(partitions),
        }
    }

    #[inline]
    fn scalar_dim(&self) -> usize {
        self.lower_triangular.triplets.scalar_dimension()
    }

    fn partitions(&self) -> &PartitionSet {
        &self.lower_triangular.triplets.partitions
    }

    fn add_lower_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
        block: &nalgebra::DMatrixView<f64>,
    ) {
        debug_assert_ge!(row_idx.partition, col_idx.partition);

        self.lower_triangular.add_block(row_idx, col_idx, block);
    }

    fn build(self) -> Self::Matrix {
        BlockSparseSymmetricMatrix {
            lower: self.lower_triangular.triplets.to_compressed(),
        }
    }
}
