use std::fmt::Debug;

use super::{
    block_sparse_matrix_builder::{
        BlockSparseMatrixBuilder,
        ToDenseImplMode,
        ToScalarTripletsImplMode,
    },
    PartitionSpec,
};
use crate::debug_assert_le;

/// A builder for a symmetric block sparse matrix.
///
/// Internally, we chose to represent the symmetric matrix as a upper triangular matrix.
/// In particular, the target block sparse matrix has the following structure:
///
/// ```ascii
/// ---------------------------------------------------------
/// | AxA ... AxA | AxB ... AxB |             | AxZ ... AxZ |
/// |   .         |  .       .  |             |  .       .  |
/// |      .      |  .       .  |   *  *  *   |  .       .  |
/// |         AxA | AxB ... AxB |             | AxZ ... AxZ |
/// ---------------------------------------------------------
/// |             | BxB ... BxB |             |             |
/// |             |   .      .  |             |      *      |
/// |             |      .   .  |             |      *      |
/// |             |         BxB |             |             |
/// ---------------------------------------------------------
/// |             |             |             |             |
/// |             |             |    *        |      *      |
/// |             |             |        *    |      *      |
/// |             |             |             |             |
/// ---------------------------------------------------------
/// |             |             |             | ZxZ ... ZxZ |
/// |             |             |             |   .      .  |
/// |             |             |             |      .   .  |
/// |             |             |             |         ZxZ |
/// ---------------------------------------------------------
/// ```
///
/// It ia split into a grid of regions and each region is split into a grid of block matrices.
/// Within each region, all block matrices have the same shape. E.g., the first region contains only
/// AxA-shaped blocks, the second region contains only AxB blocks, etc.
#[derive(Debug)]
pub struct SymmetricBlockSparseMatrixBuilder {
    pub(crate) builder: BlockSparseMatrixBuilder,
}

impl SymmetricBlockSparseMatrixBuilder {
    /// Create a sparse block matrix "filled" with zeros.
    ///
    /// The shape of the block matrix is determined by the partition specs.
    pub fn zero(partitions: &[PartitionSpec]) -> Self {
        Self {
            builder: BlockSparseMatrixBuilder::zero(partitions, partitions),
        }
    }

    /// scalar dimension of the matrix.
    pub fn scalar_dimension(&self) -> usize {
        self.builder.scalar_shape()[0]
    }

    /// region grid dimension
    pub fn region_grid_dimension(&self) -> usize {
        self.builder.region_grid_shape()[0]
    }

    /// Add a block to the matrix.
    ///
    /// This is a += operation, i.e., the block is added to the existing block.
    ///
    /// Only upper triangular blocks are accepted; hence it shall hold that
    ///  * `grid_idx(0)    <= grid_idx(1)`,
    ///  * `block_index(0) <= block_index(1)`.
    ///
    /// In release mode, lower triangular blocks are ignored. In debug mode,
    /// this function will panic if the block is lower triangular.
    pub fn add_block(
        &mut self,
        region_idx: &[usize; 2],
        block_index: [usize; 2],
        block: &nalgebra::DMatrixView<f64>,
    ) {
        debug_assert_le!(region_idx[0], region_idx[1]);
        debug_assert!(
            // not a region on the diagonal
            region_idx[0] != region_idx[1]
            // or block on or above diagonal
            || block_index[0] <= block_index[1],
            "region [{}:{}], block [{},{}]",
            region_idx[0],
            region_idx[1],
            block_index[0],
            block_index[1]
        );
        self.builder.add_block(region_idx, block_index, block);
    }

    /// Convert to upper triangular scalar triplets.
    pub fn to_upper_triangular_scalar_triplets(&self) -> Vec<(usize, usize, f64)> {
        self.builder.to_scalar_triplets_impl(
            ToScalarTripletsImplMode::UpperTriangularFromUpperTriangularBlockPattern,
        )
    }

    /// Convert the block matrix to symmetric matrix in scalar triplets format.
    pub fn to_symmetric_scalar_triplets(&self) -> Vec<(usize, usize, f64)> {
        self.builder.to_scalar_triplets_impl(
            ToScalarTripletsImplMode::SymmetricFromUpperTriangularBlockPattern,
        )
    }

    /// Convert the upper triangular block sparse matrix to a symmetric dense matrix.
    pub fn to_symmetric_dense(&self) -> nalgebra::DMatrix<f64> {
        self.builder
            .to_dense_impl(ToDenseImplMode::SymmetricFromUpperTriangularBlockPattern)
    }
}
