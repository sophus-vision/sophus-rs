use std::fmt::Debug;

use super::{
    PartitionSpec,
    block_sparse_matrix::{
        BlockSparseMatrixBuilder,
        ToDenseImplMode,
        ToScalarTripletsImplMode,
    },
};
use crate::{
    BlockSparseCompressedMatrix,
    IsSymmetricMatrix,
    IsSymmetricMatrixBuilder,
    debug_assert_ge,
};

/// A builder for a symmetric block sparse matrix.
///
/// Internally, we chose to represent the symmetric matrix as a lower triangular matrix.
/// In particular, the target block sparse matrix has the following structure:
///
/// ```ascii
/// ---------------------------------------------------------
/// | AxA         |             |             |             |
/// |  .  .       |             |             |             |
/// |  .     .    |             |             |             |
/// | AxA ... AxA |             |             |             |
/// ---------------------------------------------------------
/// | BxA ... BxA | BxB         |             |             |
/// |  .       .  |  .  .       |             |             |
/// |  .       .  |  .     .    |             |             |
/// | BxA ... BxA | BxB ... BxB |             |             |
/// ---------------------------------------------------------
/// |             |             |             |             |
/// |      *      |             |    *        |             |
/// |      *      |             |        *    |             |
/// |             |             |             |             |
/// ---------------------------------------------------------
/// | ZxA ... ZxA |             |             | ZxZ         |
/// |  .       .  |             |             |  .  .       |
/// |  .       .  |   *  *  *   |   *  *  *   |  .     .    |
/// | ZxA ... ZxA |             |             | ZxZ ... ZxZ |
/// ---------------------------------------------------------
/// ```
///
/// It ia split into a grid of regions and each region is split into a grid of block matrices.
/// Within each region, all block matrices have the same shape. E.g., the first region contains only
/// AxA-shaped blocks, the second region contains only AxB blocks, etc.
#[derive(Debug)]
pub struct BlockSparseLowerMatrixBuilder {
    /// builder
    pub builder: BlockSparseMatrixBuilder,
}

impl BlockSparseLowerMatrixBuilder {
    /// Export UPPER triangular scalar triplets (view) from lower storage.
    pub fn to_upper_triangular_scalar_triplets(
        &self,
    ) -> Vec<faer::sparse::Triplet<usize, usize, f64>> {
        self.builder.to_scalar_triplets_impl(
            ToScalarTripletsImplMode::UpperTriViewFromLowerTriangularBlockPattern,
        )
    }

    /// Export full symmetric scalar triplets.
    pub fn to_symmetric_scalar_triplets(&self) -> Vec<faer::sparse::Triplet<usize, usize, f64>> {
        self.builder.to_scalar_triplets_impl(
            ToScalarTripletsImplMode::SymmetricFromLowerTriangularBlockPattern,
        )
    }

    /// r
    #[inline]
    pub fn region_grid_dimension(&self) -> usize {
        self.builder.region_grid_shape()[0]
    }

    /// Export full symmetric dense matrix.
    pub fn to_symmetric_dense(&self) -> nalgebra::DMatrix<f64> {
        self.builder
            .to_dense_impl(ToDenseImplMode::SymmetricFromLowerTriangularBlockPattern)
    }
}

impl IsSymmetricMatrix for BlockSparseLowerMatrixBuilder {
    type Compressed = BlockSparseCompressedMatrix;

    fn compress(&self) -> Self::Compressed {
        self.builder.to_compressed()
    }
}

impl IsSymmetricMatrixBuilder for BlockSparseLowerMatrixBuilder {
    type Matrix = BlockSparseLowerMatrixBuilder;

    fn zero(partitions: &[PartitionSpec]) -> Self {
        Self {
            builder: BlockSparseMatrixBuilder::zero(partitions, partitions),
        }
    }

    #[inline]
    fn scalar_dimension(&self) -> usize {
        self.builder.scalar_shape()[0]
    }

    fn add_lower_block(
        &mut self,
        region_idx: &[usize; 2],
        block_index: [usize; 2],
        block: &nalgebra::DMatrixView<f64>,
    ) {
        debug_assert_ge!(region_idx[0], region_idx[1]);
        debug_assert!(
            // not a region on the diagonal
            region_idx[0] != region_idx[1]
            // or block on or above diagonal
            || block_index[0] >= block_index[1],
            "region [{}:{}], block [{},{}]",
            region_idx[0],
            region_idx[1],
            block_index[0],
            block_index[1]
        );
        self.builder.add_block(region_idx, block_index, block);
    }

    fn build(self) -> Self::Matrix {
        self
    }
}
