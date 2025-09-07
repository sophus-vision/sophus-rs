use std::fmt::Debug;

use crate::{
    debug_assert_ge,
    matrix::{
        BlockSparseMatrixBuilder,
        IsSymmetricMatrixBuilder,
        PartitionSpec,
        block_sparse_triplets::BlockSparseTripletMatrix,
    },
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
#[derive(Debug, Clone)]
pub struct LowerBlockSparseMatrixBuilder {
    /// builder
    pub lower_triangular: BlockSparseMatrixBuilder,
}

impl LowerBlockSparseMatrixBuilder {
    /// r
    #[inline]
    pub fn region_grid_dimension(&self) -> usize {
        self.lower_triangular.triplets.region_grid_shape()[0]
    }
}

impl IsSymmetricMatrixBuilder for LowerBlockSparseMatrixBuilder {
    type Matrix = BlockSparseTripletMatrix;

    fn zero(partitions: &[PartitionSpec]) -> Self {
        Self {
            lower_triangular: BlockSparseMatrixBuilder::zero(partitions, partitions),
        }
    }

    #[inline]
    fn scalar_dimension(&self) -> usize {
        self.lower_triangular.triplets.scalar_shape()[0]
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
        self.lower_triangular
            .add_block(region_idx, block_index, block);
    }

    fn build(self) -> Self::Matrix {
        self.lower_triangular.triplets
    }
}
