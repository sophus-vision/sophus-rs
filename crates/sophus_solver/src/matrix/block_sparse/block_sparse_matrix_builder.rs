use std::fmt::Debug;

use crate::matrix::{
    PartitionBlockIndex,
    PartitionSet,
    block_sparse::{
        BlockSparseMatrix,
        BlockSparseTripletMatrix,
        BlockTriplet,
        BlockTripletRegion,
    },
    grid::Grid,
};

/// A builder for a block sparse `N x N` matrix.
///
/// The target block sparse matrix has the following structure:
///
/// ```ascii
/// -------------------------------------------
/// | AxA ... AxA | AxB ... AxB |             |
/// |  .  .    .  |  .       .  |             |
/// |  .     . .  |  .       .  |   *  *  *   |
/// | AxA ... AxA | AxA ... AxB |             |
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
///
/// It is split into a grid of regions and each region is split into a grid of block matrices.
/// Within each region, all block matrices have the same shape. E.g., the region (0,0) contains
/// only A×A-shaped blocks, the region (0,1) contains only (A×B) blocks, etc.
#[derive(Debug, Clone)]
pub struct BlockSparseMatrixBuilder {
    pub(crate) triplets: BlockSparseTripletMatrix,
}

impl BlockSparseMatrixBuilder {
    /// Create a sparse block `N x N` matrix "filled" with zeros.
    ///
    /// The shape of the block matrix is determined by the provided of partition set.
    pub fn zero(partitions: PartitionSet) -> Self {
        let mut block_dims = Vec::new();

        for partition in partitions.specs() {
            block_dims.push(partition.block_dim);
        }

        let mut region_grid = Grid::new(
            [block_dims.len(), block_dims.len()],
            BlockTripletRegion {
                flattened_block_storage: Vec::new(),
                triplets: Vec::new(),
                block_shape: [0, 0],
            },
        );

        for r in 0..block_dims.len() {
            for c in 0..block_dims.len() {
                region_grid.get_mut(&[r, c]).block_shape = [block_dims[r], block_dims[c]];
            }
        }

        Self {
            triplets: BlockSparseTripletMatrix {
                triplet_grid: region_grid,
                partitions,
            },
        }
    }

    /// Add a block to the matrix.
    ///
    /// This is a += operation, i.e., the block is added to the existing block.
    #[inline]
    pub fn add_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
        block: &nalgebra::DMatrixView<f64>,
    ) {
        let grid_region = self
            .triplets
            .get_region_mut(&[row_idx.partition, col_idx.partition]);
        debug_assert_eq!(block.shape().0, grid_region.block_shape[0]);
        debug_assert_eq!(block.shape().1, grid_region.block_shape[1]);
        grid_region.triplets.push(BlockTriplet {
            block_idx: [row_idx.block, col_idx.block],
            storage_base: grid_region.flattened_block_storage.len(),
        });
        for col in 0..block.ncols() {
            grid_region
                .flattened_block_storage
                .extend_from_slice(block.column(col).as_slice());
        }
    }

    /// Build and return the matrix.
    pub fn build(self) -> BlockSparseMatrix {
        self.triplets.to_compressed()
    }
}
