use std::fmt::Debug;

use super::PartitionSpec;
use crate::matrix::{
    BlockSparseTripletMatrix,
    BlockTriplet,
    BlockTripletRegion,
    grid::Grid,
};

/// A builder for a block sparse matrix.
///
/// The target block sparse matrix has the following structure:
///
/// ```ascii
/// --------------------------------------------------------------------------
/// | M1×N1 ... M1×N1 | M1×N2 ... M1×N2 |                 |  M1×Ny ... M1×Ny |
/// |   .  .      .   |   .         .   |                 |    .         .   |
/// |   .      .  .   |   .         .   |     *  *  *     |    .         .   |
/// | M1×N1 ... M1×N1 | M1×N2 ... M1×N2 |                 |  M1×Ny ... M1×Ny |
/// --------------------------------------------------------------------------
/// | M2×N1 ... M2×N1 | M2×N2 ... M2×N2 |                 |                  |
/// |   .         .   |   .  .      .   |                 |         *        |
/// |   .         .   |   .      .  .   |                 |         *        |
/// | M2×N1 ... M2×N1 | M2×N2 ... M2×N2 |                 |                  |
/// --------------------------------------------------------------------------
/// |                 |                 |                 |                  |
/// |        *        |                 |      *          |         *        |
/// |        *        |                 |          *      |         *        |
/// |                 |                 |                 |                  |
/// --------------------------------------------------------------------------
/// | Mx×N1 ... Mx×N1 |                 |                 |  Mx×Ny ... Mx×Ny |
/// |   .  .      .   |                 |                 |    . .       .   |
/// |   .      .  .   |     *  *  *     |     *  *  *     |    .      .  .   |
/// | Mx×N1 ... Mx×N1 |                 |                 |  Mx×Ny ... Mx×Ny |
/// --------------------------------------------------------------------------
/// ```
///
/// It ia split into a grid of regions and each region is split into a grid of block matrices.
/// Within each region, all block matrices have the same shape. E.g., the region (0,0) contains
/// only M1×N1-shaped blocks, the region (0,1) contains only (M1×N2) blocks, etc.
#[derive(Debug, Clone)]
pub struct BlockSparseMatrixBuilder {
    pub(crate) triplets: BlockSparseTripletMatrix,
}

#[derive(Debug, Clone)]
pub(crate) struct PartitionIndexOffsets {
    pub(crate) per_row_partition: Vec<usize>,
    pub(crate) per_col_partition: Vec<usize>,
}

impl BlockSparseMatrixBuilder {
    /// Create a sparse block matrix "filled" with zeros.
    ///
    /// The shape of the block matrix is determined by the provided of row and column partitions.
    pub fn zero(row_partitions: &[PartitionSpec], col_partitions: &[PartitionSpec]) -> Self {
        let mut row_block_dims = Vec::new();
        let mut col_block_dims = Vec::new();

        let mut row_index_offsets = Vec::new();
        let mut row_index_offset = 0;
        for row_partition in row_partitions {
            row_index_offsets.push(row_index_offset);
            row_index_offset += row_partition.block_dimension * row_partition.block_count;
            row_block_dims.push(row_partition.block_dimension);
        }

        let mut col_index_offsets = Vec::new();
        let mut col_index_offset = 0;
        for col_partition in col_partitions {
            col_index_offsets.push(col_index_offset);
            col_index_offset += col_partition.block_dimension * col_partition.block_count;
            col_block_dims.push(col_partition.block_dimension);
        }

        let mut region_grid = Grid::new(
            [row_block_dims.len(), col_block_dims.len()],
            BlockTripletRegion {
                flattened_block_storage: Vec::new(),
                triplets: Vec::new(),
                block_shape: [0, 0],
            },
        );

        for r in 0..row_block_dims.len() {
            for c in 0..col_block_dims.len() {
                region_grid.get_mut(&[r, c]).block_shape = [row_block_dims[r], col_block_dims[c]];
            }
        }

        Self {
            triplets: BlockSparseTripletMatrix {
                region_grid,
                index_offsets: PartitionIndexOffsets {
                    per_row_partition: row_index_offsets,
                    per_col_partition: col_index_offsets,
                },
                scalar_shape: [row_index_offset, col_index_offset],
                row_partitions: row_partitions.to_owned(),
                col_partitions: col_partitions.to_owned(),
            },
        }
    }

    /// Add a block to the matrix.
    ///
    /// This is a += operation, i.e., the block is added to the existing block.
    #[inline]
    pub fn add_block(
        &mut self,
        region_idx: &[usize; 2],
        block_index: [usize; 2],
        block: &nalgebra::DMatrixView<f64>,
    ) {
        let grid_region = self.triplets.get_region_mut(region_idx);
        debug_assert_eq!(block.shape().0, grid_region.block_shape[0]);
        debug_assert_eq!(block.shape().1, grid_region.block_shape[1]);
        grid_region.triplets.push(BlockTriplet {
            local_block_idx: block_index,
            storage_base: grid_region.flattened_block_storage.len(),
        });
        for col in 0..block.ncols() {
            grid_region
                .flattened_block_storage
                .extend_from_slice(block.column(col).as_slice());
        }
    }
}
