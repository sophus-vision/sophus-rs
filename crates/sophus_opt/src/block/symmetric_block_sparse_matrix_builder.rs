use std::fmt::Debug;

use super::{
    grid::Grid,
    PartitionSpec,
};
use crate::debug_assert_le;

/// A builder for a symmetric block sparse matrix.
///
/// Internally, we chose the represent the symmetric matrix as a upper triangular matrix.
/// In particular, the target block sparse matrix is has the following structure:
///
/// ```ascii
/// ---------------------------------------------------------
/// | AxA ... AxA | AxB ... AxB |             | AxZ ... AxZ |
/// |   .         |  .       .  |             |  .       .  |
/// |      .      |  .       .  |   * * *     |  .       .  |
/// |         AxA | AxB ... AxB |             | AxZ ... AxZ |
/// ---------------------------------------------------------
/// |             | BxB ... BxB |             |             |
/// |             |   .      .  |             |      *      |
/// |             |      .   .  |             |      *      |
/// |             |         BxB |             |             |
/// ---------------------------------------------------------
/// |             |             |    *        |             |
/// |             |             |       *     |      *      |
/// |             |             |          *  |      *      |
/// |             |             |             |             |
/// ---------------------------------------------------------
/// |             |             |             | ZxZ ... ZxZ |
/// |             |             |             |   .      .  |
/// |             |             |             |      .   .  |
/// |             |             |             |         ZxZ |
/// ---------------------------------------------------------
/// ```
///
/// It ia split into a grid of regions and each region is split into a grid of NxM block matrices.:
/// Within each region, all block matrices have the same size. E.g., the first region contains only
/// AxA-shaped blocks, the second region contains only AxB blocks, etc.
#[derive(Debug)]
pub struct SymmetricBlockSparseMatrixBuilder {
    pub(crate) region_grid: Grid<BlockTripletRegion>,
    pub(crate) index_offset_per_segment: Vec<usize>,
    pub(crate) scalar_dimension: usize,
}

/// A single block "AxB" in the sparse upper triangular matrix.
#[derive(Debug, Clone)]
pub(crate) struct BlockTriplet {
    // index (row, column) of block within the region
    pub(crate) block_idx: [usize; 2],
    // index into flattened_block_storage
    pub(crate) start_data_idx: usize,
}

/// A homogeneous region in the block sparse matrix.
///
/// ```ascii
/// | AxB ... AxB |
/// |  .       .  |
/// |      .      |
/// | AxB ... AxB |
/// ```
///
/// It is represented by a list of index triplets, and a flattened storage of the blocks.
///
/// Note that the flattened storage in column-major order does not follow the macro shape of the
/// matrix (except by coincidence) but is purely based on the insertion order of the blocks /
/// add_block() calls.
#[derive(Debug, Clone)]
pub(crate) struct BlockTripletRegion {
    // Flattened storage of column-major matrix blocks
    pub(crate) flattened_block_storage: Vec<f64>,
    pub(crate) triplets: Vec<BlockTriplet>,
    // Dimensions (rows, columns) of each block
    pub(crate) shape: [usize; 2],
}

impl SymmetricBlockSparseMatrixBuilder {
    /// Create a zero block matrix.
    pub fn zero(partition: &[PartitionSpec]) -> Self {
        let mut block_dims = Vec::new();

        let mut index_offsets = Vec::new();
        let mut offset = 0;
        for segment in partition {
            index_offsets.push(offset);
            offset += segment.block_dim * segment.num_blocks;
            block_dims.push(segment.block_dim);
        }

        let mut region_grid = Grid::new(
            [block_dims.len(), block_dims.len()],
            BlockTripletRegion {
                flattened_block_storage: Vec::new(),
                triplets: Vec::new(),
                shape: [0, 0],
            },
        );

        for r in 0..block_dims.len() {
            for c in 0..block_dims.len() {
                region_grid.get_mut(&[r, c]).shape = [block_dims[r], block_dims[c]];
            }
        }

        Self {
            region_grid,
            index_offset_per_segment: index_offsets,
            scalar_dimension: offset,
        }
    }

    /// scalar dimension of the matrix.
    pub fn scalar_dimension(&self) -> usize {
        self.scalar_dimension
    }

    /// region grid dimension
    pub fn region_grid_dimension(&self) -> usize {
        self.index_offset_per_segment.len()
    }

    pub(crate) fn get_region(&self, grid_idx: &[usize; 2]) -> &BlockTripletRegion {
        self.region_grid.get(grid_idx)
    }

    pub(crate) fn get_region_mut(&mut self, grid_idx: &[usize; 2]) -> &mut BlockTripletRegion {
        self.region_grid.get_mut(grid_idx)
    }

    /// Add a block to the matrix.
    ///
    /// This is a += operation, i.e., the block is added to the existing block. d
    pub fn add_block(
        &mut self,
        grid_idx: &[usize; 2],
        block_index: [usize; 2],
        block: &nalgebra::DMatrixView<f64>,
    ) {
        let grid_region = self.get_region_mut(grid_idx);
        debug_assert_eq!(block.shape().0, grid_region.shape[0]);
        debug_assert_eq!(block.shape().1, grid_region.shape[1]);
        debug_assert_le!(grid_idx[0], grid_idx[1]);

        grid_region.triplets.push(BlockTriplet {
            block_idx: block_index,
            start_data_idx: grid_region.flattened_block_storage.len(),
        });
        for col in 0..block.ncols() {
            grid_region
                .flattened_block_storage
                .extend_from_slice(block.column(col).as_slice());
        }
    }

    /// Convert to upper triangular scalar triplets.
    pub fn to_upper_triangular_scalar_triplets(&self) -> Vec<(usize, usize, f64)> {
        let mut triplets = Vec::new();

        for region_x_idx in 0..self.region_grid_dimension() {
            for region_y_idx in 0..self.region_grid_dimension() {
                let region = self.get_region(&[region_x_idx, region_y_idx]);
                for block_triplet in &region.triplets {
                    let region_rows = region.shape[0];
                    let region_cols = region.shape[1];
                    let row_offset = self.index_offset_per_segment[region_x_idx]
                        + block_triplet.block_idx[0] * region_rows;
                    let col_offset = self.index_offset_per_segment[region_y_idx]
                        + block_triplet.block_idx[1] * region_cols;
                    let data_idx = block_triplet.start_data_idx;
                    let block = &region.flattened_block_storage
                        [data_idx..data_idx + (region_rows * region_cols)];

                    for c in 0..region_cols {
                        for r in 0..region_rows {
                            if region_x_idx == region_y_idx
                                && block_triplet.block_idx[0] == block_triplet.block_idx[1]
                                && r > c
                            {
                                continue;
                            }
                            triplets.push((
                                row_offset + r,
                                col_offset + c,
                                block[c * region.shape[0] + r],
                            ));
                        }
                    }
                }
            }
        }
        triplets
    }

    /// Convert the block matrix to symmetric matrix in scalar triplets format.
    pub fn to_symmetric_scalar_triplets(&self) -> Vec<(usize, usize, f64)> {
        let mut triplets = Vec::new();
        for region_x_idx in 0..self.region_grid_dimension() {
            for region_y_idx in 0..self.region_grid_dimension() {
                let region = self.get_region(&[region_x_idx, region_y_idx]);
                for block_triplet in &region.triplets {
                    let region_rows = region.shape[0];
                    let region_cols = region.shape[1];
                    let row_offset = self.index_offset_per_segment[region_x_idx]
                        + block_triplet.block_idx[0] * region_rows;
                    let col_offset = self.index_offset_per_segment[region_y_idx]
                        + block_triplet.block_idx[1] * region_cols;
                    let data_idx = block_triplet.start_data_idx;
                    let block = &region.flattened_block_storage
                        [data_idx..data_idx + (region_rows * region_cols)];

                    for c in 0..region_cols {
                        for r in 0..region_rows {
                            let value = block[c * region.shape[0] + r];
                            let scalar_r = row_offset + r;
                            let scalar_c = col_offset + c;
                            triplets.push((scalar_r, scalar_c, value));
                            if region_x_idx != region_y_idx
                                || block_triplet.block_idx[0] != block_triplet.block_idx[1]
                            {
                                triplets.push((scalar_c, scalar_r, value));
                            }
                        }
                    }
                }
            }
        }
        triplets
    }

    /// Convert the upper triangular block sparse matrix to a symmetric dense matrix.
    pub fn to_symmetric_dense(&self) -> nalgebra::DMatrix<f64> {
        let mut full_matrix =
            nalgebra::DMatrix::from_element(self.scalar_dimension(), self.scalar_dimension(), 0.0);

        for region_x_idx in 0..self.region_grid_dimension() {
            for region_y_idx in 0..self.region_grid_dimension() {
                let region = self.get_region(&[region_x_idx, region_y_idx]);
                for block_triplet in &region.triplets {
                    let region_rows = region.shape[0];
                    let region_cols = region.shape[1];
                    let row_offset = self.index_offset_per_segment[region_x_idx]
                        + block_triplet.block_idx[0] * region_rows;
                    let col_offset = self.index_offset_per_segment[region_y_idx]
                        + block_triplet.block_idx[1] * region_cols;
                    let data_idx = block_triplet.start_data_idx;
                    let block = &region.flattened_block_storage
                        [data_idx..data_idx + (region_rows * region_cols)];

                    for c in 0..region_cols {
                        for r in 0..region_rows {
                            let value = block[c * region.shape[0] + r];

                            full_matrix[(row_offset + r, col_offset + c)] += value;

                            if region_x_idx != region_y_idx
                                || block_triplet.block_idx[0] != block_triplet.block_idx[1]
                            {
                                full_matrix[(col_offset + c, row_offset + r)] += value;
                            }
                        }
                    }
                }
            }
        }

        full_matrix
    }
}
