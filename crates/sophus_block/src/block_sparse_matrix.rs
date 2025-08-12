use std::fmt::Debug;

use super::{
    PartitionSpec,
    grid::Grid,
};
use crate::{
    CompressedBlockMatrix,
    CompressedBlockRegion,
    SymbolicBlockPattern,
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
#[derive(Debug)]
pub struct BlockSparseMatrixBuilder {
    pub(crate) region_grid: Grid<BlockTripletRegion>,
    pub(crate) index_offsets: PartitionIndexOffsets,
    pub(crate) scalar_shape: [usize; 2],
}

#[derive(Debug, Clone)]
pub(crate) struct PartitionIndexOffsets {
    pub(crate) per_row_partition: Vec<usize>,
    pub(crate) per_col_partition: Vec<usize>,
}

/// A single block "AxB" in a region of the block sparse matrix.
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
/// | M1×N1 ... M1×N1 |
/// |   .  .      .   |
/// |   .      .  .   |
/// | M1×N1 ... M1×N1 |
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
    pub(crate) block_shape: [usize; 2],
}

pub(crate) enum ToScalarTripletsImplMode {
    // input: lower triangular block sparse matrix
    // output: symmetric matrix in scalar triplet representation
    SymmetricFromLowerTriangularBlockPattern,
    // input: lower triangular block sparse matrix
    // output: upper triangular matrix in scalar triplet representation
    UpperTriViewFromLowerTriangularBlockPattern,
    // input: general block sparse matrix
    // output: general matrix in scalar triplet representation
    General,
}

pub(crate) enum ToDenseImplMode {
    // input: lower triangular block sparse matrix
    // output: symmetric dense matrix
    SymmetricFromLowerTriangularBlockPattern,
    // input: general block sparse matrix
    // output: general dense matrix
    General,
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
            row_index_offset += row_partition.block_dim * row_partition.num_blocks;
            row_block_dims.push(row_partition.block_dim);
        }

        let mut col_index_offsets = Vec::new();
        let mut col_index_offset = 0;
        for col_partition in col_partitions {
            col_index_offsets.push(col_index_offset);
            col_index_offset += col_partition.block_dim * col_partition.num_blocks;
            col_block_dims.push(col_partition.block_dim);
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
            region_grid,
            index_offsets: PartitionIndexOffsets {
                per_row_partition: row_index_offsets,
                per_col_partition: col_index_offsets,
            },
            scalar_shape: [row_index_offset, col_index_offset],
        }
    }

    /// scalar dimension of the matrix.
    #[inline]
    pub fn scalar_shape(&self) -> [usize; 2] {
        self.scalar_shape
    }

    /// region grid dimension
    #[inline]
    pub fn region_grid_shape(&self) -> [usize; 2] {
        [
            self.index_offsets.per_row_partition.len(),
            self.index_offsets.per_col_partition.len(),
        ]
    }

    #[inline]
    pub(crate) fn get_region(&self, region_idx: &[usize; 2]) -> &BlockTripletRegion {
        self.region_grid.get(region_idx)
    }
    #[inline]
    pub(crate) fn get_region_mut(&mut self, region_idx: &[usize; 2]) -> &mut BlockTripletRegion {
        self.region_grid.get_mut(region_idx)
    }

    /// Add a block to the matrix.
    ///
    /// This is a += operation, i.e., the block is added to the existing block. d
    pub fn add_block(
        &mut self,
        region_idx: &[usize; 2],
        block_index: [usize; 2],
        block: &nalgebra::DMatrixView<f64>,
    ) {
        let grid_region = self.get_region_mut(region_idx);
        debug_assert_eq!(block.shape().0, grid_region.block_shape[0]);
        debug_assert_eq!(block.shape().1, grid_region.block_shape[1]);
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

    pub(crate) fn to_scalar_triplets_impl(
        &self,
        mode: ToScalarTripletsImplMode,
    ) -> Vec<faer::sparse::Triplet<usize, usize, f64>> {
        let mut triplets = Vec::new();

        for region_x_idx in 0..self.region_grid_shape()[0] {
            for region_y_idx in 0..self.region_grid_shape()[1] {
                let region = self.get_region(&[region_x_idx, region_y_idx]);
                let [region_rows, region_cols] = region.block_shape;

                for block_triplet in &region.triplets {
                    let row_offset = self.index_offsets.per_row_partition[region_x_idx]
                        + block_triplet.block_idx[0] * region_rows;
                    let col_offset = self.index_offsets.per_col_partition[region_y_idx]
                        + block_triplet.block_idx[1] * region_cols;
                    let data_idx = block_triplet.start_data_idx;
                    let block = &region.flattened_block_storage
                        [data_idx..data_idx + (region_rows * region_cols)];

                    let on_diag_region = region_x_idx == region_y_idx;
                    let on_diag_block =
                        on_diag_region && block_triplet.block_idx[0] == block_triplet.block_idx[1];

                    for c in 0..region_cols {
                        for r in 0..region_rows {
                            let value = block[c * region_rows + r];
                            let scalar_r = row_offset + r;
                            let scalar_c = col_offset + c;

                            match mode {
                                ToScalarTripletsImplMode::General => {
                                    triplets.push(faer::sparse::Triplet::new(scalar_r, scalar_c, value));
                                }
                                ToScalarTripletsImplMode::SymmetricFromLowerTriangularBlockPattern => {
                                    // Storage assumed lower by block pattern.
                                    // Emit A(i,j); if strictly below at block level, also emit mirror.
                                    triplets.push(faer::sparse::Triplet::new(scalar_r, scalar_c, value));

                                    let is_strictly_below_region = region_x_idx > region_y_idx;
                                    let is_strictly_below_inside_diag =
                                        on_diag_region && (block_triplet.block_idx[0] > block_triplet.block_idx[1]);

                                    if is_strictly_below_region || is_strictly_below_inside_diag {
                                        // Mirror
                                        triplets.push(faer::sparse::Triplet::new(scalar_c, scalar_r, value));
                                    }
                                }
                                ToScalarTripletsImplMode::UpperTriViewFromLowerTriangularBlockPattern => {
                                    // Emit ONLY upper-triangle view:
                                    // - For strictly-below blocks, emit their mirror (j,i).
                                    // - For diagonal blocks, emit entries with i <= j.
                                    // - For strictly-above (shouldn't exist under lower storage), ignore.
                                    let is_strictly_below_region = region_x_idx > region_y_idx;
                                    let is_strictly_below_inside_diag =
                                        on_diag_region && (block_triplet.block_idx[0] > block_triplet.block_idx[1]);

                                    if is_strictly_below_region || is_strictly_below_inside_diag {
                                        // Mirror to upper
                                        triplets.push(faer::sparse::Triplet::new(scalar_c, scalar_r, value));
                                    } else if on_diag_block {
                                        // Keep only (i <= j) scalars of the block
                                        if scalar_r <= scalar_c {
                                            triplets.push(faer::sparse::Triplet::new(scalar_r, scalar_c, value));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        triplets
    }

    /// Convert to scalar triplets.
    pub fn to_scalar_triplets(&self) -> Vec<faer::sparse::Triplet<usize, usize, f64>> {
        self.to_scalar_triplets_impl(ToScalarTripletsImplMode::General)
    }

    pub(crate) fn to_dense_impl(&self, mode: ToDenseImplMode) -> nalgebra::DMatrix<f64> {
        let mut full_matrix =
            nalgebra::DMatrix::from_element(self.scalar_shape()[0], self.scalar_shape()[1], 0.0);

        for region_x_idx in 0..self.region_grid_shape()[0] {
            for region_y_idx in 0..self.region_grid_shape()[1] {
                let region = self.get_region(&[region_x_idx, region_y_idx]);
                let [region_rows, region_cols] = region.block_shape;

                for block_triplet in &region.triplets {
                    let row_offset = self.index_offsets.per_row_partition[region_x_idx]
                        + block_triplet.block_idx[0] * region_rows;
                    let col_offset = self.index_offsets.per_col_partition[region_y_idx]
                        + block_triplet.block_idx[1] * region_cols;
                    let data_idx = block_triplet.start_data_idx;
                    let block = &region.flattened_block_storage
                        [data_idx..data_idx + (region_rows * region_cols)];
                    let on_diag_region = region_x_idx == region_y_idx;
                    let is_strictly_below = region_x_idx > region_y_idx
                        || (on_diag_region
                            && block_triplet.block_idx[0] > block_triplet.block_idx[1]);

                    for c in 0..region_cols {
                        for r in 0..region_rows {
                            let value = block[c * region_rows + r];
                            let scalar_r = row_offset + r;
                            let scalar_c = col_offset + c;

                            // always add the stored entry
                            full_matrix[(scalar_r, scalar_c)] += value;

                            if let ToDenseImplMode::SymmetricFromLowerTriangularBlockPattern = mode
                                && is_strictly_below
                            {
                                full_matrix[(scalar_c, scalar_r)] += value;
                            }
                        }
                    }
                }
            }
        }

        full_matrix
    }

    /// Convert the block sparse matrix to dense matrix.
    pub fn to_dense(&self) -> nalgebra::DMatrix<f64> {
        self.to_dense_impl(ToDenseImplMode::General)
    }

    /// Convert to compressed block form.
    pub fn to_compressed(&self) -> CompressedBlockMatrix {
        let rows_of_regions = self.index_offsets.per_row_partition.len();
        let cols_of_regions = self.index_offsets.per_col_partition.len();

        let mut region_grid = Grid::new(
            [rows_of_regions, cols_of_regions],
            CompressedBlockRegion::empty(),
        );

        // compress each region
        for region_x_idx in 0..rows_of_regions {
            for region_y_idx in 0..cols_of_regions {
                let region_idx = [region_x_idx, region_y_idx];
                *region_grid.get_mut(&region_idx) =
                    CompressedBlockRegion::from_block_sparse_matrix(self, region_idx);
            }
        }

        CompressedBlockMatrix {
            sym_block_pattern: SymbolicBlockPattern::from_regions(
                &self.index_offsets,
                &region_grid,
            ),
            region_grid,
            index_offsets: self.index_offsets.clone(),
        }
    }
}
