use nalgebra::{
    DMatrix,
    DMatrixView,
};

use crate::{
    debug_assert_ge,
    matrix::PartitionSpec,
    prelude::*,
};

/// Builder for dense symmetric matrix.s
#[derive(Debug, Clone)]
pub struct DenseSymmetricMatrixBuilder {
    pub(crate) matrix: DMatrix<f64>,
    per_partition_scalar_offset: Vec<usize>,
    per_partition_block_dim: Vec<usize>,
    per_partition_num_blocks: Vec<usize>,
}

impl IsSymmetricMatrixBuilder for DenseSymmetricMatrixBuilder {
    type Matrix = DMatrix<f64>;

    fn zero(partitions: &[PartitionSpec]) -> Self {
        let mut per_partition_scalar_offset = Vec::with_capacity(partitions.len());
        let mut per_partition_block_dimensions = Vec::with_capacity(partitions.len());
        let mut per_partition_block_counts = Vec::with_capacity(partitions.len());

        let mut scalar_dimension = 0usize;
        for p in partitions {
            per_partition_scalar_offset.push(scalar_dimension);
            per_partition_block_dimensions.push(p.block_dimension);
            per_partition_block_counts.push(p.block_count);
            scalar_dimension += p.block_dimension * p.block_count;
        }

        Self {
            matrix: DMatrix::<f64>::zeros(scalar_dimension, scalar_dimension),
            per_partition_scalar_offset,
            per_partition_block_dim: per_partition_block_dimensions,
            per_partition_num_blocks: per_partition_block_counts,
        }
    }

    #[inline]
    fn scalar_dimension(&self) -> usize {
        self.matrix.nrows()
    }

    fn add_lower_block(
        &mut self,
        region_idx: &[usize; 2],
        block_index: [usize; 2],
        block: &DMatrixView<f64>,
    ) {
        debug_assert_ge!(region_idx[0], region_idx[1]); // lower (by region)
        let region_row_idx = region_idx[0];
        let region_col_idx = region_idx[1];

        let row_count_of_block = self.per_partition_block_dim[region_row_idx];
        let col_count_of_block = self.per_partition_block_dim[region_col_idx];

        debug_assert_eq!(
            (block.nrows(), block.ncols()),
            (row_count_of_block, col_count_of_block),
            "DenseSymmetricMatrixBuilder: block shape mismatch in region ({region_row_idx},{region_col_idx})"
        );

        debug_assert!(
            block_index[0] < self.per_partition_num_blocks[region_row_idx],
            "block row idx {} out of range for partition {} (num_blocks={})",
            block_index[0],
            region_row_idx,
            self.per_partition_num_blocks[region_row_idx]
        );
        debug_assert!(
            block_index[1] < self.per_partition_num_blocks[region_col_idx],
            "block col idx {} out of range for partition {} (num_blocks={})",
            block_index[1],
            region_col_idx,
            self.per_partition_num_blocks[region_col_idx]
        );

        debug_assert!(
            region_row_idx != region_col_idx || block_index[0] >= block_index[1],
            "region [{}:{}], block [{},{}] violates lower-triangular storage",
            region_row_idx,
            region_col_idx,
            block_index[0],
            block_index[1]
        );

        let scalar_row_offset =
            self.per_partition_scalar_offset[region_row_idx] + block_index[0] * row_count_of_block;
        let scalar_col_offset =
            self.per_partition_scalar_offset[region_col_idx] + block_index[1] * col_count_of_block;

        // Write the stored entry A[r, c] += block[r, c].
        for c in 0..col_count_of_block {
            for r in 0..row_count_of_block {
                self.matrix[(scalar_row_offset + r, scalar_col_offset + c)] += block[(r, c)];
            }
        }

        // If this block is strictly below (by region) or strictly below inside a diagonal
        // region, also add the symmetric mirror A[c, r] += block[r, c].
        let is_strictly_below_by_region = region_row_idx > region_col_idx;
        let is_strictly_below_inside_diag =
            region_row_idx == region_col_idx && block_index[0] > block_index[1];
        if is_strictly_below_by_region || is_strictly_below_inside_diag {
            for c in 0..col_count_of_block {
                for r in 0..row_count_of_block {
                    self.matrix[(scalar_col_offset + c, scalar_row_offset + r)] += block[(r, c)];
                }
            }
        }
    }

    fn build(self) -> Self::Matrix {
        self.matrix
    }
}

impl IsCompressibleMatrix for DMatrix<f64> {
    type Compressed = DMatrix<f64>;

    fn compress(&self) -> Self::Compressed {
        // This is a dense matrix. There is nothing to compress.
        self.clone()
    }
}
