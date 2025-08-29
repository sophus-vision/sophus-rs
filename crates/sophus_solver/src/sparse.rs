pub(crate) mod csc_matrix;
pub(crate) mod faer_sparse_matrix;
pub(crate) mod triplet_matrix;

pub use csc_matrix::*;
pub use faer_sparse_matrix::*;
use nalgebra::DMatrixView;
pub use triplet_matrix::*;

use crate::{
    PartitionSpec,
    debug_assert_ge,
    prelude::*,
};

/// Builder for sparse symmetric matrix.s
#[derive(Debug)]
pub struct SparseSymmetricMatrixBuilder {
    triplets: Vec<(usize, usize, f64)>,
    per_partition_scalar_offset: Vec<usize>,
    per_partition_block_dim: Vec<usize>,
    per_partition_num_blocks: Vec<usize>,
    scalar_dimension: usize,
}

impl IsSymmetricMatrixBuilder for SparseSymmetricMatrixBuilder {
    type Matrix = TripletMatrix;

    fn zero(partitions: &[PartitionSpec]) -> Self {
        let mut per_partition_scalar_offset = Vec::with_capacity(partitions.len());
        let mut per_partition_block_dim = Vec::with_capacity(partitions.len());
        let mut per_partition_num_blocks = Vec::with_capacity(partitions.len());

        let mut scalar_dimension = 0usize;
        for p in partitions {
            per_partition_scalar_offset.push(scalar_dimension);
            per_partition_block_dim.push(p.block_dimension);
            per_partition_num_blocks.push(p.block_count);
            scalar_dimension += p.block_dimension * p.block_count;
        }

        Self {
            triplets: Vec::new(),
            per_partition_scalar_offset,
            per_partition_block_dim,
            per_partition_num_blocks,
            scalar_dimension,
        }
    }

    #[inline]
    fn scalar_dimension(&self) -> usize {
        self.scalar_dimension
    }

    fn add_lower_block(
        &mut self,
        region_idx: &[usize; 2],
        block_index: [usize; 2],
        block: &DMatrixView<f64>,
    ) {
        debug_assert_ge!(
            region_idx[0],
            region_idx[1],
            "must be added to lower triangular region"
        );
        let region_row_idx = region_idx[0];
        let region_col_idx = region_idx[1];

        let row_count_of_block = self.per_partition_block_dim[region_row_idx];
        let col_count_of_block = self.per_partition_block_dim[region_col_idx];

        debug_assert_eq!(
            (block.nrows(), block.ncols()),
            (row_count_of_block, col_count_of_block),
            "block shape check"
        );

        debug_assert!(
            block_index[0] < self.per_partition_num_blocks[region_row_idx],
            "block row bounds check"
        );
        debug_assert!(
            block_index[1] < self.per_partition_num_blocks[region_col_idx],
            "block_col_bounds_check"
        );

        debug_assert!(
            region_row_idx != region_col_idx || block_index[0] >= block_index[1],
            "region [{}:{}], block [{},{}] must be on or below diagonal",
            region_row_idx,
            region_col_idx,
            block_index[0],
            block_index[1]
        );

        let scalar_row_offset =
            self.per_partition_scalar_offset[region_row_idx] + block_index[0] * row_count_of_block;
        let scalar_col_offset =
            self.per_partition_scalar_offset[region_col_idx] + block_index[1] * col_count_of_block;

        let is_on_diag_region = region_row_idx == region_col_idx;
        let is_on_diag_block = is_on_diag_region && block_index[0] == block_index[1];
        let is_strictly_below = region_row_idx > region_col_idx
            || (is_on_diag_region && block_index[0] > block_index[1]);

        if is_strictly_below {
            // strictly-below blocks: add all entries (i >= j at scalar level is guaranteed)
            for c in 0..col_count_of_block {
                for r in 0..row_count_of_block {
                    let v = block[(r, c)];
                    if v != 0.0 {
                        let i = scalar_row_offset + r;
                        let j = scalar_col_offset + c;
                        debug_assert!(i >= j, "expected strictly-below block to yield i>=j");
                        self.triplets.push((i, j, v));
                    }
                }
            }
        } else {
            debug_assert!(is_on_diag_block, "non-below must be diagonal block");
            // diagonal block: only keep r >= c so globally i >= j
            for c in 0..col_count_of_block {
                for r in c..row_count_of_block {
                    let v = block[(r, c)];
                    if v != 0.0 {
                        let i = scalar_row_offset + r;
                        let j = scalar_col_offset + c;
                        self.triplets.push((i, j, v));
                    }
                }
            }
        }
    }

    fn build(self) -> Self::Matrix {
        TripletMatrix::new(self.triplets, self.scalar_dimension, self.scalar_dimension)
    }
}
