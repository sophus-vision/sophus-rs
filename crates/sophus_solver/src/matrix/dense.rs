use nalgebra::{
    DMatrix,
    DMatrixView,
    DMatrixViewMut,
};
use sophus_assert::debug_assert_ge;

use crate::{
    matrix::{
        IsSymmetricMatrix,
        PartitionBlockIndex,
        PartitionSet,
    },
    prelude::*,
};
/// Builder for a dense symmetric matrix.
#[derive(Debug, Clone)]
pub struct DenseSymmetricMatrixBuilder {
    pub(crate) matrix: DenseSymmetricMatrix,
    per_partition_block_dim: Vec<usize>,
    per_partition_num_blocks: Vec<usize>,
}

impl IsSymmetricMatrixBuilder for DenseSymmetricMatrixBuilder {
    type Matrix = DenseSymmetricMatrix;

    fn zero(partitions: PartitionSet) -> Self {
        let mut per_partition_block_dimensions = Vec::with_capacity(partitions.len());
        let mut per_partition_block_counts = Vec::with_capacity(partitions.len());

        let mut scalar_dimension = 0usize;
        for partition in partitions.specs() {
            per_partition_block_dimensions.push(partition.block_dim);
            per_partition_block_counts.push(partition.block_count);
            scalar_dimension += partition.block_dim * partition.block_count;
        }

        Self {
            matrix: DenseSymmetricMatrix::new(
                DMatrix::zeros(scalar_dimension, scalar_dimension),
                partitions,
            ),
            per_partition_block_dim: per_partition_block_dimensions,
            per_partition_num_blocks: per_partition_block_counts,
        }
    }

    #[inline]
    fn scalar_dim(&self) -> usize {
        self.matrix.data.nrows()
    }

    fn add_lower_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
        block: &DMatrixView<f64>,
    ) {
        let region_row_idx = row_idx.partition;
        let region_col_idx = col_idx.partition;
        debug_assert_ge!(region_row_idx, region_col_idx); // lower (by region)

        let row_count_of_block = self.per_partition_block_dim[region_row_idx];
        let col_count_of_block = self.per_partition_block_dim[region_col_idx];

        debug_assert_eq!(
            (block.nrows(), block.ncols()),
            (row_count_of_block, col_count_of_block),
            "DenseSymmetricMatrixBuilder: block shape mismatch in region ({region_row_idx},{region_col_idx})"
        );
        let block_row_idx = row_idx.block;
        let block_col_idx = col_idx.block;

        debug_assert!(
            block_row_idx < self.per_partition_num_blocks[region_row_idx],
            "block row idx {} out of range for partition {} (num_blocks={})",
            block_row_idx,
            region_row_idx,
            self.per_partition_num_blocks[region_row_idx]
        );
        debug_assert!(
            block_col_idx < self.per_partition_num_blocks[region_col_idx],
            "block col idx {} out of range for partition {} (num_blocks={})",
            block_col_idx,
            region_col_idx,
            self.per_partition_num_blocks[region_col_idx]
        );

        debug_assert!(
            region_row_idx != region_col_idx || block_row_idx >= block_col_idx,
            "region [{}:{}], block [{},{}] violates lower-triangular storage",
            region_row_idx,
            region_col_idx,
            block_row_idx,
            block_col_idx
        );

        let scalar_row_offset = self.matrix.partitions().scalar_offsets_by_partition()
            [region_row_idx]
            + block_row_idx * row_count_of_block;
        let scalar_col_offset = self.matrix.partitions().scalar_offsets_by_partition()
            [region_col_idx]
            + block_col_idx * col_count_of_block;

        // Write the stored entry A[r, c] += block[r, c].
        for c in 0..col_count_of_block {
            for r in 0..row_count_of_block {
                self.matrix.data[(scalar_row_offset + r, scalar_col_offset + c)] += block[(r, c)];
            }
        }

        // If this block is strictly below (by region) or strictly below inside a diagonal
        // region, also add the symmetric mirror A[c, r] += block[r, c].
        let is_strictly_below_by_region = region_row_idx > region_col_idx;
        let is_strictly_below_inside_diag =
            region_row_idx == region_col_idx && block_row_idx > block_col_idx;
        if is_strictly_below_by_region || is_strictly_below_inside_diag {
            for c in 0..col_count_of_block {
                for r in 0..row_count_of_block {
                    self.matrix.data[(scalar_col_offset + c, scalar_row_offset + r)] +=
                        block[(r, c)];
                }
            }
        }
    }

    fn build(self) -> Self::Matrix {
        self.matrix
    }

    fn partitions(&self) -> &PartitionSet {
        &self.matrix.partitions
    }
}

/// Dense symmetric matrix stored as a full `DMatrix<f64>`.
#[derive(Debug, Clone)]
pub struct DenseSymmetricMatrix {
    data: DMatrix<f64>,
    partitions: PartitionSet,
}

impl DenseSymmetricMatrix {
    pub(crate) fn new(symmetric_mat: DMatrix<f64>, partitions: PartitionSet) -> Self {
        DenseSymmetricMatrix {
            data: symmetric_mat,
            partitions,
        }
    }

    /// Returns an immutable view of the underlying dense matrix.
    #[inline]
    pub fn view(&self) -> DMatrixView<'_, f64> {
        self.data.as_view()
    }

    /// Returns a mutable view of the underlying dense matrix.
    #[inline]
    pub fn view_mut(&mut self) -> DMatrixViewMut<'_, f64> {
        self.data.as_view_mut()
    }

    /// Returns the block partition set.
    #[inline]
    pub fn partitions(&self) -> &PartitionSet {
        &self.partitions
    }

    /// Returns the scalar (row/column) dimension of the matrix.
    #[inline]
    pub fn scalar_dimension(&self) -> usize {
        self.data.nrows()
    }
}

impl IsSymmetricMatrix for DenseSymmetricMatrix {
    #[inline]
    fn get_block(
        &self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> nalgebra::DMatrix<f64> {
        let row = self.block_range(row_idx);
        let col = self.block_range(col_idx);

        self.data
            .view(
                (row.start_idx, col.start_idx),
                (row.block_dim, col.block_dim),
            )
            .into()
    }

    #[inline]
    fn partitions(&self) -> &PartitionSet {
        &self.partitions
    }

    fn to_dense(&self) -> DMatrix<f64> {
        self.data.clone()
    }
}
