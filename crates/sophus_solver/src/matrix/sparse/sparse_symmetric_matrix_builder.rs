use nalgebra::DMatrixView;
use sophus_assert::debug_assert_ge;

use crate::matrix::{
    IsSymmetricMatrixBuilder,
    PartitionBlockIndex,
    PartitionSet,
    sparse::{
        FaerSparseMatrix,
        FaerSparseSymmetricMatrix,
        SparseMatrix,
        TripletMatrix,
        sparse_symmetric_matrix::SparseSymmetricMatrix,
    },
};

/// Builder for a sparse symmetric matrix.
#[derive(Debug, Clone)]
pub struct SparseSymmetricMatrixBuilder {
    triplets: Vec<(usize, usize, f64)>,
    per_partition_block_dim: Vec<usize>,
    per_partition_num_blocks: Vec<usize>,
    scalar_dimension: usize,
    partitions: PartitionSet,
}

impl SparseSymmetricMatrixBuilder {
    /// Into [FaerSparseMatrix].
    pub fn into_faer(self) -> FaerSparseMatrix {
        let lower_triplets = TripletMatrix::new(self.triplets, self.scalar_dimension);
        FaerSparseMatrix::new(&lower_triplets, self.partitions)
    }

    /// Into [FaerSparseSymmetricMatrix].
    pub fn into_faer_symmetric(self) -> FaerSparseSymmetricMatrix {
        let lower_triplets = TripletMatrix::new(self.triplets, self.scalar_dimension);
        FaerSparseSymmetricMatrix::new(&lower_triplets, self.partitions)
    }
}

impl IsSymmetricMatrixBuilder for SparseSymmetricMatrixBuilder {
    type Matrix = SparseSymmetricMatrix;

    fn zero(partitions: PartitionSet) -> Self {
        let mut per_partition_block_dim = Vec::with_capacity(partitions.len());
        let mut per_partition_num_blocks = Vec::with_capacity(partitions.len());

        let mut scalar_dimension = 0usize;
        for p in partitions.specs() {
            per_partition_block_dim.push(p.block_dim);
            per_partition_num_blocks.push(p.block_count);
            scalar_dimension += p.block_dim * p.block_count;
        }

        Self {
            triplets: Vec::new(),
            per_partition_block_dim,
            per_partition_num_blocks,
            scalar_dimension,
            partitions,
        }
    }

    fn partitions(&self) -> &PartitionSet {
        &self.partitions
    }

    #[inline]
    fn scalar_dim(&self) -> usize {
        self.scalar_dimension
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
            "block shape check"
        );
        let block_row_idx = row_idx.block;
        let block_col_idx = col_idx.block;

        debug_assert!(
            block_row_idx < self.per_partition_num_blocks[region_row_idx],
            "block row bounds check"
        );
        debug_assert!(
            block_col_idx < self.per_partition_num_blocks[region_col_idx],
            "block_col_bounds_check"
        );

        debug_assert!(
            region_row_idx != region_col_idx || block_row_idx >= block_col_idx,
            "region [{}:{}], block [{},{}] must be on or below diagonal",
            region_row_idx,
            region_col_idx,
            block_row_idx,
            block_col_idx
        );

        let scalar_row_offset = self.partitions.scalar_offsets_by_partition()[region_row_idx]
            + block_row_idx * row_count_of_block;
        let scalar_col_offset = self.partitions.scalar_offsets_by_partition()[region_col_idx]
            + block_col_idx * col_count_of_block;

        let is_on_diag_region = region_row_idx == region_col_idx;
        let is_on_diag_block = is_on_diag_region && block_row_idx == block_col_idx;
        let is_strictly_below =
            region_row_idx > region_col_idx || (is_on_diag_region && block_row_idx > block_col_idx);

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
        let lower =
            SparseMatrix::from_triplets(&TripletMatrix::new(self.triplets, self.scalar_dimension));
        SparseSymmetricMatrix::new(lower, self.partitions)
    }
}
