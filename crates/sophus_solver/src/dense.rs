use nalgebra::{
    DMatrix,
    DMatrixView,
};

use crate::{
    IsSymmetricMatrixBuilder,
    PartitionSpec,
    debug_assert_ge,
    grid::Grid, // kept because trait signature brings it in
};

/// d
#[derive(Debug)]
pub struct DenseSymmetricMatrixBuilder {
    pub(crate) matrix: DMatrix<f64>,
    // symmetric case: same partitions for rows & cols
    per_partition_scalar_offset: Vec<usize>, // len = num_partitions, cumulative scalar offsets
    per_partition_block_dim: Vec<usize>,     // len = num_partitions, block dims
    per_partition_num_blocks: Vec<usize>,    // len = num_partitions, block counts (for debug)
}

impl IsSymmetricMatrixBuilder for DenseSymmetricMatrixBuilder {
    type Matrix = DMatrix<f64>;

    fn zero(partitions: &[PartitionSpec]) -> Self {
        // precompute per-partition scalar offsets and dims
        let mut per_partition_scalar_offset = Vec::with_capacity(partitions.len());
        let mut per_partition_block_dim = Vec::with_capacity(partitions.len());
        let mut per_partition_num_blocks = Vec::with_capacity(partitions.len());

        let mut scalar_dimension = 0usize;
        for p in partitions {
            per_partition_scalar_offset.push(scalar_dimension);
            per_partition_block_dim.push(p.block_dim);
            per_partition_num_blocks.push(p.num_blocks);
            scalar_dimension += p.block_dim * p.num_blocks;
        }

        Self {
            matrix: DMatrix::<f64>::zeros(scalar_dimension, scalar_dimension),
            per_partition_scalar_offset,
            per_partition_block_dim,
            per_partition_num_blocks,
        }
    }

    #[inline]
    fn scalar_dimension(&self) -> usize {
        self.matrix.nrows()
    }

    fn add_lower_block(
        &mut self,
        region_idx: &[usize; 2], // which Mx×Nx block type (partition row, partition col)
        block_index: [usize; 2], // which block inside that region (block row, block col)
        block: &DMatrixView<f64>, // the dense block to += into A
    ) {
        debug_assert_ge!(region_idx[0], region_idx[1]); // lower (by region)
        let pr = region_idx[0];
        let pc = region_idx[1];

        // partition shapes
        let mr = self.per_partition_block_dim[pr];
        let mc = self.per_partition_block_dim[pc];

        // basic shape checks
        debug_assert_eq!(
            (block.nrows(), block.ncols()),
            (mr, mc),
            "DenseSymmetricMatrixBuilder: block shape mismatch in region ({},{})",
            pr,
            pc
        );

        // block index checks (bounds)
        debug_assert!(
            block_index[0] < self.per_partition_num_blocks[pr],
            "block row idx {} out of range for partition {} (num_blocks={})",
            block_index[0],
            pr,
            self.per_partition_num_blocks[pr]
        );
        debug_assert!(
            block_index[1] < self.per_partition_num_blocks[pc],
            "block col idx {} out of range for partition {} (num_blocks={})",
            block_index[1],
            pc,
            self.per_partition_num_blocks[pc]
        );

        // lower-triangular rule inside diagonal region:
        // if region is diagonal (pr==pc), require block_row >= block_col
        debug_assert!(
            pr != pc || block_index[0] >= block_index[1],
            "region [{}:{}], block [{},{}] violates lower-triangular storage",
            pr,
            pc,
            block_index[0],
            block_index[1]
        );

        // scalar offsets of the top-left of this block
        let row_off = self.per_partition_scalar_offset[pr] + block_index[0] * mr;
        let col_off = self.per_partition_scalar_offset[pc] + block_index[1] * mc;

        // 1) write the stored entry A[r, c] += block[r, c]
        for c in 0..mc {
            for r in 0..mr {
                self.matrix[(row_off + r, col_off + c)] += block[(r, c)];
            }
        }

        // 2) if this block is strictly below (by region) or strictly below inside a diagonal
        //    region, also add the symmetric mirror A[c, r] += block[r, c].
        let strictly_below_by_region = pr > pc;
        let strictly_below_inside_diag = pr == pc && block_index[0] > block_index[1];

        if strictly_below_by_region || strictly_below_inside_diag {
            // mirror position starts at (col_off, row_off) and uses the transposed block
            for c in 0..mc {
                for r in 0..mr {
                    self.matrix[(col_off + c, row_off + r)] += block[(r, c)];
                }
            }
        }
        // note: diagonal blocks (block_index[0] == block_index[1] and pr==pc) are written once
    }

    fn build(self) -> Self::Matrix {
        self.matrix
    }
}
