pub(crate) mod csc_matrix;
pub(crate) mod faer_sparse_matrix;

pub use csc_matrix::*;
use nalgebra::DMatrixView;

use crate::{
    IsSymmetricMatrixBuilder,
    PartitionSpec,
    debug_assert_ge,
};

/// Builds a symmetric matrix from lower-block input and can export
/// LOWER-TRIANGLE triplets (i >= j) like `lower_triplets_from_dense`.
#[derive(Debug)]
pub struct SparseSymmetricMatrixBuilder {
    // store raw (i, j, x) as added — lower-only by construction
    triplets: Vec<(usize, usize, f64)>,
    // symmetric (row==col) partitions
    per_partition_scalar_offset: Vec<usize>, // cumulative scalar offsets (len = P)
    per_partition_block_dim: Vec<usize>,     // block dims per partition (len = P)
    per_partition_num_blocks: Vec<usize>,    // blocks count per partition (len = P)
    scalar_dimension: usize,
}

impl IsSymmetricMatrixBuilder for SparseSymmetricMatrixBuilder {
    type Matrix = LowerTripletsMatrix;

    fn zero(partitions: &[PartitionSpec]) -> Self {
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
        region_idx: &[usize; 2],  // (partition row, partition col)
        block_index: [usize; 2],  // (block row idx, block col idx) inside that region
        block: &DMatrixView<f64>, // dense block to add (+=)
    ) {
        debug_assert_ge!(region_idx[0], region_idx[1]); // lower by region
        let pr = region_idx[0];
        let pc = region_idx[1];

        // partition dims
        let mr = self.per_partition_block_dim[pr];
        let mc = self.per_partition_block_dim[pc];

        // shape check
        debug_assert_eq!((block.nrows(), block.ncols()), (mr, mc));

        // block bounds check
        debug_assert!(block_index[0] < self.per_partition_num_blocks[pr]);
        debug_assert!(block_index[1] < self.per_partition_num_blocks[pc]);

        // if diagonal region, enforce lower at block-index level
        debug_assert!(
            pr != pc || block_index[0] >= block_index[1],
            "region [{}:{}], block [{},{}] must be on or below diagonal",
            pr,
            pc,
            block_index[0],
            block_index[1]
        );

        // scalar offsets for this block's top-left
        let row_off = self.per_partition_scalar_offset[pr] + block_index[0] * mr;
        let col_off = self.per_partition_scalar_offset[pc] + block_index[1] * mc;

        let on_diag_region = pr == pc;
        let on_diag_block = on_diag_region && block_index[0] == block_index[1];
        let strictly_below = pr > pc || (on_diag_region && block_index[0] > block_index[1]);

        if strictly_below {
            // strictly-below blocks: add ALL entries (i >= j at scalar level is guaranteed)
            for c in 0..mc {
                for r in 0..mr {
                    let v = block[(r, c)];
                    if v != 0.0 {
                        let i = row_off + r;
                        let j = col_off + c;
                        debug_assert!(i >= j, "expected strictly-below block to yield i>=j");
                        self.triplets.push((i, j, v));
                    }
                }
            }
        } else {
            debug_assert!(on_diag_block, "non-below must be diagonal block");
            // diagonal block: only keep r >= c so global i >= j
            for c in 0..mc {
                for r in c..mr {
                    let v = block[(r, c)];
                    if v != 0.0 {
                        let i = row_off + r;
                        let j = col_off + c;
                        // i >= j holds because r >= c and row_off == col_off
                        self.triplets.push((i, j, v));
                    }
                }
            }
        }
    }

    /// Build a LOWER-ONLY CSC (optional; kept for parity with other builders).
    fn build(self) -> Self::Matrix {
        LowerTripletsMatrix {
            triplets: self.triplets,
            scalar_dimension: self.scalar_dimension,
        }
    }
}

impl SparseSymmetricMatrixBuilder {
    /// Export LOWER-TRIANGLE triplets `(n, ii, jj, xx)` exactly like
    /// `lower_triplets_from_dense()`: only entries with `i >= j`, no mirroring.
    pub fn into_lower_triplets(self) -> (usize, Vec<usize>, Vec<usize>, Vec<f64>) {
        let n = self.scalar_dimension;
        // Already lower-only; just split into arrays.
        let mut ii = Vec::with_capacity(self.triplets.len());
        let mut jj = Vec::with_capacity(self.triplets.len());
        let mut xx = Vec::with_capacity(self.triplets.len());
        for (i, j, x) in self.triplets {
            debug_assert!(i >= j);
            ii.push(i);
            jj.push(j);
            xx.push(x);
        }
        (n, ii, jj, xx)
    }
}
