use std::fmt::Debug;

use sophus_assert::debug_assert_ge;

use crate::matrix::{
    IsSymmetricMatrixBuilder,
    PartitionBlockIndex,
    PartitionSet,
    block_sparse::{
        BlockSparseMatrixBuilder,
        block_sparse_matrix_pattern::{
            BlockSparseMatrixPattern,
            BlockSparseSymbolicBuilder,
        },
        block_sparse_symmetric_matrix::BlockSparseSymmetricMatrix,
    },
};

/// Symbolic builder for a symmetric block-sparse matrix.
///
/// Records which lower-triangular blocks will be written during numeric passes,
/// without storing any values.  Call [`BlockSparseSymmetricSymbolicBuilder::into_pattern`]
/// to obtain a [`BlockSparseSymmetricMatrixPattern`] reusable across optimizer iterations.
pub struct BlockSparseSymmetricSymbolicBuilder {
    inner: BlockSparseSymbolicBuilder,
}

impl BlockSparseSymmetricSymbolicBuilder {
    /// Create a new symbolic builder for the given partition set.
    pub fn new(partitions: PartitionSet) -> Self {
        Self {
            inner: BlockSparseSymbolicBuilder::new(partitions),
        }
    }

    /// Record that the lower-triangular block `(row_idx, col_idx)` will be accumulated.
    ///
    /// Requires `row_idx.partition >= col_idx.partition`.
    #[inline]
    pub fn add_lower_block(&mut self, row_idx: PartitionBlockIndex, col_idx: PartitionBlockIndex) {
        debug_assert_ge!(row_idx.partition, col_idx.partition);
        self.inner.add_block(row_idx, col_idx);
    }

    /// Finalise the symbolic pass and return a reusable pattern.
    pub fn into_pattern(self) -> BlockSparseSymmetricMatrixPattern {
        BlockSparseSymmetricMatrixPattern {
            inner: self.inner.into_pattern(),
            workers: Vec::new(),
        }
    }
}

/// Precomputed sparsity pattern for a symmetric block-sparse matrix.
///
/// Wraps [`BlockSparseMatrixPattern`].  Call [`BlockSparseSymmetricMatrixPattern::reset`]
/// at the start of each optimizer iteration, accumulate with
/// [`BlockSparseSymmetricMatrixPattern::add_lower_block`], then call
/// [`BlockSparseSymmetricMatrixPattern::build`].
#[derive(Debug)]
pub struct BlockSparseSymmetricMatrixPattern {
    pub(crate) inner: BlockSparseMatrixPattern,
    /// Pre-allocated worker patterns for parallel Hessian populate.
    ///
    /// Stored here so they survive across optimizer iterations and avoid
    /// repeated mmap allocations (which cause minor page faults on Linux).
    /// These are NOT included in `Clone` — workers are an internal cache.
    pub(crate) workers: Vec<BlockSparseSymmetricMatrixPattern>,
}

impl Clone for BlockSparseSymmetricMatrixPattern {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            workers: Vec::new(), // don't propagate the worker cache when cloning
        }
    }
}

impl BlockSparseSymmetricMatrixPattern {
    /// Zero all value storage — call once per optimizer iteration.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Accumulate `block` into the lower-triangular position `(row_idx, col_idx)`.
    #[inline]
    pub fn add_lower_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
        block: &nalgebra::DMatrixView<f64>,
    ) {
        debug_assert_ge!(row_idx.partition, col_idx.partition);
        self.inner.add_block(row_idx, col_idx, block);
    }

    /// Assemble and return the symmetric matrix for this iteration.
    pub fn build(&self) -> BlockSparseSymmetricMatrix {
        BlockSparseSymmetricMatrix {
            lower: self.inner.build(),
        }
    }

    /// Number of partitions.
    pub fn partition_count(&self) -> usize {
        self.inner.partition_count()
    }

    /// Merge another pattern's accumulated values into this one (`self += other`).
    ///
    /// Used to reduce thread-local accumulators after a parallel H-assembly pass.
    /// Panics in debug mode if the storage sizes differ.
    pub fn merge_from(&mut self, other: &Self) {
        self.inner.merge_from(&other.inner);
    }

    /// Scalar dimension.
    pub fn scalar_dim(&self) -> usize {
        self.inner.subdivision.scalar_dim()
    }

    /// Partition set.
    pub fn partitions(&self) -> &PartitionSet {
        self.inner.subdivision.partitions()
    }

    /// Ensure at least `n` pre-allocated worker patterns exist.
    ///
    /// Creates new workers by cloning the pattern structure on the first call.
    /// Resets are deferred to the parallel populate loop so they happen concurrently
    /// with the accumulation work instead of sequentially up-front.
    pub fn ensure_workers(&mut self, n: usize) {
        while self.workers.len() < n {
            // Clone structure (self was already reset by from_block_sparse_pattern,
            // so new workers start zeroed).
            self.workers.push(self.clone());
        }
    }

    /// Take ownership of the worker pool for parallel use.
    ///
    /// Must be followed by [`return_workers`] to restore them for the next iteration.
    pub fn take_workers(&mut self) -> Vec<Self> {
        std::mem::take(&mut self.workers)
    }

    /// Return the worker pool after parallel use.
    pub fn return_workers(&mut self, workers: Vec<Self>) {
        self.workers = workers;
    }
}

/// A builder for a symmetric block sparse matrix (triplet-list, non-pattern-based).
///
/// Use [`BlockSparseSymmetricSymbolicBuilder`] + [`BlockSparseSymmetricMatrixPattern`]
/// instead when the same problem structure is solved repeatedly across optimizer
/// iterations.
///
/// Internally, the symmetric matrix is represented as a lower block-triangular matrix.
///
/// ```ascii
/// -------------------------------------------
/// | AxA         |             |             |
/// |  .  .       |             |             |
/// |  .     .    |             |             |
/// | AxA ... AxA |             |             |
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
#[derive(Debug, Clone)]
pub struct BlockSparseSymmetricMatrixBuilder {
    lower_triangular: BlockSparseMatrixBuilder,
}

impl BlockSparseSymmetricMatrixBuilder {
    /// Number of partitions horizontally (or vertically).
    #[inline]
    pub fn partition_count(&self) -> usize {
        self.lower_triangular.triplets.partition_count()
    }
}

impl IsSymmetricMatrixBuilder for BlockSparseSymmetricMatrixBuilder {
    type Matrix = BlockSparseSymmetricMatrix;

    fn zero(partitions: PartitionSet) -> Self {
        Self {
            lower_triangular: BlockSparseMatrixBuilder::zero(partitions),
        }
    }

    #[inline]
    fn scalar_dim(&self) -> usize {
        self.lower_triangular.triplets.scalar_dimension()
    }

    fn partitions(&self) -> &PartitionSet {
        &self.lower_triangular.triplets.partitions
    }

    fn add_lower_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
        block: &nalgebra::DMatrixView<f64>,
    ) {
        debug_assert_ge!(row_idx.partition, col_idx.partition);

        self.lower_triangular.add_block(row_idx, col_idx, block);
    }

    fn build(self) -> Self::Matrix {
        BlockSparseSymmetricMatrix {
            lower: self.lower_triangular.triplets.to_compressed(),
        }
    }
}
