/// Block matrix structures.
///
/// Block-structures such as [block-diagonal matrices](block::BlockDiag) and
/// [block-vectors](block::BlockVector).
pub mod block;
/// Block-sparse matrix structures.
///
/// [Block-sparse matrices](block_sparse::BlockSparseMatrix),
/// [symmetric block-sparse matrices](block_sparse::BlockSparseSymmetricMatrix)
/// and their corresponding builders.
pub mod block_sparse;
/// Dense matrix structures.
///
/// [DenseSymmetricMatrix](dense::DenseSymmetricMatrix) and its
/// [builder](dense::DenseSymmetricMatrixBuilder).
pub mod dense;
/// Sparse matrix structures
/// 
/// [Sparse matrix](sparse::SparseMatrix),
/// [sparse symmetric matrix](sparse::SparseSymmetricMatrix) and its
/// [builder](sparse::SparseSymmetricMatrixBuilder).
pub mod sparse;

pub(crate) mod grid;
pub(crate) mod symmetric_matrix;

pub use grid::*;
pub use symmetric_matrix::*;

/// A set of partitions along one axis.
///
/// In the context of this crate, a vector or an axis of a matrix is subdivided into partitions
/// of same-sized blocks.
///
/// ```ascii
/// -------------------------------------------
/// |  B0 ... B0  |  B1 ... B1  |    * * *    |
/// |  (m times)  |  (n times)  |             |
/// -------------------------------------------
///   partition 0   partition 1
/// ```
///
/// Partition 0 consists of m blocks, all of the same dimension B0, partition 1 contains n
/// blocks of dimension B1 etc.
#[derive(Debug, Clone)]
pub struct PartitionSet {
    partition_specs: Vec<PartitionSpec>,
    scalar_offsets_by_partition: Vec<usize>,
    scalar_dim: usize,
}

impl PartitionSet {
    /// Create a new partition set given a slice of partition specifications.
    pub fn new(partition_specs: Vec<PartitionSpec>) -> Self {
        let mut scalar_offsets_by_partition = Vec::with_capacity(partition_specs.len());

        let mut offset = 0usize;
        for partition in &partition_specs {
            scalar_offsets_by_partition.push(offset);
            offset += partition.block_count * partition.block_dim;
        }

        PartitionSet {
            partition_specs,
            scalar_offsets_by_partition,
            scalar_dim: offset,
        }
    }

    /// Return slice of partition specifications, which consist of block counts and block
    /// dimensions.
    #[inline]
    pub fn specs(&self) -> &[PartitionSpec] {
        &self.partition_specs
    }

    /// Number of partitions.
    #[inline]
    pub fn len(&self) -> usize {
        self.partition_specs.len()
    }

    /// Returns true if the set of partitions is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.partition_specs.is_empty()
    }

    /// Total scalar dimension.
    #[inline]
    pub fn scalar_dim(&self) -> usize {
        self.scalar_dim
    }

    /// Return slice of scalar offsets.
    #[inline]
    pub fn scalar_offsets_by_partition(&self) -> &[usize] {
        &self.scalar_offsets_by_partition
    }

    /// Return the range (scalar offset and block dimension) for a block.
    #[inline]
    pub fn block_range(&self, idx: PartitionBlockIndex) -> BlockRange {
        let partition = &self.partition_specs[idx.partition];
        let block_dim = partition.block_dim;
        let scalar_offset = self.scalar_offsets_by_partition[idx.partition] + idx.block * block_dim;
        BlockRange {
            start_idx: scalar_offset,
            block_dim,
        }
    }
}

/// Specification of a vector / matrix partition.
///
/// See [PartitionSet] for details.
#[derive(Debug, Clone)]
pub struct PartitionSpec {
    /// Number of blocks in the partition.
    pub block_count: usize,
    /// Dimension of blocks in this partition.
    pub block_dim: usize,
}

/// Index of a given block inside a partition.
#[derive(Debug, Clone, Copy)]
pub struct PartitionBlockIndex {
    /// Index of the partition the block is in.
    pub partition: usize,
    /// Local index of the block within its partition.
    pub block: usize,
}

/// Scalar offset and dimension of a given block.
#[derive(Clone, Debug, Copy, Default)]
pub struct BlockRange {
    /// Global offset index to first element in the block.
    pub start_idx: usize,
    /// Dimension of blocks in this partition.
    pub block_dim: usize,
}
