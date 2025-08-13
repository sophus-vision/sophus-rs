pub(crate) mod block_sparse_compressed_matrix;
pub(crate) mod block_sparse_matrix;
pub(crate) mod block_vector;

pub use block_sparse_compressed_matrix::*;
pub use block_sparse_matrix::*;
pub use block_vector::*;

pub(crate) mod block_sparse_symmetric_matrix;
pub use block_sparse_symmetric_matrix::*;

/// Range of a block
#[derive(Clone, Debug, Copy, Default)]
pub struct BlockRange {
    /// Index of the first element of the block
    pub index: i64,
    /// Dimension of the block
    pub dim: usize,
}

/// Additional region
#[derive(Debug, Clone)]
pub struct PartitionSpec {
    /// num blocks
    pub num_blocks: usize,
    /// block dim
    pub block_dim: usize,
}
