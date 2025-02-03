/// Block gradient vector
pub mod block_gradient;
/// Block Hessian matrix
pub mod block_hessian;
/// Block jacobian
pub mod block_jacobian;
/// builder for a block sparse matrix
pub mod block_sparse_matrix_builder;
/// Block vector
pub mod block_vector;
/// Generic grid
pub mod grid;
/// builder for a symmetric block sparse matrix
pub mod symmetric_block_sparse_matrix_builder;

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
