pub(crate) mod block_gradient;
pub(crate) mod block_hessian;
pub(crate) mod block_jacobian;
pub(crate) mod block_sparse_matrix;
pub(crate) mod block_vector;
pub(crate) mod compressed_block_matrix;
pub(crate) mod grid;

pub use block_gradient::*;
pub use block_hessian::*;
pub use block_jacobian::*;
pub use block_sparse_matrix::*;
pub use block_vector::*;
pub use compressed_block_matrix::*;
pub use grid::*;

pub(crate) mod symmetric_block_sparse_matrix;
pub use symmetric_block_sparse_matrix::*;

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
