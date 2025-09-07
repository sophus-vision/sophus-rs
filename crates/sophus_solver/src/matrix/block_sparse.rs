pub(crate) mod block_col_compressed_matrix;
pub(crate) mod block_sparse_matrix_builder;
pub(crate) mod block_sparse_triplets;
pub(crate) mod lower_block_sparse_matrix;

pub use block_col_compressed_matrix::*;
pub use block_sparse_matrix_builder::*;
pub use block_sparse_triplets::*;
pub use lower_block_sparse_matrix::*;

/// Partition specification.
#[derive(Debug, Clone)]
pub struct PartitionSpec {
    /// number of blocks in the partition
    pub block_count: usize,
    /// dimension of the block
    pub block_dimension: usize,
}
