pub(crate) mod block_vector;

pub use block_vector::*;

/// Partition specification.
#[derive(Debug, Clone)]
pub struct PartitionSpec {
    /// number of blocks in the partition
    pub block_count: usize,
    /// dimension of the block
    pub block_dimension: usize,
}
