/// Block gradient vector
pub mod block_gradient;
/// Block Hessian matrix
pub mod block_hessian;

/// Range of a block
#[derive(Clone, Debug, Copy, Default)]
pub struct BlockRange {
    /// Index of the first element of the block
    pub index: i64,
    /// Dimension of the block
    pub dim: usize,
}
