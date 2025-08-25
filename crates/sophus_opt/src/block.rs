pub(crate) mod block_gradient;
pub(crate) mod block_hessian;
pub(crate) mod block_jacobian;

pub use block_gradient::*;
pub use block_hessian::*;
pub use block_jacobian::*;

/// Range of a block
#[derive(Clone, Debug, Copy, Default)]
pub struct BlockRange {
    /// Index of the first element of the block
    pub index: i64,
    /// Dimension of the block
    pub dim: usize,
}
