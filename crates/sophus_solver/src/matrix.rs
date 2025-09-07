/// Block-sparse data structures.
pub mod block;

/// Block-sparse data structures.
pub mod block_sparse;

/// Compressed sparse matrix.
pub mod compressed_matrix;

/// Compressible matrix.s
pub mod compressible_matrix;
/// Dense data structures.
pub mod dense;
/// Grid structure.
pub mod grid;

/// Sparse data structures.
pub mod sparse;
/// Symmetric matrix.
pub mod symmetric_matrix;

pub use crate::matrix::{
    block::*,
    block_sparse::*,
    compressed_matrix::*,
    compressible_matrix::*,
    dense::*,
    sparse::*,
    symmetric_matrix::*,
};
