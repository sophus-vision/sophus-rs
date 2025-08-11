#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]
#![cfg_attr(nightly, feature(doc_auto_cfg))]

#[doc = include_str!(concat!("../",  core::env!("CARGO_PKG_README")))]
#[cfg(doctest)]
pub struct ReadmeDoctests;

pub(crate) mod block_gradient;
pub(crate) mod block_hessian;
pub(crate) mod block_jacobian;
pub(crate) mod block_sparse_matrix;
pub(crate) mod block_vector;
pub(crate) mod compressed_block_matrix;
pub(crate) mod grid;

/// Block-sparse linear solvers.
pub mod block_solvers;
/// Scalar/standard linear solvers - i.e. not leveraging block-sparse structure explicitly.
pub mod scalar_solvers;

pub(crate) mod symmetric_block_sparse_matrix;

mod asserts;

pub use block_gradient::*;
pub use block_hessian::*;
pub use block_jacobian::*;
pub use block_sparse_matrix::*;
pub use block_vector::*;
pub use compressed_block_matrix::*;
pub use grid::*;
use snafu::Snafu;
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

/// Linear solver error
#[derive(Snafu, Debug)]
pub enum LinearSolverError {
    /// Sparse LDLt error
    #[snafu(display("sparse LDLt error {}", details))]
    SparseLdltError {
        /// details
        details: SparseSolverError,
    },
    /// Sparse LU error
    #[snafu(display("sparse LU error {}", details))]
    SparseLuError {
        /// details
        details: SparseSolverError,
    },
    /// Sparse QR error
    #[snafu(display("sparse QR error {}", details))]
    SparseQrError {
        /// details
        details: SparseSolverError,
    },
    /// Dense LU error
    #[snafu(display("dense LU solve failed"))]
    DenseLuError,
}

/// Sparse solver error - forwarded from faer error enums.
#[derive(Snafu, Debug)]
pub enum SparseSolverError {
    /// An index exceeding the maximum value
    IndexOverflow,
    /// Memory allocation failed.
    OutOfMemory,
    /// LU decomposition specific error
    SymbolicSingular,
    /// LDLt Error
    LdltError,
    /// unspecific - to be forward compatible
    Unspecific,
}
/// Linear solver of a sparse symmetric matrix.
pub trait IsSparseSymmetricLinearSystem {
    /// Solve the linear system.
    fn solve(
        &self,
        triplets: &SymmetricBlockSparseMatrix,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, LinearSolverError>;
}

/// Linear solver of a dense symmetric matrix.
pub trait IsDenseLinearSystem {
    /// Solve the linear system.
    fn solve_dense(
        &self,
        mat_a: nalgebra::DMatrix<f64>,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, LinearSolverError>;
}
