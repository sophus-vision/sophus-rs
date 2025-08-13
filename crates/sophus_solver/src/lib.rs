#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]
#![cfg_attr(nightly, feature(doc_auto_cfg))]

#[doc = include_str!(concat!("../",  core::env!("CARGO_PKG_README")))]
#[cfg(doctest)]
pub struct ReadmeDoctests;

/// Block-sparse data structures.
pub mod block_sparse;
/// Dense data structures.
pub mod dense;
/// grid.
pub mod grid;
/// LDLt to solve semi-positive definite systems.
pub mod ldlt;
/// LU to solve invertible systems.
pub mod lu;
/// QR to solve invertible systems.
pub mod qr;

mod asserts;

pub use block_sparse::*;
pub use ldlt::{
    dense_ldlt::*,
    sparse_ldlt::*,
};
pub use lu::{
    dense_lu::*,
    faer_sparse_lu::*,
};
pub use qr::sparse_qr::*;
use snafu::Snafu;

/// Linear solver error
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

    // ---------------- new variants used by dense LDLᵀ code ----------------
    /// Matrix/vector sizes don’t match (e.g., A not square or b has wrong length)
    #[snafu(display("dimension mismatch"))]
    DimensionMismatch,

    /// Factorization failed (e.g., non-SPD pivot encountered)
    #[snafu(display("factorization failed"))]
    FactorizationFailed,
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

/// sym mat
pub trait IsSymmetricMatrixBuilder {
    /// mat
    type Matrix;

    /// Create a symmetric matrix "filled" with zeros.
    ///
    /// The shape of the matrix is determined by the partition specs.
    fn zero(partitions: &[PartitionSpec]) -> Self;

    /// scalar dimension of the matrix.
    fn scalar_dimension(&self) -> usize;

    /// Add a block to the matrix.
    ///
    /// This is a += operation, i.e., the block is added to the existing block.
    ///
    /// Only lower triangular blocks are accepted.
    ///
    /// In release mode, upper triangular blocks are ignored. In debug mode,
    /// this function will panic if the block is upper triangular.
    fn add_lower_block(
        &mut self,
        region_idx: &[usize; 2],
        block_index: [usize; 2],
        block: &nalgebra::DMatrixView<f64>,
    );

    /// Export UPPER triangular scalar triplets (view) from lower storage.
    fn build(&self) -> Self::Matrix;
}

/// Linear solver of a sparse symmetric matrix.
pub trait IsSparseSymmetricLinearSystem {
    /// Solve the linear system.
    fn solve(
        &self,
        triplets: &SymmetricBlockSparseMatrixBuilder,
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
