use snafu::Snafu;

/// Linear solver error
#[derive(Snafu, Debug)]
pub enum LinearSolverError {
    /// Sparse LDLᵀ error
    #[snafu(display("sparse LDLᵀ error {}", details))]
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

    /// Matrix/vector sizes don’t match (e.g., A not square or b has wrong length)
    #[snafu(display("dimension mismatch"))]
    DimensionMismatch,

    /// Factorization failure
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
    /// LDLᵀ Error
    LdltError,
    /// unspecific - to be forward compatible
    Unspecific,
}
