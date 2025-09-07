use snafu::Snafu;

/// Linear solver error.
#[derive(Snafu, Debug)]
#[snafu(visibility(pub(crate)))]
pub enum LinearSolverError {
    /// Error in faer sparse LDLᵀ.
    #[snafu(display("faer sparse LDLᵀ: {}", faer_error))]
    FaerSparseLdltError {
        /// source
        faer_error: FearSparseSolverError,
    },

    /// Error in faer sparse LU.
    #[snafu(display("faer sparse LU: {}", faer_error))]
    FaerSparseLuError {
        /// source
        faer_error: FearSparseSolverError,
    },

    /// Error in faer sparse QR.
    #[snafu(display("sparse QR: {}", faer_error))]
    FaerSparseQrError {
        /// source
        faer_error: FearSparseSolverError,
    },

    /// Dense LU error
    #[snafu(display("dense LU: {}", source))]
    DenseLuError {
        /// source
        source: LuDecompositionError,
    },

    /// Dense LDLᵀ error
    #[snafu(display("dense LDLᵀ: {}", source))]
    DenseLdltError {
        /// source
        source: LdltDecompositionError,
    },

    /// Sparse LDLᵀ error
    #[snafu(display("sparse LDLᵀ: {}", source))]
    SparseLdltError {
        /// source
        source: LdltDecompositionError,
    },

    /// Sparse LDLᵀ error
    #[snafu(display("block-sparse LDLᵀ: {}", source))]
    BlockSparseLdltError {
        /// source
        source: BlockSparseLdltError,
    },
}

/// Error of LU decomposition.
#[derive(Snafu, Debug)]
#[snafu(visibility(pub(crate)))]
pub enum LuDecompositionError {
    /// Pivot is near-singular.
    #[snafu(display("near-singular pivot"))]
    NearSingularPivot,
}

/// Error of LDLᵀ decomposition.
#[derive(Snafu, Debug)]
#[snafu(visibility(pub(crate)))]
pub enum LdltDecompositionError {
    /// Pivot `d[j]` is not finite.
    #[snafu(display("Non-finite pivot d[{}] = {}", j, d_jj))]
    NonFinitePivot {
        /// index
        j: usize,
        /// pivot value `d[j]`
        d_jj: f64,
    },

    /// Pivot `d[j]` is negative.
    #[snafu(display("negative pivot d[{}] = {}", j, d_jj))]
    NegativeFinitePivot {
        /// index
        j: usize,
        /// pivot value `d[j]`
        d_jj: f64,
    },
}

/// Error of LDLᵀ decomposition.
#[derive(Snafu, Debug)]
#[snafu(visibility(pub(crate)))]
pub enum BlockSparseLdltError {
    /// Factorization failure
    #[snafu(display("block diagonal ({}, {}): {}", partition_idx, local_block_idx, source))]
    BlockDiagLdltError {
        /// source
        source: LdltDecompositionError,
        /// partition index
        partition_idx: usize,
        /// local block index
        local_block_idx: usize,
    },
}

/// Wrapper error type - for faer sparse solver errors.
#[derive(Snafu, Debug)]
pub enum FearSparseSolverError {
    /// An index exceeding the maximum value.
    IndexOverflow,
    /// Memory allocation failed.
    OutOfMemory,
    /// LU decomposition specific error.
    SymbolicSingular,
    /// LDLᵀ Error.
    LdltError,
    /// unspecific error - to be forward compatible.
    Unspecific,
}
