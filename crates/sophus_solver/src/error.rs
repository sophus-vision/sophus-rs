use snafu::Snafu;

use crate::matrix::PartitionBlockIndex;

/// Definiteness classification of a matrix, determined during LDLᵀ factorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Definiteness {
    /// All pivots positive — positive definite.
    PositiveDefinite,
    /// All pivots non-negative, at least one zero — positive semi-definite (rank-deficient).
    PositiveSemiDefinite,
    /// At least one negative pivot — indefinite (e.g. KKT system).
    Indefinite,
}

/// Result of LDLᵀ factorization: rank and definiteness.
#[derive(Debug, Clone, Copy)]
pub struct LdltResult {
    /// Number of non-zero pivots.
    pub rank: usize,
    /// Definiteness classification.
    pub definiteness: Definiteness,
    /// Pivot condition: `min(|d[j]|) / max(|d[j]|)` over non-zero pivots.
    ///
    /// Values near 1.0 indicate well-conditioned; values near `f64::EPSILON` (~1e-16)
    /// indicate loss of precision. A value of 0.0 means all pivots were zero (rank 0).
    pub pivot_condition: f64,
    /// True if Bunch-Kaufman fallback was used for any diagonal block.
    ///
    /// BK fallback activates when standard LDLᵀ encounters a near-zero pivot
    /// in an indefinite block (pivot_condition too low for stable factorization).
    pub used_bk_fallback: bool,
}

/// Linear solver error.
#[derive(Snafu, Debug, Clone)]
#[snafu(visibility(pub(crate)))]
pub enum LinearSolverError {
    /// Solver does not support `inverse_block` (requires an LDLᵀ-based solver).
    #[snafu(display("inverse_block not supported for solver '{solver}'"))]
    UnsupportedForInverseBlock {
        /// Solver name.
        solver: String,
    },

    /// Error in faer sparse LDLᵀ.
    #[snafu(display("faer sparse LDLᵀ: {}", faer_error))]
    FaerSparseLdltError {
        /// source
        faer_error: FaerSparseSolverError,
    },

    /// Error in faer sparse LU.
    #[snafu(display("faer sparse LU: {}", faer_error))]
    FaerSparseLuError {
        /// source
        faer_error: FaerSparseSolverError,
    },

    /// Error in faer sparse QR.
    #[snafu(display("sparse QR: {}", faer_error))]
    FaerSparseQrError {
        /// source
        faer_error: FaerSparseSolverError,
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

    /// Block-sparse LDLᵀ error
    #[snafu(display("block-sparse LDLᵀ: {}", source))]
    BlockSparseLdltError {
        /// source
        source: BlockSparseLdltError,
    },

    /// Marginalized variable block `H_mm[b]` is singular; Schur complement not applicable.
    #[snafu(display("SingularMargBlock: partition={} block={}", partition, block))]
    SingularMargBlock {
        /// partition index
        partition: usize,
        /// block index within partition
        block: usize,
    },

    /// The range-space KKT Schur complement M = G_f S_ff⁻¹ G_fᵀ is singular.
    ///
    /// This means the equality constraints are linearly dependent or
    /// under-determined with respect to the free variables.
    #[snafu(display(
        "SingularKktConstraint: M = G_f S_ff⁻¹ G_fᵀ is singular (num_constraints={})",
        num_constraints
    ))]
    SingularKktConstraint {
        /// number of constraint scalars
        num_constraints: usize,
    },
}

/// Error of LU decomposition.
#[derive(Snafu, Debug, Clone)]
#[snafu(visibility(pub(crate)))]
pub enum LuDecompositionError {
    /// Pivot is near-singular.
    #[snafu(display("near-singular pivot"))]
    NearSingularPivot,
}

/// Error of LDLᵀ decomposition.
#[derive(Snafu, Debug, Clone)]
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
}

/// Error of LDLᵀ decomposition.
#[derive(Snafu, Debug, Clone)]
#[snafu(visibility(pub(crate)))]
pub enum BlockSparseLdltError {
    /// Factorization failure
    #[snafu(display("block diagonal ({:?}): {}", idx, source))]
    BlockDiagLdltError {
        /// source
        source: LdltDecompositionError,
        /// partition index
        idx: PartitionBlockIndex,
    },
}

/// Wrapper error type - for faer sparse solver errors.
#[derive(Snafu, Debug, Clone)]
pub enum FaerSparseSolverError {
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
