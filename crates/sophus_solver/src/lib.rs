#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]
#![cfg_attr(nightly, allow(unused_features))]
#![cfg_attr(nightly, feature(doc_cfg))]

/// Conditionally invokes `puffin::profile_scope!` on native targets only.
#[cfg(not(target_arch = "wasm32"))]
macro_rules! profile_scope {
    ($($arg:tt)*) => { puffin::profile_scope!($($arg)*) };
}

/// No-op on WASM.
#[cfg(target_arch = "wasm32")]
macro_rules! profile_scope {
    ($($arg:tt)*) => {};
}

#[doc = include_str!(concat!("../",  core::env!("CARGO_PKG_README")))]
#[cfg(doctest)]
pub struct ReadmeDoctests;

use std::fmt::Debug;

pub use error::*;
pub use ldlt::schur_ldlt::SchurFactor;
pub use matrix::schur::Schur;
use nalgebra::DMatrix;

use crate::{
    ldlt::{
        BlockSparseLdlt,
        BlockSparseLdltFactor,
        DenseLdlt,
        DenseLdltFactor,
        FaerSparseLblt,
        FaerSparseLbltSystem,
        FaerSparseLdlt,
        FaerSparseLdltSystem,
        SparseLdlt,
        SparseLdltFactor,
        faer_sparse_lblt::FaerSparseLbltSymbolic,
        faer_sparse_ldlt::FaerSparseLdltSymbolic,
        min_norm_ldlt::{
            block_sparse_min_norm_ldlt::BlockSparseMinNormPsd,
            dense_min_norm_ldlt::DenseMinNormFactor,
            sparse_min_norm_ldlt::SparseMinNormPsd,
        },
        sparse_ldlt::SparseLdltSymbolic,
    },
    lu::{
        DenseLu,
        DenseLuSystem,
        FaerSparseLu,
        FaerSparseLuSystem,
    },
    matrix::{
        IsSymmetricMatrix,
        IsSymmetricMatrixBuilder,
        PartitionBlockIndex,
        PartitionSet,
        SymmetricMatrixBuilderEnum,
        SymmetricMatrixEnum,
        block_sparse::block_sparse_symmetric_matrix_builder::BlockSparseSymmetricMatrixPattern,
        direct_solve::DirectSolveMatrix,
    },
    qr::{
        FaerSparseQr,
        FaerSparseQrFactor,
    },
};

/// Covariance computation from factorized Hessians with optional equality constraints.
pub mod covariance;
/// Solver error handling.
pub mod error;
/// Matrix operation kernels.
pub mod kernel;
/// LDLᵀ decomposition-based solvers for semi-definite linear systems.
pub mod ldlt;
/// LU decomposition-based solvers.
pub mod lu;
/// Matrix structures: dense, sparse and block-sparse.
///
/// In some real world applications, such a non-linear least squares optimization,
/// matrix appear in block form. For instance we might have a `N x N` matrix with four
/// *regions*, each containing matrix-blocks of different sizes:
///
/// ```ascii
///      ----------------------------
///      | A₀,₀  A₀,₁  A₀,₂  | B₀,₀ |
///      |                   |      |
///      | A₁,₀  A₁,₁  A₁,₂  | B₁,₀ |
/// M =  |                   |      |
///      | A₂,₀  A₂,₁  A₂,₂  | B₂,₀ |
///      ----------------------------
///      | C₀,₀  C₀,₁  C₀,₂  | D₀,₀ |
///      ----------------------------
/// ```
///
/// Blocks in region (0,0) might be 4x4 matrices, and blocks in region (0,1)
/// 4x2 matrices, blocks in region (1,0) 2x4 matrices, and in region (1,1) we have
/// 2x2 matrices.
///
/// We use [PartitionSet]s to describe the overall structure of matrix `M`.
/// Let's define the *row partition set* vertically. There are two partitions.
/// In the first partition, we have three rows of blocks and each matrix block has a height of 4.
/// In the second partition, we have just one block row, and each block has height 2.
/// Due to the symmetric structure of matrix M, the *column partitions* horizontally equal the row
/// partitions. Hence, the first column partition has three columns of blocks with width 4, etc.
///
/// ```
/// use nalgebra::DMatrix;
/// use sophus_autodiff::linalg::MatF64;
/// use sophus_solver::{
///     matrix::{
///         PartitionBlockIndex,
///         PartitionSet,
///         PartitionSpec,
///         block_sparse::BlockSparseSymmetricMatrixBuilder,
///     },
///     prelude::*,
/// };
///
/// let partitions = PartitionSet::new(vec![
///     PartitionSpec {
///         eliminate_last: false,
///         block_dim: 4,
///         block_count: 3,
///     },
///     PartitionSpec {
///         eliminate_last: false,
///         block_dim: 2,
///         block_count: 1,
///     },
/// ]);
///
/// // A is a symmetric matrix. Internally, it represented by a lower block-triangular matrix.
/// let mut mat_a = BlockSparseSymmetricMatrixBuilder::zero(partitions);
///
/// let p0_b0 = PartitionBlockIndex {
///     partition: 0,
///     block: 0,
/// };
/// let p0_b2 = PartitionBlockIndex {
///     partition: 0,
///     block: 2,
/// };
/// let p1_b0 = PartitionBlockIndex {
///     partition: 1,
///     block: 0,
/// };
///
/// // Adds blocks to region (0,0), at block index (0,0) and block index (2,0).
/// mat_a.add_lower_block(
///     p0_b0,
///     p0_b0,
///     &MatF64::from_array2([
///         [1.1, 2.2, 3.3, 4.4], //
///         [2.2, 2.2, 3.3, 4.4],
///         [3.3, 3.3, 3.3, 4.4],
///         [4.4, 4.4, 4.4, 4.4],
///     ])
///     .as_view(),
/// );
/// mat_a.add_lower_block(
///     p0_b2,
///     p0_b0,
///     &MatF64::from_array2([
///         [1.1, 1.1, 1.1, 1.1], //
///         [1.1, 1.1, 1.1, 1.1],
///         [1.1, 1.1, 1.1, 1.1],
///         [1.1, 1.1, 1.1, 1.1],
///     ])
///     .as_view(),
/// );
///
/// // Adds block to region (1,0), at block index (0,2).
/// mat_a.add_lower_block(
///     p1_b0,
///     p0_b2,
///     &MatF64::from_array2([
///         [2.2, 2.2, 2.2, 2.2], //
///         [1.1, 2.2, 3.3, 4.4],
///     ])
///     .as_view(),
/// );
///
/// // Adds block to region (1,1), at block index (0,0).
/// mat_a.add_lower_block(
///     p1_b0,
///     p1_b0,
///     &MatF64::from_array2([
///         [6.6, 8.8], //
///         [8.8, 9.9],
///     ])
///     .as_view(),
/// );
///
/// let mat_a = mat_a.build();
///
/// let mat_a_dense = DMatrix::from_row_slice(
///     14,
///     14,
///     &[
///         1.1, 2.2, 3.3, 4.4, 0.0, 0.0, 0.0, 0.0, 1.1, 1.1, 1.1, 1.1, 0.0, 0.0, //
///         2.2, 2.2, 3.3, 4.4, 0.0, 0.0, 0.0, 0.0, 1.1, 1.1, 1.1, 1.1, 0.0, 0.0, //
///         3.3, 3.3, 3.3, 4.4, 0.0, 0.0, 0.0, 0.0, 1.1, 1.1, 1.1, 1.1, 0.0, 0.0, //
///         4.4, 4.4, 4.4, 4.4, 0.0, 0.0, 0.0, 0.0, 1.1, 1.1, 1.1, 1.1, 0.0, 0.0, //
///         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
///         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
///         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
///         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
///         1.1, 1.1, 1.1, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2, 1.1, //
///         1.1, 1.1, 1.1, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2, 2.2, //
///         1.1, 1.1, 1.1, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2, 3.3, //
///         1.1, 1.1, 1.1, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2, 4.4, //
///         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2, 2.2, 2.2, 2.2, 6.6, 8.8, //
///         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1, 2.2, 3.3, 4.4, 8.8, 9.9,
///     ],
/// );
///
/// assert_eq!(mat_a.to_dense(), mat_a_dense);
/// ```
///
/// ## Symmetric matrices
///
/// The focus of this crate is symmetric matrices, e.g. matrices which implements the
/// [IsSymmetricMatrix] trait. They are constructed by a [builder](IsSymmetricMatrixBuilder).
///
/// The concrete implementation may use a (dense)[matrix::dense], a (sparse)[matrix::sparse]
/// or (block-sparse)[matrix::block_sparse] representation.
pub mod matrix;
/// QR decomposition-based solvers.
pub mod qr;
/// SVD decomposition-based solvers, for computing the pseudo-inverse.
pub mod svd;
/// Set of linear system test problems.
pub mod test_examples;

/// Linear solver of linear system.
pub trait IsLinearSolver {
    /// Builder for a symmetric matrix.
    type SymmetricMatrixBuilder: IsSymmetricMatrixBuilder;

    /// Factorization of a matrix.
    type Factor: IsFactor;

    /// Name of solver variant.
    const NAME: &'static str;

    /// Create matrix builder compatible with this solver.
    fn zero(&self, partitions: PartitionSet) -> SymmetricMatrixBuilderEnum;

    /// Set the parallelize option to true or false.
    ///
    /// Note: Not all solvers support parallel execution. In this case, this function is no-op.
    fn set_parallelize(&mut self, parallelize: bool);

    /// Name of solver variant.
    fn name(&self) -> String {
        Self::NAME.into()
    }

    /// Return factorization of provided matrix `A`.
    fn factorize(
        &self,
        mat_a: &<Self::SymmetricMatrixBuilder as IsSymmetricMatrixBuilder>::Matrix,
    ) -> Result<Self::Factor, LinearSolverError>;

    /// Solve linear system: ``A x = b``.
    fn solve(
        &self,
        mat_a: &<Self::SymmetricMatrixBuilder as IsSymmetricMatrixBuilder>::Matrix,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, LinearSolverError> {
        let system = self.factorize(mat_a)?;
        system.solve(b)
    }

    /// Solve linear system in place ``x := A⁻¹ x``.
    fn solve_inplace(
        &self,
        mat_a: &<Self::SymmetricMatrixBuilder as IsSymmetricMatrixBuilder>::Matrix,
        b: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        let system = self.factorize(mat_a)?;
        system.solve_inplace(b)
    }
}

#[derive(Copy, Clone, Debug)]
/// Linear solver enum.
pub enum LinearSolverEnum {
    /// Dense solver using LDLᵀ factorization.
    DenseLdlt(DenseLdlt),
    /// Dense solver using LU factorization.
    DenseLu(DenseLu),
    /// Sparse solver using LDLᵀ factorization.
    SparseLdlt(SparseLdlt),
    /// Block-sparse solver using LDLᵀ factorization.
    BlockSparseLdlt(BlockSparseLdlt),
    /// Sparse solver using faer's QR factorization.
    FaerSparseQr(FaerSparseQr),
    /// Sparse solver using faer's LU factorization.
    FaerSparseLu(FaerSparseLu),
    /// Sparse solver using faer's LDLᵀ factorization.
    FaerSparseLdlt(FaerSparseLdlt),
    /// Sparse solver using faer's LBLᵀ (Bunch-Kaufman) factorization for indefinite systems.
    FaerSparseLblt(FaerSparseLblt),
    /// Schur-complement solve: block-sparse LDLᵀ for the reduced S system.
    SchurBlockSparseLdlt(BlockSparseLdlt),
    /// Schur-complement solve: sparse LDLᵀ for the reduced S system.
    SchurSparseLdlt(SparseLdlt),
    /// Schur-complement solve: faer sparse LDLᵀ for the reduced S system.
    SchurFaerSparseLdlt(FaerSparseLdlt),
}

impl Default for LinearSolverEnum {
    fn default() -> Self {
        LinearSolverEnum::FaerSparseLdlt(FaerSparseLdlt::default())
    }
}

impl LinearSolverEnum {
    /// Get all available solvers
    pub fn all_solvers() -> Vec<LinearSolverEnum> {
        let mut solvers = LinearSolverEnum::dense_solvers();
        solvers.extend(LinearSolverEnum::sparse_solvers());
        solvers
    }

    /// Set the parallelize option to true or false.
    ///
    /// Note: Not all solvers support parallel execution. In this case, this function is no-op.
    pub fn set_parallelize(&mut self, parallelize: bool) {
        match self {
            LinearSolverEnum::DenseLdlt(dense_ldlt) => dense_ldlt.set_parallelize(parallelize),
            LinearSolverEnum::DenseLu(dense_lu) => dense_lu.set_parallelize(parallelize),
            LinearSolverEnum::SparseLdlt(sparse_ldlt) => sparse_ldlt.set_parallelize(parallelize),
            LinearSolverEnum::BlockSparseLdlt(block_sparse_ldlt) => {
                block_sparse_ldlt.set_parallelize(parallelize)
            }
            LinearSolverEnum::FaerSparseQr(faer_sparse_qr) => {
                faer_sparse_qr.set_parallelize(parallelize)
            }
            LinearSolverEnum::FaerSparseLu(faer_sparse_lu) => {
                faer_sparse_lu.set_parallelize(parallelize)
            }
            LinearSolverEnum::FaerSparseLdlt(faer_sparse_ldlt) => {
                faer_sparse_ldlt.set_parallelize(parallelize)
            }
            LinearSolverEnum::FaerSparseLblt(s) => s.set_parallelize(parallelize),
            LinearSolverEnum::SchurBlockSparseLdlt(s) => s.set_parallelize(parallelize),
            LinearSolverEnum::SchurSparseLdlt(s) => s.set_parallelize(parallelize),
            LinearSolverEnum::SchurFaerSparseLdlt(s) => s.set_parallelize(parallelize),
        }
    }

    /// Get list off all dense solvers.
    pub fn dense_solvers() -> Vec<LinearSolverEnum> {
        vec![
            LinearSolverEnum::DenseLdlt(DenseLdlt::default()),
            LinearSolverEnum::DenseLu(DenseLu {}),
        ]
    }

    /// Get list of all sparse solvers (non-Schur).
    pub fn sparse_solvers() -> Vec<LinearSolverEnum> {
        vec![
            LinearSolverEnum::SparseLdlt(SparseLdlt::default()),
            LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default()),
            LinearSolverEnum::FaerSparseQr(FaerSparseQr {}),
            LinearSolverEnum::FaerSparseLu(FaerSparseLu {}),
            LinearSolverEnum::FaerSparseLdlt(FaerSparseLdlt::default()),
            LinearSolverEnum::FaerSparseLblt(FaerSparseLblt::default()),
        ]
    }

    /// Get list of all Schur-complement sparse solvers.
    ///
    /// These solvers require the hessian to be wrapped in a [`crate::matrix::schur::Schur`]
    /// structure; they cannot be used directly with [`LinearSolverEnum::factorize`].
    pub fn schur_sparse_solvers() -> Vec<LinearSolverEnum> {
        vec![
            LinearSolverEnum::SchurBlockSparseLdlt(BlockSparseLdlt::default()),
            LinearSolverEnum::SchurSparseLdlt(SparseLdlt::default()),
            LinearSolverEnum::SchurFaerSparseLdlt(FaerSparseLdlt::default()),
        ]
    }

    /// Returns true if this solver supports equality constraints (KKT systems).
    ///
    /// Most solvers handle symmetric indefinite systems (LDLᵀ accepts negative
    /// pivots). The exception is `FaerSparseLdlt` which wraps faer's LDLᵀ that
    /// assumes positive-definite input.
    pub fn supports_eq_constraints(&self) -> bool {
        !matches!(self, LinearSolverEnum::FaerSparseLdlt(_))
    }

    /// Returns true if this is a Schur-complement variant.
    pub fn is_schur(&self) -> bool {
        matches!(
            self,
            LinearSolverEnum::SchurBlockSparseLdlt(_)
                | LinearSolverEnum::SchurSparseLdlt(_)
                | LinearSolverEnum::SchurFaerSparseLdlt(_)
        )
    }

    /// For Schur variants, returns the inner solver used for S; panics for non-Schur variants.
    pub fn schur_inner_solver(&self) -> LinearSolverEnum {
        match self {
            LinearSolverEnum::SchurBlockSparseLdlt(s) => LinearSolverEnum::BlockSparseLdlt(*s),
            LinearSolverEnum::SchurSparseLdlt(s) => LinearSolverEnum::SparseLdlt(*s),
            LinearSolverEnum::SchurFaerSparseLdlt(s) => LinearSolverEnum::FaerSparseLdlt(*s),
            _ => panic!("not a Schur variant"),
        }
    }

    /// Convert this solver to its Schur-complement variant, if one exists.
    ///
    /// Returns `Some(SchurXxx(...))` for `BlockSparseLdlt`, `SparseLdlt`, and `FaerSparseLdlt`.
    /// Returns `None` for solvers that have no Schur variant (dense, LU, QR, already-Schur).
    pub fn to_schur(&self) -> Option<LinearSolverEnum> {
        match self {
            LinearSolverEnum::BlockSparseLdlt(s) => {
                Some(LinearSolverEnum::SchurBlockSparseLdlt(*s))
            }
            LinearSolverEnum::SparseLdlt(s) => Some(LinearSolverEnum::SchurSparseLdlt(*s)),
            LinearSolverEnum::FaerSparseLdlt(s) => Some(LinearSolverEnum::SchurFaerSparseLdlt(*s)),
            _ => None,
        }
    }

    /// All solvers that handle indefinite (KKT) systems.
    ///
    /// Includes solvers with and without BK fallback. `SparseLdlt` accepts
    /// negative pivots but has no fallback for ill-conditioned cases.
    /// Excludes `FaerSparseLdlt` (assumes PD) and Schur variants.
    /// See also [`robust_indefinite_solvers`] for the subset with BK fallback.
    pub fn indefinite_solvers() -> Vec<LinearSolverEnum> {
        vec![
            LinearSolverEnum::DenseLdlt(DenseLdlt::default()),
            LinearSolverEnum::DenseLu(DenseLu {}),
            LinearSolverEnum::SparseLdlt(SparseLdlt::default()),
            LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default()),
            LinearSolverEnum::FaerSparseQr(FaerSparseQr {}),
            LinearSolverEnum::FaerSparseLu(FaerSparseLu {}),
            LinearSolverEnum::FaerSparseLblt(FaerSparseLblt::default()),
        ]
    }

    /// Solvers robust to ill-conditioned indefinite systems.
    ///
    /// These use pivoting (BK, LU, QR) or partition-aware ordering to handle
    /// near-zero pivots in KKT systems. Excludes:
    /// - `SparseLdlt`: no BK fallback, no partition-aware ordering
    /// - `FaerSparseLdlt`: assumes PD
    /// - `FaerSparseLblt`: faer's AMD doesn't respect partition ordering
    pub fn robust_indefinite_solvers() -> Vec<LinearSolverEnum> {
        vec![
            LinearSolverEnum::DenseLdlt(DenseLdlt::default()),
            LinearSolverEnum::DenseLu(DenseLu {}),
            LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default()),
            LinearSolverEnum::FaerSparseQr(FaerSparseQr {}),
            LinearSolverEnum::FaerSparseLu(FaerSparseLu {}),
        ]
    }

    /// Solvers for BA-like problems with Schur complement (PD only, no eq constraints).
    ///
    /// Returns pairs of `(standard, schur_name)` where `standard.to_schur()` gives the
    /// Schur variant.
    pub fn ba_solvers() -> Vec<(LinearSolverEnum, &'static str)> {
        vec![
            (
                LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default()),
                "block-sparse LDLt",
            ),
            (
                LinearSolverEnum::FaerSparseLdlt(FaerSparseLdlt::default()),
                "faer sparse LDLt",
            ),
        ]
    }

    /// Solvers for BA with equality constraints (scale constraint, etc.).
    ///
    /// Includes both non-Schur indefinite solvers and Schur variants.
    pub fn ba_eq_solvers() -> Vec<(LinearSolverEnum, &'static str, bool)> {
        vec![
            // Non-Schur indefinite solvers (solve full KKT system).
            (
                LinearSolverEnum::FaerSparseLu(FaerSparseLu {}),
                "faer sparse LU",
                false,
            ),
            (
                LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default()),
                "block-sparse LDLt",
                false,
            ),
            (
                LinearSolverEnum::FaerSparseLblt(FaerSparseLblt::default()),
                "faer sparse LBLt",
                false,
            ),
            // Schur variants (range-space KKT on reduced system).
            (
                LinearSolverEnum::SchurBlockSparseLdlt(BlockSparseLdlt::default()),
                "Schur+block-sparse LDLt",
                true,
            ),
            (
                LinearSolverEnum::SchurFaerSparseLdlt(FaerSparseLdlt::default()),
                "Schur+faer sparse LDLt",
                true,
            ),
        ]
    }
}

impl LinearSolverEnum {
    /// Create matrix builder compatible with this solver.
    pub fn zero(&self, partitions: PartitionSet) -> SymmetricMatrixBuilderEnum {
        match self {
            LinearSolverEnum::DenseLdlt(dense_ldlt) => dense_ldlt.zero(partitions),
            LinearSolverEnum::DenseLu(dense_lu) => dense_lu.zero(partitions),
            LinearSolverEnum::FaerSparseQr(faer_sparse_qr) => faer_sparse_qr.zero(partitions),
            LinearSolverEnum::FaerSparseLu(faer_sparse_lu) => faer_sparse_lu.zero(partitions),
            LinearSolverEnum::FaerSparseLdlt(faer_sparse_ldlt) => faer_sparse_ldlt.zero(partitions),
            LinearSolverEnum::FaerSparseLblt(s) => s.zero(partitions),
            LinearSolverEnum::BlockSparseLdlt(block_sparse_ldlt) => {
                block_sparse_ldlt.zero(partitions)
            }
            LinearSolverEnum::SparseLdlt(sparse_ldlt) => sparse_ldlt.zero(partitions),
            // Schur variants always use BlockSparseLower for H (the Schur forward pass requires
            // block-sparse structure).
            LinearSolverEnum::SchurBlockSparseLdlt(_)
            | LinearSolverEnum::SchurSparseLdlt(_)
            | LinearSolverEnum::SchurFaerSparseLdlt(_) => {
                use crate::matrix::IsSymmetricMatrixBuilder;
                SymmetricMatrixBuilderEnum::BlockSparseLower(
                    crate::matrix::block_sparse::BlockSparseSymmetricMatrixBuilder::zero(
                        partitions,
                    ),
                    *self,
                )
            }
        }
    }

    /// Name of solver variant.
    pub fn name(&self) -> String {
        match self {
            LinearSolverEnum::DenseLdlt(dense_ldlt) => dense_ldlt.name(),
            LinearSolverEnum::DenseLu(dense_lu) => dense_lu.name(),
            LinearSolverEnum::FaerSparseQr(faer_sparse_qr) => faer_sparse_qr.name(),
            LinearSolverEnum::FaerSparseLu(faer_sparse_lu) => faer_sparse_lu.name(),
            LinearSolverEnum::FaerSparseLdlt(faer_sparse_ldlt) => faer_sparse_ldlt.name(),
            LinearSolverEnum::FaerSparseLblt(s) => s.name(),
            LinearSolverEnum::SparseLdlt(sparse_ldlt) => sparse_ldlt.name(),
            LinearSolverEnum::BlockSparseLdlt(block_sparse_ldlt) => block_sparse_ldlt.name(),
            LinearSolverEnum::SchurBlockSparseLdlt(s) => format!("Schur({})", s.name()),
            LinearSolverEnum::SchurSparseLdlt(s) => format!("Schur({})", s.name()),
            LinearSolverEnum::SchurFaerSparseLdlt(s) => format!("Schur({})", s.name()),
        }
    }

    /// Return factorization of the inner direct-solve matrix.
    ///
    /// Core dispatch method used by `factorize` and `DirectSolve::solve`.
    /// Accepts `BlockSparseLower` input for `SparseLdlt` and `FaerSparseLdlt`: converts
    /// on-the-fly so callers can always use the fast `BlockSparsePattern` populate path.
    pub(crate) fn factorize_inner(
        &self,
        inner: &DirectSolveMatrix,
        cached_symb: Option<CachedSymbolicFactor>,
    ) -> Result<FactorEnum, LinearSolverError> {
        match self {
            LinearSolverEnum::DenseLdlt(dense_ldlt) => {
                let owned;
                let dense = if let DirectSolveMatrix::Dense(d) = inner {
                    d
                } else if let DirectSolveMatrix::BlockSparseLower(b) = inner {
                    owned = b.to_dense_symmetric();
                    &owned
                } else {
                    inner.as_dense().unwrap()
                };
                Ok(FactorEnum::DenseLdlt(dense_ldlt.factorize(dense)?))
            }
            LinearSolverEnum::DenseLu(dense_lu) => {
                let owned;
                let dense = if let DirectSolveMatrix::Dense(d) = inner {
                    d
                } else if let DirectSolveMatrix::BlockSparseLower(b) = inner {
                    owned = b.to_dense_symmetric();
                    &owned
                } else {
                    inner.as_dense().unwrap()
                };
                Ok(FactorEnum::DenseLu(dense_lu.factorize(dense)?))
            }
            LinearSolverEnum::SparseLdlt(sparse_ldlt) => {
                // Accept BlockSparseLower (produced by the fast BlockSparsePattern populate path)
                // in addition to SparseLower.
                let cached = cached_symb.and_then(|c| {
                    if let CachedSymbolicFactorInner::SparseLdlt(s) = c.0 {
                        Some(s)
                    } else {
                        None
                    }
                });
                let owned;
                let sparse = if let DirectSolveMatrix::SparseLower(s) = inner {
                    s
                } else if let DirectSolveMatrix::BlockSparseLower(b) = inner {
                    owned = b.to_sparse_symmetric();
                    &owned
                } else {
                    inner.as_sparse_lower().unwrap()
                };
                Ok(FactorEnum::SparseLdlt(
                    sparse_ldlt.factorize_with_cached_symb(sparse, cached)?,
                ))
            }
            LinearSolverEnum::BlockSparseLdlt(block_sparse_ldlt) => {
                Ok(FactorEnum::BlockSparseLdlt(
                    block_sparse_ldlt.factorize(inner.as_block_sparse_lower().unwrap())?,
                ))
            }
            LinearSolverEnum::FaerSparseQr(faer_sparse_qr) => {
                let owned;
                let faer = if let DirectSolveMatrix::FaerSparse(s) = inner {
                    s
                } else if let DirectSolveMatrix::BlockSparseLower(b) = inner {
                    owned = b.to_faer_sparse();
                    &owned
                } else {
                    inner.as_faer_sparse().unwrap()
                };
                Ok(FactorEnum::FaerSparseQr(faer_sparse_qr.factorize(faer)?))
            }
            LinearSolverEnum::FaerSparseLu(faer_sparse_lu) => {
                let owned;
                let faer = if let DirectSolveMatrix::FaerSparse(s) = inner {
                    s
                } else if let DirectSolveMatrix::BlockSparseLower(b) = inner {
                    owned = b.to_faer_sparse();
                    &owned
                } else {
                    inner.as_faer_sparse().unwrap()
                };
                Ok(FactorEnum::FaerSparseLu(faer_sparse_lu.factorize(faer)?))
            }
            LinearSolverEnum::FaerSparseLdlt(faer_sparse_ldlt) => {
                let cached = cached_symb.and_then(|c| {
                    if let CachedSymbolicFactorInner::FaerSparseLdlt(f) = c.0 {
                        Some(f)
                    } else {
                        None
                    }
                });
                // Accept BlockSparseLower in addition to FaerSparseUpper.
                let owned;
                let faer = if let DirectSolveMatrix::FaerSparseUpper(s) = inner {
                    s
                } else if let DirectSolveMatrix::BlockSparseLower(b) = inner {
                    // Always use the structural conversion (includes ALL scalar positions
                    // within existing blocks, even zeros). This ensures consistent sparsity
                    // across all iterations so the symbolic factor computed on iteration 1
                    // remains valid for iterations 2..N. If value-filtered conversion were
                    // used (which skips zeros), the sparsity could shrink between iterations,
                    // making the cached symbolic invalid and causing out-of-bounds panics in
                    // faer's numeric factorization.
                    owned = b.to_faer_sparse_symmetric_structural();
                    &owned
                } else {
                    inner.as_faer_sparse_upper().unwrap()
                };
                Ok(FactorEnum::FaerSparseLdlt(
                    faer_sparse_ldlt.factorize_with_cached_symb(faer, cached)?,
                ))
            }
            LinearSolverEnum::FaerSparseLblt(faer_sparse_lblt) => {
                let cached = cached_symb.and_then(|c| {
                    if let CachedSymbolicFactorInner::FaerSparseLblt(f) = c.0 {
                        Some(f)
                    } else {
                        None
                    }
                });
                // Accept BlockSparseLower in addition to FaerSparseUpper.
                let owned;
                let faer = if let DirectSolveMatrix::FaerSparseUpper(s) = inner {
                    s
                } else if let DirectSolveMatrix::BlockSparseLower(b) = inner {
                    owned = b.to_faer_sparse_symmetric_structural();
                    &owned
                } else {
                    inner.as_faer_sparse_upper().unwrap()
                };
                Ok(FactorEnum::FaerSparseLblt(
                    faer_sparse_lblt.factorize_with_cached_symb(faer, cached)?,
                ))
            }
            LinearSolverEnum::SchurBlockSparseLdlt(_)
            | LinearSolverEnum::SchurSparseLdlt(_)
            | LinearSolverEnum::SchurFaerSparseLdlt(_) => {
                panic!(
                    "use Schur<M>::solve() instead of LinearSolverEnum::factorize_inner for Schur variants"
                )
            }
        }
    }

    /// Return factorization of provided matrix `A`.
    ///
    /// Accepts `BlockSparseLower` input for `SparseLdlt` and `FaerSparseLdlt`: converts
    /// on-the-fly so callers can always use the fast `BlockSparsePattern` populate path.
    pub fn factorize(&self, mat_a: &SymmetricMatrixEnum) -> Result<FactorEnum, LinearSolverError> {
        match mat_a {
            SymmetricMatrixEnum::Direct(ds) => self.factorize_inner(&ds.inner, None),
            SymmetricMatrixEnum::Schur(_) => {
                panic!(
                    "use Schur<M>::solve() instead of LinearSolverEnum::factorize for Schur variants"
                )
            }
        }
    }

    /// Solve linear system: ``A x = b``.
    pub fn solve(
        &self,
        mat_a: &SymmetricMatrixEnum,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, LinearSolverError> {
        let system = self.factorize(mat_a)?;
        system.solve(b)
    }

    /// Solve linear system in place ``x := A⁻¹ x``.
    pub fn solve_inplace(
        &self,
        mat_a: &SymmetricMatrixEnum,
        b: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        let system = self.factorize(mat_a)?;
        system.solve_inplace(b)
    }
}

/// Factorization of matrix `A`.
pub trait IsFactor {
    /// Symmetric matrix `A`.
    type Matrix: IsSymmetricMatrix;

    /// Solve linear system: ``A x = b``.
    fn solve(
        &self,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, LinearSolverError> {
        let mut x = b.clone();
        self.solve_inplace(&mut x)?;
        Ok(x)
    }

    /// Solve linear system in place ``x := A⁻¹ x``.
    fn solve_inplace(&self, b: &mut nalgebra::DVector<f64>) -> Result<(), LinearSolverError>;
}

#[derive(Debug)]
/// Factorization of matrix `A`.
pub enum FactorEnum {
    /// Dense solver using LDLᵀ factorization.
    DenseLdlt(DenseLdltFactor),
    /// Dense solver using LU factorization.
    DenseLu(DenseLuSystem),
    /// Sparse solver using LDLᵀ factorization.
    SparseLdlt(SparseLdltFactor),
    /// Block-sparse solver using LDLᵀ factorization.
    BlockSparseLdlt(BlockSparseLdltFactor),
    /// Sparse solver using faer's QR factorization.
    FaerSparseQr(FaerSparseQrFactor),
    /// Sparse solver using faer's LU factorization.
    FaerSparseLu(FaerSparseLuSystem),
    /// Sparse solver using faer's LDLᵀ factorization.
    FaerSparseLdlt(FaerSparseLdltSystem),
    /// Sparse solver using faer's LBLᵀ (Bunch-Kaufman) factorization.
    FaerSparseLblt(FaerSparseLbltSystem),
    /// Schur-complement factorization.
    Schur(Box<SchurFactor>),
}

impl Clone for FactorEnum {
    fn clone(&self) -> Self {
        match self {
            FactorEnum::DenseLdlt(f) => FactorEnum::DenseLdlt(f.clone()),
            FactorEnum::DenseLu(f) => FactorEnum::DenseLu(f.clone()),
            FactorEnum::SparseLdlt(f) => FactorEnum::SparseLdlt(f.clone()),
            FactorEnum::BlockSparseLdlt(f) => FactorEnum::BlockSparseLdlt(f.clone()),
            FactorEnum::FaerSparseQr(f) => FactorEnum::FaerSparseQr(f.clone()),
            FactorEnum::FaerSparseLu(f) => FactorEnum::FaerSparseLu(f.clone()),
            FactorEnum::FaerSparseLdlt(f) => FactorEnum::FaerSparseLdlt(f.clone()),
            FactorEnum::FaerSparseLblt(_) => {
                panic!(
                    "FaerSparseLblt factorization cannot be cloned (SymbolicCholesky is not Clone)"
                )
            }
            FactorEnum::Schur(f) => FactorEnum::Schur(f.clone()),
        }
    }
}

#[derive(Debug)]
enum CachedSymbolicFactorInner {
    SparseLdlt(SparseLdltSymbolic),
    FaerSparseLdlt(FaerSparseLdltSymbolic),
    FaerSparseLblt(FaerSparseLbltSymbolic),
    SchurPattern(BlockSparseSymmetricMatrixPattern),
}

impl Clone for CachedSymbolicFactorInner {
    fn clone(&self) -> Self {
        match self {
            CachedSymbolicFactorInner::SparseLdlt(s) => {
                CachedSymbolicFactorInner::SparseLdlt(s.clone())
            }
            CachedSymbolicFactorInner::FaerSparseLdlt(s) => {
                CachedSymbolicFactorInner::FaerSparseLdlt(s.clone())
            }
            CachedSymbolicFactorInner::FaerSparseLblt(_) => {
                panic!(
                    "FaerSparseLblt symbolic factor cannot be cloned (SymbolicCholesky is not Clone)"
                )
            }
            CachedSymbolicFactorInner::SchurPattern(p) => {
                CachedSymbolicFactorInner::SchurPattern(p.clone())
            }
        }
    }
}

/// Cached symbolic factor (AMD ordering + sparsity analysis) for reuse across iterations.
///
/// The sparsity pattern of the Hessian does not change between optimizer iterations, so the
/// expensive AMD ordering and symbolic factorization can be computed once and reused.
#[derive(Clone, Debug)]
pub struct CachedSymbolicFactor(CachedSymbolicFactorInner);

impl CachedSymbolicFactor {
    /// Create a `CachedSymbolicFactor` from a Schur S sparsity pattern.
    pub(crate) fn from_schur_pattern(p: BlockSparseSymmetricMatrixPattern) -> Self {
        CachedSymbolicFactor(CachedSymbolicFactorInner::SchurPattern(p))
    }

    /// Extract the Schur S sparsity pattern, or `None` if this is not a `SchurPattern`.
    pub(crate) fn into_schur_pattern(self) -> Option<BlockSparseSymmetricMatrixPattern> {
        if let CachedSymbolicFactorInner::SchurPattern(p) = self.0 {
            Some(p)
        } else {
            None
        }
    }
}

impl FactorEnum {
    /// Extract the cached symbolic factor for reuse in the next iteration.
    pub fn into_symbolic(self) -> Option<CachedSymbolicFactor> {
        match self {
            FactorEnum::SparseLdlt(f) => f
                .into_symbolic()
                .map(|s| CachedSymbolicFactor(CachedSymbolicFactorInner::SparseLdlt(s))),
            FactorEnum::FaerSparseLdlt(f) => Some(CachedSymbolicFactor(
                CachedSymbolicFactorInner::FaerSparseLdlt(f.into_symbolic()),
            )),
            FactorEnum::FaerSparseLblt(f) => Some(CachedSymbolicFactor(
                CachedSymbolicFactorInner::FaerSparseLblt(f.into_symbolic()),
            )),
            // S pattern is extracted from SchurFactor before wrapping; nothing to pull out here.
            FactorEnum::Schur(_) => None,
            _ => None,
        }
    }
}

impl FactorEnum {
    /// Return matrix factorization to compute the (pseudo) inverse of `A`.
    pub fn into_invertible(self) -> Option<InvertibleMatrix> {
        match self {
            FactorEnum::DenseLdlt(factor) => {
                Some(InvertibleMatrix::Dense(DenseMinNormFactor::new(factor)))
            }
            FactorEnum::BlockSparseLdlt(factor) => Some(InvertibleMatrix::BlockSparse(
                BlockSparseMinNormPsd::new(factor),
            )),
            FactorEnum::SparseLdlt(factor) => {
                Some(InvertibleMatrix::Sparse(SparseMinNormPsd::new(factor)))
            }
            FactorEnum::Schur(sf) => Some(InvertibleMatrix::Schur(sf)),
            _ => None,
        }
    }

    /// Return a mutable reference to the inner `SchurFactor`, if this is a `Schur` variant.
    pub(crate) fn as_schur_mut(&mut self) -> Option<&mut SchurFactor> {
        if let FactorEnum::Schur(sf) = self {
            Some(sf.as_mut())
        } else {
            None
        }
    }
}

impl IsFactor for FactorEnum {
    type Matrix = SymmetricMatrixEnum;

    fn solve_inplace(&self, b: &mut nalgebra::DVector<f64>) -> Result<(), LinearSolverError> {
        match self {
            FactorEnum::DenseLdlt(dense_ldlt_system) => dense_ldlt_system.solve_inplace(b),
            FactorEnum::DenseLu(dense_lu_system) => dense_lu_system.solve_inplace(b),
            FactorEnum::SparseLdlt(sparse_ldlt_system) => sparse_ldlt_system.solve_inplace(b),
            FactorEnum::BlockSparseLdlt(block_sparse_ldlt_system) => {
                block_sparse_ldlt_system.solve_inplace(b)
            }
            FactorEnum::FaerSparseQr(faer_sparse_qr_system) => {
                faer_sparse_qr_system.solve_inplace(b)
            }
            FactorEnum::FaerSparseLu(faer_sparse_lu_system) => {
                faer_sparse_lu_system.solve_inplace(b)
            }
            FactorEnum::FaerSparseLdlt(faer_sparse_ldlt_system) => {
                faer_sparse_ldlt_system.solve_inplace(b)
            }
            FactorEnum::FaerSparseLblt(s) => s.solve_inplace(b),
            FactorEnum::Schur(sf) => {
                let dx = sf.solve()?;
                b.copy_from(&dx);
                Ok(())
            }
        }
    }
}
/// Matrix factorization to compute invertible solutions, i.e. the (pseudo) inverse of `A`.
pub trait IsInvertible {
    /// Return the (Moore–Penrose) pseudo-inverse.
    fn pseudo_inverse(&mut self) -> DMatrix<f64>;

    /// Get block at index `(row_idx, col_idx)` of the (Moore–Penrose) pseudo-inverse.
    fn pseudo_inverse_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> nalgebra::DMatrix<f64>;
}

#[derive(Clone, Debug)]
/// Matrix factorization to compute invertible solutions, i.e. the (pseudo) inverse of `A`.
pub enum InvertibleMatrix {
    /// Dense min-norm factorization, based on LDLᵀ.
    Dense(DenseMinNormFactor),
    /// Sparse min-norm factorization, based on LDLᵀ
    Sparse(SparseMinNormPsd),
    /// Block-sparse min-norm factorization, based on LDLᵀ
    BlockSparse(BlockSparseMinNormPsd),
    /// Schur-complement factorization.
    Schur(Box<SchurFactor>),
}

impl IsInvertible for InvertibleMatrix {
    fn pseudo_inverse(&mut self) -> DMatrix<f64> {
        match self {
            InvertibleMatrix::Dense(dense_gram_schmidt) => dense_gram_schmidt.pseudo_inverse(),
            InvertibleMatrix::Sparse(min_norm_psd) => min_norm_psd.pseudo_inverse(),
            InvertibleMatrix::BlockSparse(min_norm_psd) => min_norm_psd.pseudo_inverse(),
            InvertibleMatrix::Schur(sf) => {
                let partitions = sf.full_partitions.clone();
                let n = partitions.scalar_dim();
                let mut result = DMatrix::<f64>::zeros(n, n);
                for row_p in 0..partitions.len() {
                    for row_b in 0..partitions.specs()[row_p].block_count {
                        let row_idx = PartitionBlockIndex {
                            partition: row_p,
                            block: row_b,
                        };
                        let row_range = partitions.block_range(row_idx);
                        for col_p in 0..partitions.len() {
                            for col_b in 0..partitions.specs()[col_p].block_count {
                                let col_idx = PartitionBlockIndex {
                                    partition: col_p,
                                    block: col_b,
                                };
                                let col_range = partitions.block_range(col_idx);
                                let block = sf.inverse_block(row_idx, col_idx);
                                result
                                    .view_mut(
                                        (row_range.start_idx, col_range.start_idx),
                                        (row_range.block_dim, col_range.block_dim),
                                    )
                                    .copy_from(&block);
                            }
                        }
                    }
                }
                result
            }
        }
    }

    fn pseudo_inverse_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> nalgebra::DMatrix<f64> {
        match self {
            InvertibleMatrix::Dense(dense_gram_schmidt) => {
                dense_gram_schmidt.pseudo_inverse_block(row_idx, col_idx)
            }
            InvertibleMatrix::Sparse(min_norm_psd) => {
                min_norm_psd.pseudo_inverse_block(row_idx, col_idx)
            }
            InvertibleMatrix::BlockSparse(min_norm_psd) => {
                min_norm_psd.pseudo_inverse_block(row_idx, col_idx)
            }
            InvertibleMatrix::Schur(sf) => sf.inverse_block(row_idx, col_idx),
        }
    }
}

/// sophus_solver prelude.
///
/// It is recommended to import this prelude when working with `sophus_solver` types:
///
/// ```
/// use sophus_solver::prelude::*;
/// ```
///
/// or
///
/// ```ignore
/// use sophus::prelude::*;
/// ```
///
/// to import all preludes when using the `sophus` umbrella crate.
pub mod prelude {
    pub use sophus_autodiff::prelude::*;

    pub use crate::{
        IsLinearSolver,
        matrix::{
            IsSymmetricMatrix,
            IsSymmetricMatrixBuilder,
        },
    };
}
