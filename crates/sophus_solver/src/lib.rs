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

pub use error::*;

use crate::{
    ldlt::{
        BlockSparseLdlt,
        BlockSparseLdltFactor,
        DenseLdlt,
        DenseLdltFactor,
        FaerSparseLdlt,
        FaerSparseLdltSystem,
        SparseLdlt,
        SparseLdltFactor,
        faer_sparse_ldlt::FaerSparseLdltSymbolic,
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
        PartitionSet,
        SymmetricMatrixBuilderEnum,
        SymmetricMatrixEnum,
        direct_solve::DirectSolveMatrix,
    },
    qr::{
        FaerSparseQr,
        FaerSparseQrFactor,
    },
};

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
///         block_dim: 4,
///         block_count: 3,
///     },
///     PartitionSpec {
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
        ]
    }

    /// Returns true if this is a Schur-complement variant.
    pub fn is_schur(&self) -> bool {
        false
    }

    /// For Schur variants, returns the inner solver used for S; panics for non-Schur variants.
    pub fn schur_inner_solver(&self) -> LinearSolverEnum {
        panic!("not a Schur variant")
    }

    /// Get solvers which can be used for indefinite linear systems.
    pub fn indefinite_solvers() -> Vec<LinearSolverEnum> {
        vec![
            LinearSolverEnum::DenseLu(DenseLu {}),
            LinearSolverEnum::FaerSparseQr(FaerSparseQr {}),
            LinearSolverEnum::FaerSparseLu(FaerSparseLu {}),
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
            LinearSolverEnum::BlockSparseLdlt(block_sparse_ldlt) => {
                block_sparse_ldlt.zero(partitions)
            }
            LinearSolverEnum::SparseLdlt(sparse_ldlt) => sparse_ldlt.zero(partitions),
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
            LinearSolverEnum::SparseLdlt(sparse_ldlt) => sparse_ldlt.name(),
            LinearSolverEnum::BlockSparseLdlt(block_sparse_ldlt) => block_sparse_ldlt.name(),
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
                let owned;
                let faer = if let DirectSolveMatrix::FaerSparseUpper(s) = inner {
                    s
                } else if let DirectSolveMatrix::BlockSparseLower(b) = inner {
                    owned = b.to_faer_sparse_symmetric_structural();
                    &owned
                } else {
                    inner.as_faer_sparse_upper().unwrap()
                };
                Ok(FactorEnum::FaerSparseLdlt(
                    faer_sparse_ldlt.factorize_with_cached_symb(faer, cached)?,
                ))
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

#[derive(Clone, Debug)]
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
}

#[derive(Clone, Debug)]
enum CachedSymbolicFactorInner {
    SparseLdlt(SparseLdltSymbolic),
    FaerSparseLdlt(FaerSparseLdltSymbolic),
}

/// Cached symbolic factor (AMD ordering + sparsity analysis) for reuse across iterations.
///
/// The sparsity pattern of the Hessian does not change between optimizer iterations, so the
/// expensive AMD ordering and symbolic factorization can be computed once and reused.
#[derive(Clone, Debug)]
pub struct CachedSymbolicFactor(CachedSymbolicFactorInner);

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
            _ => None,
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
        }
    }
}

/// sophus_solver prelude.
///
/// It is recommended to import this prelude when working with `sophus_solver types:
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
