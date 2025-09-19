#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]
#![cfg_attr(nightly, feature(doc_auto_cfg))]

#[doc = include_str!(concat!("../",  core::env!("CARGO_PKG_README")))]
#[cfg(doctest)]
pub struct ReadmeDoctests;

use std::fmt::Debug;

pub use error::*;
use nalgebra::DMatrix;

use crate::{
    ldlt::{
        BlockSparseLdlt,
        BlockSparseLdltFactor,
        DenseLdlt,
        DenseLdltFactor,
        FaerSparseLdlt,
        FaerSparseLdltSystem,
        IntoMinNormPsd,
        SparseLdlt,
        SparseLdltFactor,
        min_norm_ldlt::{
            block_sparse_min_norm_ldlt::BlockSparseMinNormPsd,
            dense_min_norm_ldlt::DenseMinNormFactor,
            sparse_min_norm_ldlt::SparseMinNormPsd,
        },
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
        dense::DenseSymmetricMatrixBuilder,
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

    /// Get list of all sparse solvers
    pub fn sparse_solvers() -> Vec<LinearSolverEnum> {
        vec![
            LinearSolverEnum::SparseLdlt(SparseLdlt::default()),
            LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt::default()),
            LinearSolverEnum::FaerSparseQr(FaerSparseQr {}),
            LinearSolverEnum::FaerSparseLu(FaerSparseLu {}),
            LinearSolverEnum::FaerSparseLdlt(FaerSparseLdlt::default()),
        ]
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
            LinearSolverEnum::DenseLdlt(_) => {
                SymmetricMatrixBuilderEnum::Dense(DenseSymmetricMatrixBuilder::zero(partitions))
            }
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

    /// Return factorization of provided matrix `A`.
    pub fn factorize(&self, mat_a: &SymmetricMatrixEnum) -> Result<FactorEnum, LinearSolverError> {
        match self {
            LinearSolverEnum::DenseLdlt(dense_ldlt) => Ok(FactorEnum::DenseLdlt(
                dense_ldlt.factorize(mat_a.as_dense().unwrap())?,
            )),
            LinearSolverEnum::DenseLu(dense_lu) => Ok(FactorEnum::DenseLu(
                dense_lu.factorize(mat_a.as_dense().unwrap())?,
            )),
            LinearSolverEnum::SparseLdlt(sparse_ldlt) => Ok(FactorEnum::SparseLdlt(
                sparse_ldlt.factorize(mat_a.as_sparse_lower().unwrap())?,
            )),
            LinearSolverEnum::BlockSparseLdlt(block_sparse_ldlt) => {
                Ok(FactorEnum::BlockSparseLdlt(
                    block_sparse_ldlt.factorize(mat_a.as_block_sparse_lower().unwrap())?,
                ))
            }
            LinearSolverEnum::FaerSparseQr(faer_sparse_qr) => Ok(FactorEnum::FaerSparseQr(
                faer_sparse_qr.factorize(mat_a.as_faer_sparse().unwrap())?,
            )),
            LinearSolverEnum::FaerSparseLu(faer_sparse_lu) => Ok(FactorEnum::FaerSparseLu(
                faer_sparse_lu.factorize(mat_a.as_faer_sparse().unwrap())?,
            )),
            LinearSolverEnum::FaerSparseLdlt(faer_sparse_ldlt) => Ok(FactorEnum::FaerSparseLdlt(
                faer_sparse_ldlt.factorize(mat_a.as_faer_sparse_upper().unwrap())?,
            )),
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

impl FactorEnum {
    /// Return matrix factorization to compute the (pseudo) inverse of `A`.
    pub fn into_min_norm_factor(self) -> Option<MinNormSystemEnum> {
        match self {
            FactorEnum::DenseLdlt(dense_ldlt_system) => Some(MinNormSystemEnum::Dense(
                dense_ldlt_system.into_min_norm_ldlt(),
            )),
            FactorEnum::BlockSparseLdlt(factor) => {
                Some(MinNormSystemEnum::BlockSparse(factor.into_min_norm_ldlt()))
            }
            FactorEnum::SparseLdlt(factor) => {
                Some(MinNormSystemEnum::Sparse(factor.into_min_norm_ldlt()))
            }

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
/// Matrix factorization to compute min-norm solutions, i.e. the (pseudo) inverse of `A`.
pub trait IsMinNormFactor {
    /// Return the (Moore–Penrose) pseudo-inverse.
    fn pseudo_inverse(&self) -> DMatrix<f64>;

    /// Get block at index `(row_idx, col_idx)` of the (Moore–Penrose) pseudo-inverse.`
    fn pseudo_inverse_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> nalgebra::DMatrix<f64>;
}

#[derive(Clone, Debug)]
/// Matrix factorization to compute min-norm solutions, i.e. the (pseudo) inverse of `A`.x
pub enum MinNormSystemEnum {
    /// Dense min-norm factorization, based on LDLᵀ.
    Dense(DenseMinNormFactor),
    /// Sparse min-norm factorization, based on LDLᵀ
    Sparse(SparseMinNormPsd),
    /// Block-sparse min-norm factorization, based on LDLᵀ
    BlockSparse(BlockSparseMinNormPsd),
}

impl IsMinNormFactor for MinNormSystemEnum {
    fn pseudo_inverse(&self) -> DMatrix<f64> {
        match self {
            MinNormSystemEnum::Dense(dense_gram_schmidt) => dense_gram_schmidt.pseudo_inverse(),
            MinNormSystemEnum::Sparse(min_norm_psd) => min_norm_psd.pseudo_inverse(),
            MinNormSystemEnum::BlockSparse(min_norm_psd) => min_norm_psd.pseudo_inverse(),
        }
    }

    fn pseudo_inverse_block(
        &mut self,
        row_idx: PartitionBlockIndex,
        col_idx: PartitionBlockIndex,
    ) -> nalgebra::DMatrix<f64> {
        match self {
            MinNormSystemEnum::Dense(dense_gram_schmidt) => {
                dense_gram_schmidt.pseudo_inverse_block(row_idx, col_idx)
            }
            MinNormSystemEnum::Sparse(min_norm_psd) => {
                min_norm_psd.pseudo_inverse_block(row_idx, col_idx)
            }
            MinNormSystemEnum::BlockSparse(min_norm_psd) => {
                min_norm_psd.pseudo_inverse_block(row_idx, col_idx)
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
