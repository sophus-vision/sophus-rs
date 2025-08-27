#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]
#![cfg_attr(nightly, feature(doc_auto_cfg))]

#[doc = include_str!(concat!("../",  core::env!("CARGO_PKG_README")))]
#[cfg(doctest)]
pub struct ReadmeDoctests;

use std::fmt::Debug;

/// Block-sparse data structures.
pub mod block_sparse;
/// Compressed sparse matrix.
pub mod compressed_matrix;
/// Compressible matrix.s
pub mod compressible_matrix;
/// Dense data structures.
pub mod dense;
/// Solver errors.
pub mod error;
/// Grid structure.
pub mod grid;
/// Solve possibly indefinite linear systems.
pub mod indefinite_solver;
/// Solver semi-positive definite (and hence symmetric) systems.
pub mod positive_semidefinite_solver;
/// Sparse data structures.
pub mod sparse;
/// Symmetric matrix.
pub mod symmetric_matrix;

mod asserts;

use crate::positive_semidefinite_solver::sparse_ldlt::SparseLdlt;
pub use crate::{
    block_sparse::*,
    compressed_matrix::*,
    compressible_matrix::{
        CompressibleMatrixEnum,
        IsCompressibleMatrix,
    },
    error::*,
    indefinite_solver::{
        dense_lu::*,
        faer_sparse_lu::*,
        faer_sparse_qr::*,
    },
    positive_semidefinite_solver::{
        dense_ldlt::*,
        faer_sparse_ldlt::FaerSparseLdlt,
    },
    symmetric_matrix::*,
};

/// Linear solver of linear system.
pub trait IsLinearSolver {
    /// Compressible sparse matrix.
    type Matrix: IsCompressibleMatrix;

    /// Name of solver variant.
    const NAME: &'static str;

    /// Create matrix builder compatible with this solver.
    fn matrix_builder(&self, partitions: &[PartitionSpec]) -> SymmetricMatrixBuilderEnum;

    /// Solve the linear system.
    fn solve(
        &self,
        parallelize: bool,
        matrix: &<Self::Matrix as IsCompressibleMatrix>::Compressed,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, LinearSolverError> {
        let mut x = b.clone();
        self.solve_in_place(parallelize, matrix, &mut x)?;
        Ok(x)
    }

    /// Name of solver variant.
    fn name(&self) -> String {
        Self::NAME.into()
    }

    /// Solve the linear system in-place.
    fn solve_in_place(
        &self,
        parallelize: bool,
        matrix: &<Self::Matrix as IsCompressibleMatrix>::Compressed,
        b: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError>;
}

/// Linear solver enum.
#[derive(Copy, Clone, Debug)]
pub enum LinearSolverEnum {
    /// Dense solver using LDLᵀ factorization.
    DenseLdlt(DenseLdlt),
    /// Dense solver using LU factorization.
    DenseLu(DenseLu),
    /// Sparse solver using LDLᵀ factorization.
    SparseLdlt(SparseLdlt),
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

impl IsLinearSolver for LinearSolverEnum {
    type Matrix = CompressibleMatrixEnum;

    const NAME: &'static str = "enum";

    fn matrix_builder(&self, partitions: &[PartitionSpec]) -> SymmetricMatrixBuilderEnum {
        match self {
            LinearSolverEnum::DenseLdlt(dense_ldlt) => dense_ldlt.matrix_builder(partitions),
            LinearSolverEnum::DenseLu(dense_lu) => dense_lu.matrix_builder(partitions),
            LinearSolverEnum::FaerSparseQr(faer_sparse_qr) => {
                faer_sparse_qr.matrix_builder(partitions)
            }
            LinearSolverEnum::FaerSparseLu(faer_sparse_lu) => {
                faer_sparse_lu.matrix_builder(partitions)
            }
            LinearSolverEnum::FaerSparseLdlt(faer_sparse_ldlt) => {
                faer_sparse_ldlt.matrix_builder(partitions)
            }
            LinearSolverEnum::SparseLdlt(sparse_ldlt) => sparse_ldlt.matrix_builder(partitions),
        }
    }

    fn solve_in_place(
        &self,
        parallelize: bool,
        matrix: &<Self::Matrix as IsCompressibleMatrix>::Compressed,
        b: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        match self {
            LinearSolverEnum::DenseLdlt(dense_ldlt) => {
                dense_ldlt.solve_in_place(parallelize, matrix.as_dense().unwrap(), b)
            }
            LinearSolverEnum::DenseLu(dense_lu) => {
                dense_lu.solve_in_place(parallelize, matrix.as_dense().unwrap(), b)
            }
            LinearSolverEnum::FaerSparseLdlt(faer_sparse_ldlt) => faer_sparse_ldlt.solve_in_place(
                parallelize,
                matrix.as_faer_sparse_upper().unwrap(),
                b,
            ),
            LinearSolverEnum::FaerSparseQr(faer_sparse_qr) => {
                faer_sparse_qr.solve_in_place(parallelize, matrix.as_faer_sparse().unwrap(), b)
            }
            LinearSolverEnum::FaerSparseLu(faer_sparse_lu) => {
                faer_sparse_lu.solve_in_place(parallelize, matrix.as_faer_sparse().unwrap(), b)
            }
            LinearSolverEnum::SparseLdlt(sparse_ldlt) => {
                sparse_ldlt.solve_in_place(parallelize, matrix.as_sparse_lower().unwrap(), b)
            }
        }
    }

    fn name(&self) -> String {
        match self {
            LinearSolverEnum::DenseLdlt(dense_ldlt) => dense_ldlt.name(),
            LinearSolverEnum::DenseLu(dense_lu) => dense_lu.name(),
            LinearSolverEnum::FaerSparseQr(faer_sparse_qr) => faer_sparse_qr.name(),
            LinearSolverEnum::FaerSparseLu(faer_sparse_lu) => faer_sparse_lu.name(),
            LinearSolverEnum::FaerSparseLdlt(faer_sparse_ldlt) => faer_sparse_ldlt.name(),
            LinearSolverEnum::SparseLdlt(sparse_ldlt) => sparse_ldlt.name(),
        }
    }
}

impl LinearSolverEnum {
    /// Get all available solvers
    pub fn all_solvers() -> Vec<LinearSolverEnum> {
        let mut solvers = LinearSolverEnum::dense_solvers();
        solvers.extend(&LinearSolverEnum::sparse_solvers());
        solvers
    }
    /// Get list off all dense solvers.
    pub fn dense_solvers() -> Vec<LinearSolverEnum> {
        vec![
            LinearSolverEnum::DenseLdlt(DenseLdlt {}),
            LinearSolverEnum::DenseLu(DenseLu {}),
        ]
    }

    /// Get list of all sparse solvers
    pub fn sparse_solvers() -> Vec<LinearSolverEnum> {
        vec![
            LinearSolverEnum::SparseLdlt(SparseLdlt::default()),
            LinearSolverEnum::FaerSparseQr(FaerSparseQr {}),
            LinearSolverEnum::FaerSparseLu(FaerSparseLu {}),
            LinearSolverEnum::FaerSparseLdlt(FaerSparseLdlt::default()),
        ]
    }

    /// Get solvers which can be used for indefinite systems
    pub fn indefinite_solvers() -> Vec<LinearSolverEnum> {
        vec![
            LinearSolverEnum::DenseLu(DenseLu {}),
            LinearSolverEnum::FaerSparseQr(FaerSparseQr {}),
            LinearSolverEnum::FaerSparseLu(FaerSparseLu {}),
        ]
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
        IsCompressibleMatrix,
        IsLinearSolver,
        IsSymmetricMatrixBuilder,
    };
}
