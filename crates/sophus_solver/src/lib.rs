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
/// Dense data structures.
pub mod compressed_matrix;
/// Dense data structures.
pub mod dense;
/// Block-sparse data structures.
pub mod error;
/// grid.
pub mod grid;
/// LU to solve invertible systems.
pub mod indefinite_solver;
/// LDLt to solve semi-positive definite systems.
pub mod psd_solver;
/// Sparse data structures.
pub mod sparse;
/// Block-sparse data structures.
pub mod symmetric_matrix;

mod asserts;

pub use block_sparse::*;
pub use compressed_matrix::*;
pub use error::*;
pub use indefinite_solver::{
    dense_lu::*,
    faer_sparse_lu::*,
    sparse_qr::*,
};
pub use psd_solver::{
    dense_ldlt::*,
    sparse_ldlt::*,
};
pub use symmetric_matrix::*;

use crate::psd_solver::{
    block_sparse_ldlt::BlockSparseLdlt,
    faer_sparse_ldlt::FaerSparseLdlt,
};

/// Linear solver of linear system.
pub trait IsLinearSolver {
    /// mat
    type Matrix: IsSymmetricMatrix;

    /// Solve the linear system.
    fn solve(
        &self,
        matrix: &<Self::Matrix as IsSymmetricMatrix>::Compressed,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, LinearSolverError> {
        let mut x = b.clone();
        self.solve_in_place(matrix, &mut x)?;
        Ok(x)
    }

    /// Solve the linear system in-place.
    fn solve_in_place(
        &self,
        matrix: &<Self::Matrix as IsSymmetricMatrix>::Compressed,
        b: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError>;
}

/// l
#[derive(Copy, Clone, Debug)]
pub enum LinearSolverEnum {
    ///d
    DenseLdlt(DenseLdlt),
    ///d
    DenseLu(DenseLu),
    /// s
    SparseLdlt(SparseLdlt),
    /// s
    BlockSparseLdlt(BlockSparseLdlt),
    /// s
    FaerSparseQr(FaerSparseQr),
    /// s
    FaerSparseLu(FaerSparseLu),
    /// s
    FaerSparseLdlt(FaerSparseLdlt),
}

impl Default for LinearSolverEnum {
    fn default() -> Self {
        LinearSolverEnum::SparseLdlt(SparseLdlt::default())
    }
}

impl IsLinearSolver for LinearSolverEnum {
    type Matrix = SymmetricMatrixEnum;

    fn solve_in_place(
        &self,
        matrix: &<Self::Matrix as IsSymmetricMatrix>::Compressed,
        b: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        match (self, matrix) {
            (LinearSolverEnum::DenseLdlt(dense_ldlt), CompressedMatrixEnum::Dense(matrix)) => {
                dense_ldlt.solve_in_place(matrix, b)
            }
            (LinearSolverEnum::DenseLu(dense_lu), CompressedMatrixEnum::Dense(matrix)) => {
                dense_lu.solve_in_place(matrix, b)
            }
            (
                LinearSolverEnum::SparseLdlt(sparse_ldlt),
                CompressedMatrixEnum::SparseLower(matrix),
            ) => sparse_ldlt.solve_in_place(matrix, b),
            (
                LinearSolverEnum::BlockSparseLdlt(block_sparse_ldlt),
                CompressedMatrixEnum::BlockSparseLower(matrix),
            ) => block_sparse_ldlt.solve_in_place(matrix, b),
            (
                LinearSolverEnum::FaerSparseLdlt(faer_sparse_ldlt),
                CompressedMatrixEnum::FaerSparseUpper(matrix),
            ) => faer_sparse_ldlt.solve_in_place(matrix, b),
            (
                LinearSolverEnum::FaerSparseQr(faer_sparse_qr),
                CompressedMatrixEnum::FaerSparse(matrix),
            ) => faer_sparse_qr.solve_in_place(matrix, b),
            (
                LinearSolverEnum::FaerSparseLu(faer_sparse_lu),
                CompressedMatrixEnum::FaerSparse(matrix),
            ) => faer_sparse_lu.solve_in_place(matrix, b),

            _ => panic!("{self:?}"),
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
    ///d
    pub fn dense_solvers() -> Vec<LinearSolverEnum> {
        vec![
            LinearSolverEnum::DenseLdlt(DenseLdlt {}),
            LinearSolverEnum::DenseLu(DenseLu {}),
        ]
    }

    /// Get all sparse solvers
    pub fn sparse_solvers() -> Vec<LinearSolverEnum> {
        vec![
            LinearSolverEnum::SparseLdlt(Default::default()),
            LinearSolverEnum::BlockSparseLdlt(BlockSparseLdlt {}),
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
