use faer::{
    prelude::*,
    sparse::{
        FaerError,
        linalg::solvers::Qr,
    },
};

use crate::{
    FaerSparseSolverError,
    IsFactor,
    LinearSolverEnum,
    LinearSolverError,
    matrix::{
        PartitionSet,
        SymmetricMatrixBuilderEnum,
        sparse::{
            FaerSparseMatrix,
            FaerSparseMatrixBuilder,
        },
    },
    prelude::*,
};

/// Sparse QR solver - wrapper around faer's sp_qr implementation.
#[derive(Copy, Clone, Debug)]

pub struct FaerSparseQr;

/// Sparse QR factorization - using thr faer crate.
#[derive(Clone, Debug)]
pub struct FaerSparseQrFactor {
    qr: Qr<usize, f64>,
    n: usize,
}

impl IsLinearSolver for FaerSparseQr {
    type SymmetricMatrixBuilder = FaerSparseMatrixBuilder;
    type Factor = FaerSparseQrFactor;

    const NAME: &'static str = "faer sparse QR";

    fn factorize(&self, mat_a: &FaerSparseMatrix) -> Result<Self::Factor, LinearSolverError> {
        match mat_a.square.sp_qr() {
            Ok(qr) => Ok(FaerSparseQrFactor {
                qr,
                n: mat_a.square.nrows(),
            }),
            Err(e) => Err(LinearSolverError::FaerSparseQrError {
                faer_error: match e {
                    FaerError::IndexOverflow => FaerSparseSolverError::IndexOverflow,
                    FaerError::OutOfMemory => FaerSparseSolverError::OutOfMemory,
                    _ => FaerSparseSolverError::Unspecific,
                },
            }),
        }
    }

    fn zero(&self, partitions: PartitionSet) -> SymmetricMatrixBuilderEnum {
        SymmetricMatrixBuilderEnum::FaerSparse(
            FaerSparseMatrixBuilder::zero(partitions),
            LinearSolverEnum::FaerSparseQr(*self),
        )
    }

    /// Does not support parallel execution.
    fn set_parallelize(&mut self, _parallelize: bool) {
        // no-op
    }
}

impl IsFactor for FaerSparseQrFactor {
    type Matrix = FaerSparseMatrix;

    fn solve_inplace(&self, b: &mut nalgebra::DVector<f64>) -> Result<(), LinearSolverError> {
        let x_slice_mut = b.as_mut_slice();
        let mut x_ref = faer::MatMut::<f64>::from_column_major_slice_mut(x_slice_mut, self.n, 1);
        self.qr
            .solve_in_place(faer::reborrow::ReborrowMut::rb_mut(&mut x_ref));
        Ok(())
    }
}
