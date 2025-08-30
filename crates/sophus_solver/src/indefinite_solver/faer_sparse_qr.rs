use faer::{
    prelude::Solve,
    sparse::FaerError,
};

use crate::{
    LinearSolverError,
    SparseSolverError,
    SymmetricMatrixBuilderEnum,
    prelude::*,
    sparse::{
        SparseSymmetricMatrixBuilder,
        faer_sparse_matrix::{
            FaerCompressedMatrix,
            FaerTripletMatrix,
        },
    },
};

/// Sparse QR solver
///
/// Sparse QR decomposition - wrapper around faer's sp_qr implementation.
#[derive(Copy, Clone, Debug)]

pub struct FaerSparseQr;

impl IsLinearSolver for FaerSparseQr {
    type Matrix = FaerTripletMatrix;

    const NAME: &'static str = "faer sparse QR";

    fn matrix_builder(&self, partitions: &[crate::PartitionSpec]) -> SymmetricMatrixBuilderEnum {
        SymmetricMatrixBuilderEnum::FaerSparse(SparseSymmetricMatrixBuilder::zero(partitions))
    }

    fn solve_in_place(
        &self,
        _parallelize: bool,
        mat: &FaerCompressedMatrix,
        b: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        let x_slice_mut = b.as_mut_slice();
        let mut x_ref =
            faer::MatMut::<f64>::from_column_major_slice_mut(x_slice_mut, mat.csc.nrows(), 1);

        match mat.csc.sp_qr() {
            Ok(qr) => {
                qr.solve_in_place(faer::reborrow::ReborrowMut::rb_mut(&mut x_ref));
                Ok(())
            }
            Err(e) => Err(LinearSolverError::SparseQrError {
                details: match e {
                    FaerError::IndexOverflow => SparseSolverError::IndexOverflow,
                    FaerError::OutOfMemory => SparseSolverError::OutOfMemory,
                    _ => SparseSolverError::Unspecific,
                },
            }),
        }
    }
}
