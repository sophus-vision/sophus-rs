use faer::{
    prelude::Solve,
    sparse::FaerError,
};

use crate::{
    IsSparseSymmetricLinearSystem,
    IsSymmetricMatrixBuilder,
    LinearSolverError,
    SparseSolverError,
    SymmetricBlockSparseMatrixBuilder,
};

/// Sparse QR solver
///
/// Sparse QR decomposition - wrapper around faer's sp_qr implementation.
pub struct SparseQr;

impl IsSparseSymmetricLinearSystem for SparseQr {
    fn solve(
        &self,
        sym_mat: &SymmetricBlockSparseMatrixBuilder,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, LinearSolverError> {
        let dim = sym_mat.scalar_dimension();
        let csc = faer::sparse::SparseColMat::try_new_from_triplets(
            dim,
            dim,
            &sym_mat.to_symmetric_scalar_triplets(),
        )
        .unwrap();
        let mut x = b.clone();
        let x_slice_mut = x.as_mut_slice();
        let mut x_ref = faer::MatMut::<f64>::from_column_major_slice_mut(x_slice_mut, dim, 1);

        match csc.sp_qr() {
            Ok(qr) => {
                qr.solve_in_place(faer::reborrow::ReborrowMut::rb_mut(&mut x_ref));
                Ok(x)
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
