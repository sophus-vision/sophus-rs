use faer::{
    MatMut,
    prelude::{
        ReborrowMut,
        Solve,
    },
    sparse::{
        FaerError,
        SparseColMat,
        linalg::LuError,
    },
};

use super::{
    IsSparseSymmetricLinearSystem,
    NllsError,
    SparseSolverError,
};
use crate::block::symmetric_block_sparse_matrix_builder::SymmetricBlockSparseMatrixBuilder;

/// Sparse LU solver
///
/// Sparse LU decomposition - wrapper around faer's sp_lu implementation.
pub struct SparseLu;

impl IsSparseSymmetricLinearSystem for SparseLu {
    fn solve(
        &self,
        upper_triangular: &SymmetricBlockSparseMatrixBuilder,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, NllsError> {
        let dim = upper_triangular.scalar_dimension();
        let csc: SparseColMat<usize, f64> = SparseColMat::try_new_from_triplets(
            dim,
            dim,
            &upper_triangular.to_symmetric_scalar_triplets(),
        )
        .unwrap();
        let mut x = b.clone();
        let mut x_ref = MatMut::<f64>::from_column_major_slice_mut(x.as_mut_slice(), dim, 1);

        match csc.sp_lu() {
            Ok(lu) => {
                lu.solve_in_place(ReborrowMut::rb_mut(&mut x_ref));
                Ok(x)
            }
            Err(e) => Err(NllsError::SparseLuError {
                details: match e {
                    LuError::Generic(faer_err) => match faer_err {
                        FaerError::IndexOverflow => SparseSolverError::IndexOverflow,
                        FaerError::OutOfMemory => SparseSolverError::OutOfMemory,
                        _ => SparseSolverError::Unspecific,
                    },
                    LuError::SymbolicSingular { .. } => SparseSolverError::SymbolicSingular,
                },
            }),
        }
    }
}
