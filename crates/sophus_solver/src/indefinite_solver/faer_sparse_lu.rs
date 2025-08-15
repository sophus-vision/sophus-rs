use faer::{
    MatMut,
    prelude::{
        ReborrowMut,
        Solve,
    },
    sparse::{
        FaerError,
        linalg::LuError,
    },
};

use crate::{
    IsLinearSolver,
    LinearSolverError,
    SparseSolverError,
    sparse::faer_sparse_matrix::{
        FaerCompressedMatrix,
        FaerTripletsMatrix,
    },
};

/// Sparse LU solver
///
/// Sparse LU decomposition - wrapper around faer's sp_lu implementation.
#[derive(Copy, Clone, Debug)]

pub struct FaerSparseLu;

impl IsLinearSolver for FaerSparseLu {
    type Matrix = FaerTripletsMatrix;

    const NAME: &'static str = "faer sparse LU";

    fn solve_in_place(
        &self,
        mat: &FaerCompressedMatrix,
        b: &mut nalgebra::DVector<f64>,
    ) -> Result<(), LinearSolverError> {
        let mut x_ref =
            MatMut::<f64>::from_column_major_slice_mut(b.as_mut_slice(), mat.csc.nrows(), 1);

        match mat.csc.sp_lu() {
            Ok(lu) => {
                lu.solve_in_place(ReborrowMut::rb_mut(&mut x_ref));
                Ok(())
            }
            Err(e) => Err(LinearSolverError::SparseLuError {
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

// impl IsSparseSymmetricLinearSystem for FearSparseLu {
//     fn solve(
//         &self,
//         sym_mat: &BlockSparseSymmetricMatrixBuilder,
//         b: &nalgebra::DVector<f64>,
//     ) -> Result<nalgebra::DVector<f64>, LinearSolverError> {
//         let dim = sym_mat.scalar_dimension();
//         let csc: SparseColMat<usize, f64> =
//             SparseColMat::try_new_from_triplets(dim, dim,
// &sym_mat.to_symmetric_scalar_triplets())                 .unwrap();
//         let mut x = b.clone();
//         let mut x_ref = MatMut::<f64>::from_column_major_slice_mut(x.as_mut_slice(), dim, 1);

//         match csc.sp_lu() {
//             Ok(lu) => {
//                 lu.solve_in_place(ReborrowMut::rb_mut(&mut x_ref));
//                 Ok(x)
//             }
//             Err(e) => Err(LinearSolverError::SparseLuError {
//                 details: match e {
//                     LuError::Generic(faer_err) => match faer_err {
//                         FaerError::IndexOverflow => SparseSolverError::IndexOverflow,
//                         FaerError::OutOfMemory => SparseSolverError::OutOfMemory,
//                         _ => SparseSolverError::Unspecific,
//                     },
//                     LuError::SymbolicSingular { .. } => SparseSolverError::SymbolicSingular,
//                 },
//             }),
//         }
//     }
// }
