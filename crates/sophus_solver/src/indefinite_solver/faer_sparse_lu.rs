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

/// Sparse LU solver
///
/// Sparse LU decomposition - wrapper around faer's sp_lu implementation.
#[derive(Copy, Clone, Debug)]

pub struct FaerSparseLu;

impl IsLinearSolver for FaerSparseLu {
    type Matrix = FaerTripletMatrix;

    const NAME: &'static str = "faer sparse LU";

    fn matrix_builder(&self, partitions: &[crate::PartitionSpec]) -> SymmetricMatrixBuilderEnum {
        SymmetricMatrixBuilderEnum::FaerSparse(SparseSymmetricMatrixBuilder::zero(partitions))
    }

    fn solve_in_place(
        &self,
        _parallelize: bool,
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
