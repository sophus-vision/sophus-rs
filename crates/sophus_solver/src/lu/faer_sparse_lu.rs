use faer::{
    MatMut,
    prelude::{
        ReborrowMut,
        Solve,
    },
    sparse::{
        FaerError,
        linalg::{
            LuError,
            solvers::Lu,
        },
    },
};

use crate::{
    IsFactor,
    error::{
        FearSparseSolverError,
        LinearSolverError,
    },
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

/// Sparse LU solver
///
/// Sparse LU decomposition - wrapper around faer's sp_lu implementation.
#[derive(Copy, Clone, Debug)]

pub struct FaerSparseLu;

/// j
#[derive(Clone, Debug)]
pub struct FaerSparseLuSystem {
    mat_lu: Lu<usize, f64>,
    n: usize,
}

impl IsLinearSolver for FaerSparseLu {
    type SymmetricMatrixBuilder = FaerSparseMatrixBuilder;
    type Factor = FaerSparseLuSystem;

    const NAME: &'static str = "faer sparse LU";

    fn factorize(&self, mat_a: &FaerSparseMatrix) -> Result<Self::Factor, LinearSolverError> {
        match mat_a.square.sp_lu() {
            Ok(mat_lu) => Ok(FaerSparseLuSystem {
                mat_lu,
                n: mat_a.square.nrows(),
            }),
            Err(e) => Err(LinearSolverError::FaerSparseLuError {
                faer_error: match e {
                    LuError::Generic(faer_err) => match faer_err {
                        FaerError::IndexOverflow => FearSparseSolverError::IndexOverflow,
                        FaerError::OutOfMemory => FearSparseSolverError::OutOfMemory,
                        _ => FearSparseSolverError::Unspecific,
                    },
                    LuError::SymbolicSingular { .. } => FearSparseSolverError::SymbolicSingular,
                },
            }),
        }
    }

    fn zero(&self, partitions: PartitionSet) -> SymmetricMatrixBuilderEnum {
        SymmetricMatrixBuilderEnum::FaerSparse(FaerSparseMatrixBuilder::zero(partitions))
    }

    /// Does not support parallel execution. This function is no-op.
    fn set_parallelize(&mut self, _parallelize: bool) {}
}

impl IsFactor for FaerSparseLuSystem {
    type Matrix = FaerSparseMatrix;

    fn solve_inplace(&self, b: &mut nalgebra::DVector<f64>) -> Result<(), LinearSolverError> {
        let mut x_ref = MatMut::<f64>::from_column_major_slice_mut(b.as_mut_slice(), self.n, 1);

        self.mat_lu.solve_in_place(ReborrowMut::rb_mut(&mut x_ref));
        Ok(())
    }
}
