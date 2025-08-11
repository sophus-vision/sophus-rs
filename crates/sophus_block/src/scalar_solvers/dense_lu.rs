use crate::{
    IsDenseLinearSystem,
    IsSparseSymmetricLinearSystem,
    LinearSolverError,
    SymmetricBlockSparseMatrix,
};

/// Dense LU solver
///
/// LU decomposition with full pivoting - wrapper around nalgebra's full_piv_lu.
pub struct DenseLu;

impl IsSparseSymmetricLinearSystem for DenseLu {
    fn solve(
        &self,
        triplets: &SymmetricBlockSparseMatrix,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, LinearSolverError> {
        self.solve_dense(triplets.to_symmetric_dense(), b)
    }
}

impl IsDenseLinearSystem for DenseLu {
    fn solve_dense(
        &self,
        mat_a: nalgebra::DMatrix<f64>,
        b: &nalgebra::DVector<f64>,
    ) -> Result<nalgebra::DVector<f64>, LinearSolverError> {
        match mat_a.full_piv_lu().solve(b) {
            Some(x) => Ok(x),
            None => Err(LinearSolverError::DenseLuError),
        }
    }
}
