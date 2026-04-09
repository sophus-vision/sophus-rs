use nalgebra::DVector;

use crate::matrix::SymmetricMatrixEnum;

/// Linear system examples where `A` is positive semi-definite.
pub mod positive_semidefinite;

/// A linear system `A x = b`.
pub struct LinearSystem {
    /// The left-hand side matrix `A`.
    pub mat_a: SymmetricMatrixEnum,
    /// The right-hand side vector `b`.
    pub b: DVector<f64>,
}
