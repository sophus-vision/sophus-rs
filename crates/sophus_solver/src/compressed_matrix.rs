use enum_as_inner::EnumAsInner;
use nalgebra::DMatrix;

use crate::sparse::{
    CscMatrix,
    faer_sparse_matrix::{
        FaerCompressedMatrix,
        FaerUpperCompressedMatrix,
    },
};

/// Compressed matrix enum.
#[derive(Debug, EnumAsInner)]
pub enum CompressedMatrixEnum {
    /// Dense matrix - trivial case, since te source matrix is dense.
    Dense(DMatrix<f64>),
    /// Compressed sparse lower matrix,
    SparseLower(CscMatrix),
    /// Compressed sparse matrix to be used with faer crate,
    FaerSparse(FaerCompressedMatrix),
    /// Compressed sparse upper-triangular matrix to be used with faer crate,
    FaerSparseUpper(FaerUpperCompressedMatrix),
}
