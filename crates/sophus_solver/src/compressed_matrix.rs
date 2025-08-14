use nalgebra::DMatrix;

use crate::{
    BlockSparseLowerCompressedMatrix,
    sparse::{
        CscMatrix,
        LowerCscMatrix,
        faer_sparse_matrix::{
            FaerCompressedMatrix,
            FaerUpperCompressedMatrix,
        },
    },
};

/// c
pub enum CompressedMatrixEnum {
    ///d
    Dense(DMatrix<f64>),
    /// s
    SparseLower(LowerCscMatrix),
    /// s
    Sparse(CscMatrix),
    /// s
    BlockSparseLower(BlockSparseLowerCompressedMatrix),
    /// h
    FaerSparse(FaerCompressedMatrix),
    /// f
    FaerSparseUpper(FaerUpperCompressedMatrix),
}
