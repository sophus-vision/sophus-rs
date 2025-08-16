use nalgebra::DMatrix;

use crate::{
    block_csc_matrix::BlockCscMatrix,
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
    BlockSparseLower(BlockCscMatrix),
    /// h
    FaerSparse(FaerCompressedMatrix),
    /// f
    FaerSparseUpper(FaerUpperCompressedMatrix),
}
