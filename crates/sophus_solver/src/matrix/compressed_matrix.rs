use enum_as_inner::EnumAsInner;
use nalgebra::DMatrix;

use crate::matrix::{
    block_col_compressed_matrix::BlockColCompressedMatrix,
    sparse::{
        ColumnCompressedMatrix,
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
    SparseLower(ColumnCompressedMatrix),
    /// Compressed block-sparse lower matrix,
    BlockSparseLower(BlockColCompressedMatrix),
    /// Compressed sparse matrix to be used with faer crate,
    FaerSparse(FaerCompressedMatrix),
    /// Compressed sparse upper-triangular matrix to be used with faer crate,
    FaerSparseUpper(FaerUpperCompressedMatrix),
}
