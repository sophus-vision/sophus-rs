use nalgebra::DMatrix;

use crate::matrix::{
    CompressedMatrixEnum,
    block_sparse_triplets::BlockSparseTripletMatrix,
    sparse::{
        FaerTripletMatrix,
        FaerUpperTripletMatrix,
        TripletMatrix,
    },
};

/// Compressible matrix trait.
pub trait IsCompressibleMatrix {
    /// Compressed matrix form.
    type Compressed;

    /// Compress this matrix.
    fn compress(&self) -> Self::Compressed;
}

/// Compressible matrix enum.
pub enum CompressibleMatrixEnum {
    /// Dense matrix - trivial compressible matrix, since it is dense.
    Dense(DMatrix<f64>),
    /// Matrix triplets of lower-triangular matrix,
    SparseLower(TripletMatrix),
    /// Matrix triplets of lower-triangular matrix,
    BlockSparseLower(BlockSparseTripletMatrix),
    /// Matrix triplets to be used with faer crate,
    FaerSparse(FaerTripletMatrix),
    /// Triplets of upper triangular matrix to be used with faer crate,
    FaerSparseUpper(FaerUpperTripletMatrix),
}

impl IsCompressibleMatrix for CompressibleMatrixEnum {
    type Compressed = CompressedMatrixEnum;

    fn compress(&self) -> Self::Compressed {
        match self {
            CompressibleMatrixEnum::Dense(matrix) => CompressedMatrixEnum::Dense(matrix.compress()),
            CompressibleMatrixEnum::SparseLower(lower_triplets_matrix) => {
                CompressedMatrixEnum::SparseLower(lower_triplets_matrix.compress())
            }
            CompressibleMatrixEnum::FaerSparse(faer_triplets_matrix) => {
                CompressedMatrixEnum::FaerSparse(faer_triplets_matrix.compress())
            }
            CompressibleMatrixEnum::FaerSparseUpper(faer_upper_triplets_matrix) => {
                CompressedMatrixEnum::FaerSparseUpper(faer_upper_triplets_matrix.compress())
            }
            CompressibleMatrixEnum::BlockSparseLower(block_sparse_lower_matrix_builder) => {
                CompressedMatrixEnum::BlockSparseLower(block_sparse_lower_matrix_builder.compress())
            }
        }
    }
}
