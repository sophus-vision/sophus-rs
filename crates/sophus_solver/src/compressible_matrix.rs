use nalgebra::DMatrix;

use crate::{
    CompressedMatrixEnum,
    sparse::{
        TripletMatrix,
        faer_sparse_matrix::{
            FaerTripletMatrix,
            FaerUpperTripletMatrix,
        },
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
    /// Matrix triplets to be used with faer crate,
    FaerSparse(FaerTripletMatrix),
    /// Triplets of upper triangular matrix to be used with faer crate,
    FaerSparseUpper(FaerUpperTripletMatrix),
}

impl IsCompressibleMatrix for CompressibleMatrixEnum {
    type Compressed = CompressedMatrixEnum;

    fn compress(&self) -> Self::Compressed {
        let c = match self {
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
        };
        tracing::trace!("compress");
        c
    }
}
