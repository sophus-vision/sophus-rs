mod curves;
mod matrix_valued_maps;
mod scalar_valued_maps;
mod vector_valued_maps;

pub use crate::maps::{
    curves::{
        MatrixValuedCurve,
        ScalarValuedCurve,
        VectorValuedCurve,
    },
    matrix_valued_maps::{
        MatrixValuedMatrixMap,
        MatrixValuedVectorMap,
    },
    scalar_valued_maps::{
        ScalarValuedMatrixMap,
        ScalarValuedVectorMap,
    },
    vector_valued_maps::{
        VectorValuedMatrixMap,
        VectorValuedVectorMap,
    },
};
