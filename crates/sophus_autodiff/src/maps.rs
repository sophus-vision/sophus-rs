/// curve - a function from ℝ to M, where M is a manifold
pub mod curves;
/// matrix-valued map - a function from M to ℝʳ x ℝᶜ, where M is a manifold
pub mod matrix_valued_maps;
/// scalar-valued map - a function from M to ℝ, where M is a manifold
pub mod scalar_valued_maps;
/// vector-valued map - a function from M to ℝⁿ, where M is a manifold
pub mod vector_valued_maps;

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
