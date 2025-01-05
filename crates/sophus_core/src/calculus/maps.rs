/// curve - a function from ℝ to M, where M is a manifold
pub mod curves;
pub use crate::calculus::maps::curves::MatrixValuedCurve;
pub use crate::calculus::maps::curves::ScalarValuedCurve;
pub use crate::calculus::maps::curves::VectorValuedCurve;

/// scalar-valued map - a function from M to ℝ, where M is a manifold
pub mod scalar_valued_maps;
pub use crate::calculus::maps::scalar_valued_maps::ScalarValuedMatrixMap;
pub use crate::calculus::maps::scalar_valued_maps::ScalarValuedVectorMap;

/// vector-valued map - a function from M to ℝⁿ, where M is a manifold
pub mod vector_valued_maps;
pub use crate::calculus::maps::vector_valued_maps::VectorValuedMatrixMap;
pub use crate::calculus::maps::vector_valued_maps::VectorValuedVectorMap;

/// matrix-valued map - a function from M to ℝʳ x ℝᶜ, where M is a manifold
pub mod matrix_valued_maps;
pub use crate::calculus::maps::matrix_valued_maps::MatrixValuedMatrixMap;
pub use crate::calculus::maps::matrix_valued_maps::MatrixValuedVectorMap;
