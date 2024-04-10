/// curve - a function from ℝ to M, where M is a manifold
pub mod curves;
pub use crate::calculus::maps::curves::MatrixValuedCurve;
pub use crate::calculus::maps::curves::ScalarValuedCurve;
pub use crate::calculus::maps::curves::VectorValuedCurve;

/// matrix-valued map - a function from M to ℝʳ x ℝᶜ, where M is a manifold
pub mod matrix_valued_maps;
pub use crate::calculus::maps::matrix_valued_maps::MatrixValuedMapFromMatrix;
pub use crate::calculus::maps::matrix_valued_maps::MatrixValuedMapFromVector;

/// scalar-valued map - a function from M to ℝ, where M is a manifold
pub mod scalar_valued_maps;
pub use crate::calculus::maps::scalar_valued_maps::ScalarValuedMapFromMatrix;
pub use crate::calculus::maps::scalar_valued_maps::ScalarValuedMapFromVector;

/// vector-valued map - a function from M to ℝⁿ, where M is a manifold
pub mod vector_valued_maps;
pub use crate::calculus::maps::vector_valued_maps::VectorValuedMapFromMatrix;
pub use crate::calculus::maps::vector_valued_maps::VectorValuedMapFromVector;
