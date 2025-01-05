/// curve - a function from ℝ to M, where M is a manifold
pub mod curves;
/// matrix-valued map - a function from M to ℝʳ x ℝᶜ, where M is a manifold
pub mod matrix_valued_maps;
/// scalar-valued map - a function from M to ℝ, where M is a manifold
pub mod scalar_valued_maps;
/// vector-valued map - a function from M to ℝⁿ, where M is a manifold
pub mod vector_valued_maps;

pub use crate::maps::curves::MatrixValuedCurve;
pub use crate::maps::curves::ScalarValuedCurve;
pub use crate::maps::curves::VectorValuedCurve;
pub use crate::maps::matrix_valued_maps::MatrixValuedMatrixMap;
pub use crate::maps::matrix_valued_maps::MatrixValuedVectorMap;
pub use crate::maps::scalar_valued_maps::ScalarValuedMatrixMap;
pub use crate::maps::scalar_valued_maps::ScalarValuedVectorMap;
pub use crate::maps::vector_valued_maps::VectorValuedMatrixMap;
pub use crate::maps::vector_valued_maps::VectorValuedVectorMap;
