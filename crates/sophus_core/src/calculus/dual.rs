/// Dual matrix.
pub mod dual_matrix;
pub use crate::calculus::dual::dual_matrix::DualMatrix;

#[cfg(feature = "simd")]
/// Dual batch matrix.
pub mod dual_batch_matrix;
#[cfg(feature = "simd")]
pub use crate::calculus::dual::dual_batch_matrix::DualBatchMatrix;

/// Dual scalar.
pub mod dual_scalar;
pub use crate::calculus::dual::dual_scalar::DualScalar;

#[cfg(feature = "simd")]
/// Dual batch scalar.
pub mod dual_batch_scalar;
#[cfg(feature = "simd")]
pub use crate::calculus::dual::dual_batch_scalar::DualBatchScalar;

/// Dual vector.
pub mod dual_vector;
pub use crate::calculus::dual::dual_vector::DualVector;

#[cfg(feature = "simd")]
/// Dual batch vector.
pub mod dual_batch_vector;
#[cfg(feature = "simd")]
pub use crate::calculus::dual::dual_batch_vector::DualBatchVector;
