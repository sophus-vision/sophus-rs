#[cfg(feature = "simd")]
/// Dual batch matrix.
pub mod dual_batch_matrix;
#[cfg(feature = "simd")]
/// Dual batch scalar.
pub mod dual_batch_scalar;
#[cfg(feature = "simd")]
/// Dual batch vector.
pub mod dual_batch_vector;
/// Dual matrix.
pub mod dual_matrix;
/// Dual scalar.
pub mod dual_scalar;
/// Dual vector.
pub mod dual_vector;
/// Dual matrix traits.
pub mod matrix;
/// Dual scalar traits.
pub mod scalar;
/// Dual vector traits.
pub mod vector;

#[cfg(feature = "simd")]
pub use crate::dual::dual_batch_matrix::DualBatchMatrix;
#[cfg(feature = "simd")]
pub use crate::dual::dual_batch_scalar::DualBatchScalar;
#[cfg(feature = "simd")]
pub use crate::dual::dual_batch_vector::DualBatchVector;
pub use crate::dual::{
    dual_matrix::DualMatrix,
    dual_scalar::DualScalar,
    dual_vector::DualVector,
};
