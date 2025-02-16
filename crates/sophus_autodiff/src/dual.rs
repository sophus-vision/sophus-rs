#[cfg(feature = "simd")]
mod dual_batch_matrix;
#[cfg(feature = "simd")]
mod dual_batch_scalar;
#[cfg(feature = "simd")]
mod dual_batch_vector;
mod dual_matrix;
mod dual_scalar;
mod dual_vector;
mod matrix;
mod scalar;
mod vector;

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
    matrix::*,
    scalar::*,
    vector::*,
};
