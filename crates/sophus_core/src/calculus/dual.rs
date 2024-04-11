/// DualScalar matrix.
pub mod dual_matrix;
pub use crate::calculus::dual::dual_matrix::DualBatchMatrix;
pub use crate::calculus::dual::dual_matrix::DualMatrix;

/// DualScalar scalar.
pub mod dual_scalar;
pub use crate::calculus::dual::dual_scalar::DualBatchScalar;
pub use crate::calculus::dual::dual_scalar::DualScalar;

/// DualScalar vector.
pub mod dual_vector;
pub use crate::calculus::dual::dual_vector::DualBatchVector;
pub use crate::calculus::dual::dual_vector::DualVector;
