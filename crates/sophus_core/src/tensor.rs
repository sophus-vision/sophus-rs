#![deny(missing_docs)]
//! # Calculus module

/// Arc tensor
pub mod arc_tensor;
pub use crate::tensor::arc_tensor::ArcTensor;

/// Tensor element
pub mod element;

/// Mutable tensor
pub mod mut_tensor;
pub use crate::tensor::mut_tensor::MutTensor;

/// Mutable tensor view
pub mod mut_tensor_view;
pub use crate::tensor::mut_tensor_view::MutTensorView;

/// Tensor view
pub mod tensor_view;
pub use crate::tensor::tensor_view::TensorView;
