#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]
//! # Tensor module

/// Arc tensor
pub mod arc_tensor;
/// Tensor element
pub mod element;
/// Mutable tensor
pub mod mut_tensor;
/// Mutable tensor view
pub mod mut_tensor_view;
/// Tensor view
pub mod tensor_view;

pub use crate::arc_tensor::ArcTensor;
pub use crate::mut_tensor::MutTensor;
pub use crate::mut_tensor_view::MutTensorView;
pub use crate::tensor_view::TensorView;

/// sophus_tensor prelude
pub mod prelude {
    pub use crate::element::IsStaticTensor;
    pub use crate::mut_tensor::InnerScalarToVec;
    pub use crate::mut_tensor::InnerVecToMat;
    pub use crate::mut_tensor_view::IsMutTensorLike;
    pub use crate::tensor_view::IsTensorLike;
    pub use crate::tensor_view::IsTensorView;
    pub use sophus_autodiff::prelude::*;
}
