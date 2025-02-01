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

pub use crate::{
    arc_tensor::ArcTensor,
    mut_tensor::MutTensor,
    mut_tensor_view::MutTensorView,
    tensor_view::TensorView,
};

/// sophus_tensor prelude
pub mod prelude {
    pub use sophus_autodiff::prelude::*;

    pub use crate::{
        element::IsStaticTensor,
        mut_tensor::{
            InnerScalarToVec,
            InnerVecToMat,
        },
        mut_tensor_view::IsMutTensorLike,
        tensor_view::{
            IsTensorLike,
            IsTensorView,
        },
    };
}
