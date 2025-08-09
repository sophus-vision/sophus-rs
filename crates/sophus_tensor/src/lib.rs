#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![no_std]
#![allow(clippy::needless_range_loop)]
#![doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]
#![cfg_attr(any(docsrs, nightly), feature(doc_auto_cfg))]

#[doc = include_str!(concat!("../",  core::env!("CARGO_PKG_README")))]
#[cfg(doctest)]
pub struct ReadmeDoctests;

#[cfg(feature = "std")]
extern crate std;

mod arc_tensor;
mod element;
mod mut_tensor;
mod mut_tensor_view;
mod tensor_view;

/// sophus_tensor prelude.
///
/// It is recommended to import this prelude when working with `sophus_tensor types:
///
/// ```
/// use sophus_tensor::prelude::*;
/// ```
///
/// or
///
/// ```ignore
/// use sophus::prelude::*;
/// ```
///
/// to import all preludes when using the `sophus` umbrella crate.
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

pub use arc_tensor::*;
pub use element::*;
pub use mut_tensor::*;
pub use mut_tensor_view::*;
pub use tensor_view::*;
