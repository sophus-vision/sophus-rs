#![cfg_attr(feature = "simd", feature(portable_simd))]
#![allow(clippy::needless_range_loop)]
#![doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]
#![cfg_attr(nightly, feature(doc_auto_cfg))]

#[doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]
#[cfg(doctest)]
pub struct ReadmeDoctests;

#[doc(inline)]
pub use sophus_autodiff as autodiff;
#[doc(inline)]
pub use sophus_geo as geo;
#[doc(inline)]
pub use sophus_image as image;
#[doc(inline)]
pub use sophus_lie as lie;
#[doc(inline)]
pub use sophus_opt as opt;
#[doc(inline)]
pub use sophus_renderer as renderer;
#[doc(inline)]
pub use sophus_sensor as sensor;
#[doc(inline)]
pub use sophus_sim as sim;
#[doc(inline)]
pub use sophus_spline as spline;
#[doc(inline)]
pub use sophus_tensor as tensor;
#[doc(inline)]
pub use sophus_timeseries as timeseries;
#[doc(inline)]
pub use sophus_viewer as viewer;

/// Examples for the `sophus` umbrella crate. Note that this is not a comprehensive list of
/// examples, and most usage examples are found in the individual sub-crates. In particular, the
/// unit tests in each sub-crate are a good source of examples.
pub mod examples;
pub use eframe;
pub use nalgebra;
pub use ndarray;
pub use thingbuf;

/// sophus prelude.
///
/// It is recommended to import this prelude when working with `sophus` types:
///
/// ```
/// use sophus_autodiff::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{
        autodiff::prelude::*,
        image::prelude::*,
        lie::prelude::*,
        opt::prelude::*,
    };
}
