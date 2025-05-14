#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![doc = include_str!(concat!("../", std::env!("CARGO_PKG_README")))]
#![cfg_attr(nightly, feature(doc_auto_cfg))]

#[doc = include_str!(concat!("../",  core::env!("CARGO_PKG_README")))]
#[cfg(doctest)]
pub struct ReadmeDoctests;

/// Render camera
pub mod camera;
/// The renderable structs.
pub mod renderables;
/// offscreen texture for rendering
pub mod textures;

/// sophus_renderer prelude.
///
/// It is recommended to import this prelude when working with `sophus_renderer types:
///
/// ```
/// use sophus_renderer::prelude::*;
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
    pub(crate) use alloc::{
        collections::btree_map::BTreeMap,
        format,
        string::{
            String,
            ToString,
        },
        sync::Arc,
        vec,
        vec::Vec,
    };

    pub use sophus_autodiff::prelude::*;
    pub use sophus_image::prelude::*;
    pub use sophus_lie::prelude::*;
    pub use sophus_opt::prelude::*;

    extern crate alloc;
}

mod offscreen_renderer;
mod pipeline_builder;
mod pixel_renderer;
mod render_context;
mod scene_renderer;
mod types;
mod uniform_buffers;

pub use offscreen_renderer::*;
pub use pipeline_builder::*;
pub use pixel_renderer::*;
pub use render_context::*;
pub use scene_renderer::*;
pub use types::*;
pub use uniform_buffers::*;
