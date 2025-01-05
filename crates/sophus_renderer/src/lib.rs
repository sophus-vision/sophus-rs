#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![no_std]

//! Renderer

/// Aspect ratio
pub mod aspect_ratio;
/// Render camera
pub mod camera;
/// The rendering implementation
pub mod offscreen_renderer;
/// render uniforms
pub mod pipeline_builder;
/// The pixel renderer for 2D rendering.
pub mod pixel_renderer;
/// The render context
pub mod render_context;
/// The renderable structs.
pub mod renderables;
/// The scene renderer for 3D rendering.
pub mod scene_renderer;
/// offscreen texture for rendering
pub mod textures;
/// Types used in the renderer API
pub mod types;
/// pipeline builder
pub mod uniform_buffers;

pub use crate::render_context::RenderContext;

/// prelude
pub mod prelude {
    pub(crate) use alloc::collections::btree_map::BTreeMap;
    pub(crate) use alloc::format;
    pub(crate) use alloc::string::String;
    pub(crate) use alloc::string::ToString;
    pub(crate) use alloc::sync::Arc;
    pub(crate) use alloc::vec;
    pub(crate) use alloc::vec::Vec;
    pub use sophus_image::prelude::*;

    extern crate alloc;
}
