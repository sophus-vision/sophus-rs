#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]

//! Simple viewer for 2D and 3D visualizations.

/// The render context
pub mod render_context;
/// The renderable structs.
pub mod renderables;
/// The rendering implementation
pub mod renderer;
/// The simple viewer.
pub mod viewer;

pub use crate::render_context::RenderContext;
