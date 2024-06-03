#![feature(portable_simd)]
#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]

//! Simple viewer for 2D and 3D visualizations.

/// The actor for the viewer.
pub mod actor;
/// Interactions
pub mod interactions;
/// The offscreen texture for rendering.
pub mod offscreen;
/// The pixel renderer for 2D rendering.
pub mod pixel_renderer;
/// The renderable structs.
pub mod renderables;
/// The scene renderer for 3D rendering.
pub mod scene_renderer;
/// The simple viewer.
pub mod simple_viewer;
/// The view struct.
pub mod views;

use eframe::egui_wgpu::Renderer;
use eframe::epaint::mutex::RwLock;
use std::sync::Arc;

/// The state of the viewer.
#[derive(Clone)]
pub struct ViewerRenderState {
    pub(crate) wgpu_state: Arc<RwLock<Renderer>>,
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) _adapter: Arc<wgpu::Adapter>,
}
