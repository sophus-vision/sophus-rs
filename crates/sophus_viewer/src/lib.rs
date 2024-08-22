#![cfg_attr(feature = "simd", feature(portable_simd))]
#![deny(missing_docs)]
#![allow(clippy::needless_range_loop)]

//! Simple viewer for 2D and 3D visualizations.

/// The offscreen texture for rendering.
pub mod offscreen_renderer;
/// The renderable structs.
pub mod renderables;
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

impl ViewerRenderState {
    /// Create a new viewer render state.
    pub fn new(cc: &eframe::CreationContext) -> Self {
        ViewerRenderState {
            _adapter: cc.wgpu_render_state.as_ref().unwrap().adapter.clone(),
            device: cc.wgpu_render_state.as_ref().unwrap().device.clone(),
            queue: cc.wgpu_render_state.as_ref().unwrap().queue.clone(),
            wgpu_state: cc.wgpu_render_state.as_ref().unwrap().renderer.clone(),
        }
    }
}
