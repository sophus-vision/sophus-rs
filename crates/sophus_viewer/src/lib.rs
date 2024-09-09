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

use eframe::egui_wgpu::RenderState;
use eframe::egui_wgpu::Renderer;
use eframe::epaint::mutex::RwLock;
use std::sync::Arc;

/// The state of the viewer.
#[derive(Clone)]
pub struct ViewerRenderState {
    /// state
    pub wgpu_state: Arc<RwLock<Renderer>>,
    /// device
    pub device: Arc<wgpu::Device>,
    /// queue
    pub queue: Arc<wgpu::Queue>,
    /// adapter
    pub adapter: Arc<wgpu::Adapter>,
}

impl ViewerRenderState {
    /// Create a new viewer render state.
    pub fn from_render_state(render_state: &RenderState) -> Self {
        ViewerRenderState {
            adapter: render_state.adapter.clone(),
            device: render_state.device.clone(),
            queue: render_state.queue.clone(),
            wgpu_state: render_state.renderer.clone(),
        }
    }

    /// Create a new viewer render state.
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&Default::default(), None)
            .await
            .unwrap();

        let renderer = Renderer::new(&device, wgpu::TextureFormat::Rgba8Unorm, None, 1);

        ViewerRenderState {
            wgpu_state: Arc::new(RwLock::new(renderer)),
            device: device.into(),
            queue: queue.into(),
            adapter: adapter.into(),
        }
    }

    /// Create a new viewer render state.
    pub fn from_egui_cc(cc: &eframe::CreationContext) -> Self {
        ViewerRenderState::from_render_state(cc.wgpu_render_state.as_ref().unwrap())
    }
}
