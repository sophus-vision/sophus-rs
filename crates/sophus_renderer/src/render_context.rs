use std::fmt::Debug;

use eframe::{
    egui_wgpu,
    epaint::mutex::RwLock,
    wgpu,
};

use crate::{
    prelude::*,
    types::SOPHUS_RENDER_MULTISAMPLE_COUNT,
};

/// The render context
#[derive(Clone)]
pub struct RenderContext {
    /// state
    pub egui_wgpu_renderer: Arc<RwLock<egui_wgpu::Renderer>>,
    /// device
    pub wgpu_device: Arc<wgpu::Device>,
    /// queue
    pub wgpu_queue: Arc<wgpu::Queue>,
    /// adapter
    pub wgpu_adapter: Arc<wgpu::Adapter>,
}

impl Debug for RenderContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenderContext")
            .field("wgpu_device", &self.wgpu_device)
            .field("wgpu_queue", &self.wgpu_queue)
            .field("wgpu_adapter", &self.wgpu_adapter)
            .finish()
    }
}

impl RenderContext {
    /// Creates a render context from a egui_wgpu render state
    pub fn from_render_state(render_state: &egui_wgpu::RenderState) -> Self {
        RenderContext {
            wgpu_adapter: Arc::new(render_state.adapter.clone()),
            wgpu_device: Arc::new(render_state.device.clone()),
            wgpu_queue: Arc::new(render_state.queue.clone()),
            egui_wgpu_renderer: render_state.renderer.clone(),
        }
    }

    /// Creates a new render context, include wgpu device, adapter and queue
    /// as well as egui_wgpu renderer.
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
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
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

        const DITHERING: bool = false;

        let renderer = egui_wgpu::Renderer::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            None,
            SOPHUS_RENDER_MULTISAMPLE_COUNT,
            DITHERING,
        );

        RenderContext {
            egui_wgpu_renderer: Arc::new(RwLock::new(renderer)),
            wgpu_device: device.into(),
            wgpu_queue: queue.into(),
            wgpu_adapter: adapter.into(),
        }
    }

    /// Creates a render context from an eframe creation context.
    pub fn from_egui_cc(cc: &eframe::CreationContext) -> Self {
        RenderContext::from_render_state(cc.wgpu_render_state.as_ref().unwrap())
    }
}
