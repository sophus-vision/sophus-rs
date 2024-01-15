use wgpu::DepthStencilState;

use super::ViewerRenderState;
use crate::sensor::perspective_camera::KannalaBrandtCamera;

pub struct DepthRenderer {
    pub depth_stencil: Option<wgpu::DepthStencilState>,
    pub depth_texture: wgpu::Texture,
    pub depth_texture_view: wgpu::TextureView,
    pub depth_texture_sampler: wgpu::Sampler,
}

impl DepthRenderer {
    pub fn new(
        state: &ViewerRenderState,
        cam: &KannalaBrandtCamera<f64>,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

        let size = wgpu::Extent3d {
            width: cam.image_size().width as u32,
            height: cam.image_size().height as u32,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("depth texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT // 3.
                        | wgpu::TextureUsages::TEXTURE_BINDING|wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        let texture = state.device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = state.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            ..Default::default()
        });
        Self {
            depth_texture: texture,
            depth_texture_view: view,
            depth_texture_sampler: sampler,
            depth_stencil,
        }
    }
}
