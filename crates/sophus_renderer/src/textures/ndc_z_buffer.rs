use eframe::wgpu;
use sophus_image::ImageSize;

use crate::{
    RenderContext,
    types::DOG_MULTISAMPLE_COUNT,
};

#[derive(Debug)]
pub(crate) struct NdcZBuffer {
    pub(crate) _multisample_texture: wgpu::Texture,
    pub(crate) multisample_texture_view: wgpu::TextureView,

    pub(crate) final_texture: wgpu::Texture,
    pub(crate) final_texture_view: wgpu::TextureView,
}

impl NdcZBuffer {
    pub(crate) fn new(render_state: &RenderContext, view_port_size: &ImageSize) -> Self {
        let size = wgpu::Extent3d {
            width: view_port_size.width as u32,
            height: view_port_size.height as u32,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("ndc depth multisample texture"),
            size,
            mip_level_count: 1,
            sample_count: DOG_MULTISAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let multisample_texture = render_state.wgpu_device.create_texture(&desc);
        let multisample_texture_view =
            multisample_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let desc = wgpu::TextureDescriptor {
            label: Some("depth final texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        };
        let final_texture = render_state.wgpu_device.create_texture(&desc);
        let final_texture_view = final_texture.create_view(&wgpu::TextureViewDescriptor::default());
        NdcZBuffer {
            _multisample_texture: multisample_texture,
            multisample_texture_view,
            final_texture,
            final_texture_view,
        }
    }
}
