use eframe::wgpu;
use sophus_image::ImageSize;

use crate::{
    RenderContext,
    types::SOPHUS_RENDER_MULTISAMPLE_COUNT,
};

#[derive(Debug)]
pub(crate) struct NdcZBuffer {
    pub(crate) _multisample_texture: wgpu::Texture,
    pub(crate) multisample_texture_view: wgpu::TextureView,

    pub(crate) final_texture: wgpu::Texture,
    pub(crate) final_texture_view: wgpu::TextureView,

    /// A copy of the above, taken before anything is drawn over the finished image.
    ///
    /// What is drawn there has to know what the scene left behind, to be occluded by it, and to
    /// leave its own depth behind in turn, to be picked out of later. A pass cannot read the
    /// texture it is writing, so it reads this and writes that.
    pub(crate) read_copy_texture: wgpu::Texture,
    pub(crate) read_copy_texture_view: wgpu::TextureView,
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
            sample_count: SOPHUS_RENDER_MULTISAMPLE_COUNT,
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

        let read_copy_texture = render_state
            .wgpu_device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("depth read copy texture"),
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                ..desc
            });
        let read_copy_texture_view =
            read_copy_texture.create_view(&wgpu::TextureViewDescriptor::default());

        NdcZBuffer {
            _multisample_texture: multisample_texture,
            multisample_texture_view,
            final_texture,
            final_texture_view,
            read_copy_texture,
            read_copy_texture_view,
        }
    }
}
