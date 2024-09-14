use sophus_image::ImageSize;

use crate::RenderContext;

#[derive(Debug)]
pub(crate) struct MainRenderZBuffer {
    pub(crate) ndc_z_texture: wgpu::Texture,
    pub(crate) ndc_z_texture_view: wgpu::TextureView,
}

impl MainRenderZBuffer {
    pub(crate) fn new(render_state: &RenderContext, view_port_size: &ImageSize) -> Self {
        let size = wgpu::Extent3d {
            width: view_port_size.width as u32,
            height: view_port_size.height as u32,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("depth texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = render_state.wgpu_device.create_texture(&desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        MainRenderZBuffer {
            ndc_z_texture: texture,
            ndc_z_texture_view: view,
        }
    }
}
