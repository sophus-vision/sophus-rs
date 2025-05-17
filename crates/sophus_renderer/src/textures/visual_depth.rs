use eframe::{
    egui,
    wgpu,
};
use sophus_image::ImageSize;

use crate::RenderContext;

#[derive(Debug)]
pub(crate) struct VisualDepthTexture {
    pub visual_texture: wgpu::Texture,
    pub(crate) egui_tex_id: egui::TextureId,
    pub(crate) render_state: RenderContext,
}

impl Drop for VisualDepthTexture {
    fn drop(&mut self) {
        self.render_state
            .egui_wgpu_renderer
            .write()
            .free_texture(&self.egui_tex_id);
    }
}

impl VisualDepthTexture {
    /// Create a new depth renderer.
    pub fn new(render_state: &RenderContext, view_port_size: &ImageSize) -> Self {
        let w = view_port_size.width as u32;
        let h = view_port_size.height as u32;

        let visual_texture = render_state
            .wgpu_device
            .create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
            });

        let visual_texture_view =
            visual_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let egui_tex_id = render_state
            .egui_wgpu_renderer
            .write()
            .register_native_texture(
                render_state.wgpu_device.as_ref(),
                &visual_texture_view,
                wgpu::FilterMode::Linear,
            );
        Self {
            visual_texture,
            egui_tex_id,
            render_state: render_state.clone(),
        }
    }
}
