mod pixel_line;
mod pixel_point;

use eframe::wgpu;
pub use pixel_line::*;
pub use pixel_point::*;

use crate::{
    RenderContext,
    pipeline_builder::{
        PipelineBuilder,
        PointVertex2,
        TargetTexture,
    },
    pixel_renderer::{
        pixel_line::PixelLineRenderer,
        pixel_point::PixelPointRenderer,
    },
    prelude::*,
    types::SceneFocusMarker,
    uniform_buffers::VertexShaderUniformBuffers,
};

/// Renderer for pixel data
pub struct PixelRenderer {
    pub(crate) line_renderer: PixelLineRenderer,
    pub(crate) point_renderer: PixelPointRenderer,
    pub(crate) pixel_pipeline_builder: PipelineBuilder,
}

impl PixelRenderer {
    /// Create a new pixel renderer
    pub fn new(render_context: &RenderContext, uniforms: Arc<VertexShaderUniformBuffers>) -> Self {
        let pixel_pipeline_builder = PipelineBuilder::new_pixel(
            render_context,
            Arc::new(TargetTexture {
                rgba_output_format: wgpu::TextureFormat::Rgba8Unorm,
            }),
            uniforms.clone(),
        );

        Self {
            line_renderer: PixelLineRenderer::new(render_context, &pixel_pipeline_builder),
            point_renderer: PixelPointRenderer::new(render_context, &pixel_pipeline_builder),
            pixel_pipeline_builder,
        }
    }

    pub(crate) fn show_interaction_marker(
        &self,
        state: &RenderContext,
        marker: &Option<SceneFocusMarker>,
    ) {
        *self.point_renderer.show_interaction_marker.lock() = match marker {
            Some(marker) => {
                let mut vertex_data = vec![];

                let depth_color = marker.color;

                for _i in 0..6 {
                    vertex_data.push(PointVertex2 {
                        _pos: [marker.u, marker.v],
                        _color: [depth_color.r, depth_color.g, depth_color.b, depth_color.a],
                        _point_size: 5.0,
                    });
                }
                state.wgpu_queue.write_buffer(
                    &self.point_renderer.interaction_vertex_buffer,
                    0,
                    bytemuck::cast_slice(&vertex_data),
                );

                true
            }
            None => false,
        };
    }

    pub(crate) fn paint<'rp>(
        &'rp self,
        command_encoder: &'rp mut wgpu::CommandEncoder,
        texture_view: &'rp wgpu::TextureView,
    ) {
        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        render_pass.set_bind_group(
            0,
            &self.pixel_pipeline_builder.uniforms.render_bind_group,
            &[],
        );

        self.line_renderer.paint(&mut render_pass);
        self.point_renderer.paint(&mut render_pass);
    }
}
