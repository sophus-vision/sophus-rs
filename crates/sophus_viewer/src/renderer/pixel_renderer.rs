/// Line renderer
pub mod pixel_line;
/// Pixel point renderer
pub mod pixel_point;

use crate::renderer::pipeline_builder::PipelineBuilder;
use crate::renderer::pipeline_builder::PointVertex2;
use crate::renderer::pipeline_builder::TargetTexture;
use crate::renderer::pixel_renderer::pixel_line::PixelLineRenderer;
use crate::renderer::pixel_renderer::pixel_point::PixelPointRenderer;
use crate::renderer::textures::depth_image::ndc_z_to_color;
use crate::renderer::uniform_buffers::VertexShaderUniformBuffers;
use crate::viewer::interactions::InteractionEnum;
use crate::RenderContext;
use std::sync::Arc;

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
        interaction: &InteractionEnum,
    ) {
        if let Some(scene_focus) = interaction.maybe_scene_focus() {
            *self.point_renderer.show_interaction_marker.lock().unwrap() =
                if interaction.is_active() {
                    let mut vertex_data = vec![];

                    let depth_color = ndc_z_to_color(scene_focus.ndc_z).cast::<f32>() / 255.0;

                    for _i in 0..6 {
                        vertex_data.push(PointVertex2 {
                            _pos: [
                                scene_focus.uv_in_virtual_camera[0] as f32,
                                scene_focus.uv_in_virtual_camera[1] as f32,
                            ],
                            _color: [depth_color[0], depth_color[1], depth_color[2], 1.0],
                            _point_size: 5.0,
                        });
                    }
                    state.wgpu_queue.write_buffer(
                        &self.point_renderer.interaction_vertex_buffer,
                        0,
                        bytemuck::cast_slice(&vertex_data),
                    );

                    true
                } else {
                    false
                };
        }
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
