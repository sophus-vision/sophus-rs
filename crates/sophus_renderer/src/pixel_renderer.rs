mod pixel_ellipse;
mod pixel_line;
mod pixel_point;
mod scene_overlay;

use eframe::wgpu;
pub use pixel_ellipse::*;
pub use pixel_line::*;
pub use pixel_point::*;
use sophus_autodiff::linalg::{
    MatF64,
    SVec,
};

use crate::{
    RenderContext,
    pipeline_builder::{
        PipelineBuilder,
        TargetTexture,
    },
    pixel_renderer::{
        pixel_ellipse::ellipse_vertex,
        pixel_line::PixelLineRenderer,
        pixel_point::PixelPointRenderer,
        scene_overlay::SceneOverlayRenderer,
    },
    prelude::*,
    renderables::{
        Color,
        Ellipse2,
    },
    textures::DepthTextures,
    types::ScenePivotMarker,
    uniform_buffers::{
        OVERLAY_POSE_SLOT,
        VertexShaderUniformBuffers,
    },
};

/// How big the dot at the pivot is, in image pixels, and how far the grey behind it stands out.
const PIVOT_RADIUS_PIXELS: f64 = 3.5;
const PIVOT_HALO_PIXELS: f64 = 1.5;

/// Renderer for pixel data
pub struct PixelRenderer {
    pub(crate) line_renderer: PixelLineRenderer,
    pub(crate) point_renderer: PixelPointRenderer,
    pub(crate) ellipse_renderer: PixelEllipseRenderer,
    /// the scene's own lines and points, which are drawn here rather than into the intermediate
    pub(crate) scene_overlay: SceneOverlayRenderer,
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
            ellipse_renderer: PixelEllipseRenderer::new(render_context, &pixel_pipeline_builder),
            scene_overlay: SceneOverlayRenderer::new(render_context, &pixel_pipeline_builder),
            pixel_pipeline_builder,
        }
    }

    /// The pivot an interaction turns about, drawn as a dot in the colour the marker arrives
    /// with - how far away the point is, mapped the way the depth view maps it - with a ring of
    /// grey behind it so that it reads against whatever it is held over.
    pub(crate) fn show_interaction_marker(
        &self,
        context: &RenderContext,
        marker: &Option<ScenePivotMarker>,
    ) {
        let Some(marker) = marker else {
            *self.ellipse_renderer.show_interaction_marker.lock() = false;
            return;
        };

        let dot = |radius: f64, color: Color| Ellipse2 {
            center: SVec::<f32, 2>::new(marker.u, marker.v),
            shape: MatF64::<2, 2>::identity() * radius,
            line_width: 0.0,
            color,
        };
        let vertex_data = [
            dot(
                PIVOT_RADIUS_PIXELS + PIVOT_HALO_PIXELS,
                Color {
                    r: 0.16,
                    g: 0.16,
                    b: 0.18,
                    a: 0.9,
                },
            ),
            dot(PIVOT_RADIUS_PIXELS, marker.color),
        ]
        .iter()
        .filter_map(ellipse_vertex)
        .collect::<Vec<_>>();

        context.wgpu_queue.write_buffer(
            &self.ellipse_renderer.interaction_vertex_buffer,
            0,
            bytemuck::cast_slice(&vertex_data),
        );
        *self.ellipse_renderer.show_interaction_marker.lock() = true;
    }

    pub(crate) fn paint<'rp>(
        &'rp self,
        render_context: &RenderContext,
        command_encoder: &'rp mut wgpu::CommandEncoder,
        texture_view: &'rp wgpu::TextureView,
        depth: &DepthTextures,
    ) {
        // The scene's own lines and points, in a pass of their own: they write the scene's
        // inverse distance as well as the colour, which the 2d renderables over them do not.
        //
        // A pass cannot read the texture it writes, so the depth is copied first and the copy is
        // what they are occluded against. Both only when there is something to draw.
        let scene_depth = &depth.main_render_ndc_z_texture;
        if !self.scene_overlay.is_empty() {
            command_encoder.copy_texture_to_texture(
                scene_depth.final_texture.as_image_copy(),
                scene_depth.read_copy_texture.as_image_copy(),
                scene_depth.final_texture.size(),
            );
            let depth_bind_group =
                render_context
                    .wgpu_device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("scene overlay depth bind group"),
                        layout: &self.scene_overlay.depth_bind_group_layout,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(
                                &scene_depth.read_copy_texture_view,
                            ),
                        }],
                    });

            let mut overlay_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene overlay pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: texture_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &scene_depth.final_texture_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // They are held in world coordinates, so they are drawn from the slot which holds the
            // camera's own pose rather than any entity's.
            let uniforms = &self.pixel_pipeline_builder.uniforms;
            overlay_pass.set_bind_group(
                0,
                &uniforms.render_bind_group,
                &[OVERLAY_POSE_SLOT * uniforms.camera_from_entity_pose_buffer.slot_stride as u32],
            );
            overlay_pass.set_bind_group(1, &depth_bind_group, &[]);
            self.scene_overlay.paint(&mut overlay_pass);
        }

        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: texture_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        // 2d renderables have no entity pose; bind the first (identity) pose slot
        render_pass.set_bind_group(
            0,
            &self.pixel_pipeline_builder.uniforms.render_bind_group,
            &[0],
        );

        self.line_renderer.paint(&mut render_pass);
        self.ellipse_renderer.paint(&mut render_pass);
        self.point_renderer.paint(&mut render_pass);
    }
}
