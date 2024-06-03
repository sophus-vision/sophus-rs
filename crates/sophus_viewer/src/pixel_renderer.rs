/// Line renderer
pub mod line;
/// Pixel point renderer
pub mod pixel_point;

use bytemuck::Pod;
use bytemuck::Zeroable;
use sophus_image::ImageSize;
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;
use wgpu::DepthStencilState;

use crate::interactions::InteractionEnum;
use crate::pixel_renderer::line::PixelLineRenderer;
use crate::pixel_renderer::pixel_point::PixelPointRenderer;
use crate::scene_renderer::depth_renderer::DepthRenderer;
use crate::ViewerRenderState;

/// Renderer for pixel data
pub struct PixelRenderer {
    pub(crate) uniform_bind_group: wgpu::BindGroup,
    pub(crate) _uniform_buffer: wgpu::Buffer,
    pub(crate) line_renderer: PixelLineRenderer,
    pub(crate) point_renderer: PixelPointRenderer,
}

#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct OrthoCam {
    width: f32,
    height: f32,
    dummy0: f32,
    dummy1: f32,
}

/// 2D vertex
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex2 {
    _pos: [f32; 2],
    _color: [f32; 4],
}

/// 2D line vertex
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LineVertex2 {
    pub(crate) _pos: [f32; 2],
    pub(crate) _color: [f32; 4],
    pub(crate) _normal: [f32; 2],
    pub(crate) _line_width: f32,
}

/// 2D point vertex
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PointVertex2 {
    pub(crate) _pos: [f32; 2],
    pub(crate) _point_size: f32,
    pub(crate) _color: [f32; 4],
}

impl PixelRenderer {
    /// Create a new pixel renderer
    pub fn new(
        wgpu_render_state: &ViewerRenderState,
        image_size: &ImageSize,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let device = &wgpu_render_state.device;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pixel group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(16),
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pixel pipeline"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let camera_uniform = OrthoCam {
            width: image_size.width as f32,
            height: image_size.height as f32,
            dummy0: 0.0,
            dummy1: 0.0,
        };

        println!("image_size: {:?}", image_size);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pixel uniform buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pixel uniform bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            _uniform_buffer: uniform_buffer,
            uniform_bind_group,
            line_renderer: PixelLineRenderer::new(
                wgpu_render_state,
                &pipeline_layout,
                depth_stencil.clone(),
            ),
            point_renderer: PixelPointRenderer::new(
                wgpu_render_state,
                &pipeline_layout,
                depth_stencil,
            ),
        }
    }

    pub(crate) fn show_interaction_marker(
        &self,
        state: &ViewerRenderState,
        interaction_state: &InteractionEnum,
    ) {
        match interaction_state {
            InteractionEnum::OrbitalInteraction(orbital_interaction) => {
                if let Some(scene_focus) = orbital_interaction.maybe_scene_focus {
                    *self.point_renderer.show_interaction_marker.lock().unwrap() =
                        if orbital_interaction.maybe_pointer_state.is_some()
                            || orbital_interaction.maybe_scroll_state.is_some()
                        {
                            let mut vertex_data = vec![];

                            for _i in 0..6 {
                                vertex_data.push(PointVertex2 {
                                    _pos: [scene_focus.uv[0] as f32, scene_focus.uv[1] as f32],
                                    _color: [0.5, 0.5, 0.5, 1.0],
                                    _point_size: 5.0,
                                });
                            }
                            state.queue.write_buffer(
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
            InteractionEnum::InplaneInteraction(inplane_interaction) => {
                if let Some(scene_focus) = inplane_interaction.maybe_scene_focus {
                    *self.point_renderer.show_interaction_marker.lock().unwrap() =
                        if inplane_interaction.maybe_pointer_state.is_some()
                            || inplane_interaction.maybe_scroll_state.is_some()
                        {
                            let mut vertex_data = vec![];

                            for _i in 0..6 {
                                vertex_data.push(PointVertex2 {
                                    _pos: [scene_focus.uv[0] as f32, scene_focus.uv[1] as f32],
                                    _color: [0.5, 0.5, 0.5, 1.0],
                                    _point_size: 5.0,
                                });
                            }
                            state.queue.write_buffer(
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
        }
    }

    pub(crate) fn clear_vertex_data(&mut self) {
        self.line_renderer.vertex_data.clear();
        self.point_renderer.vertex_data.clear();
    }

    pub(crate) fn prepare(&self, state: &ViewerRenderState) {
        state.queue.write_buffer(
            &self.point_renderer.vertex_buffer,
            0,
            bytemuck::cast_slice(self.point_renderer.vertex_data.as_slice()),
        );
        state.queue.write_buffer(
            &self.line_renderer.vertex_buffer,
            0,
            bytemuck::cast_slice(self.line_renderer.vertex_data.as_slice()),
        );
    }

    pub(crate) fn paint<'rp>(
        &'rp self,
        command_encoder: &'rp mut wgpu::CommandEncoder,
        texture_view: &'rp wgpu::TextureView,
        depth: &DepthRenderer,
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
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth.depth_texture_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        self.line_renderer
            .paint(&mut render_pass, &self.uniform_bind_group);
        self.point_renderer
            .paint(&mut render_pass, &self.uniform_bind_group);
    }
}
