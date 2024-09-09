/// Line renderer
pub mod pixel_line;
/// Pixel point renderer
pub mod pixel_point;

use bytemuck::Pod;
use bytemuck::Zeroable;
use sophus_image::ImageSize;
use sophus_sensor::DynCamera;
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;
use wgpu::DepthStencilState;

use crate::offscreen_renderer::pixel_renderer::pixel_line::PixelLineRenderer;
use crate::offscreen_renderer::pixel_renderer::pixel_point::PixelPointRenderer;
use crate::offscreen_renderer::textures::ZBufferTexture;
use crate::offscreen_renderer::TranslationAndScaling;
use crate::offscreen_renderer::Zoom2d;
use crate::views::interactions::InteractionEnum;
use crate::ViewerRenderState;

/// Renderer for pixel data
pub struct PixelRenderer {
    pub(crate) uniform_bind_group: wgpu::BindGroup,
    pub(crate) ortho_cam_buffer: wgpu::Buffer,
    pub(crate) zoom_buffer: wgpu::Buffer,
    pub(crate) line_renderer: PixelLineRenderer,
    pub(crate) point_renderer: PixelPointRenderer,
}

#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct OrthoCam {
    virtual_camera_width: f32,
    virtual_camera_height: f32,
    viewport_scale: f32, // virtual_camera_width / viewport_width
    dummy: f32,
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
            label: Some("pixel render layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(16),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(16),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pixel pipeline"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let camera_uniform = OrthoCam {
            virtual_camera_width: image_size.width as f32,
            virtual_camera_height: image_size.height as f32,
            viewport_scale: 1.0,
            dummy: 0.0,
        };

        let ortho_cam_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pixel uniform buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let t_and_s_uniform = Zoom2d::default();

        let t_and_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("zoom buffer"),
            contents: bytemuck::cast_slice(&[t_and_s_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pixel uniform bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ortho_cam_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: t_and_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            ortho_cam_buffer,
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
            zoom_buffer: t_and_buffer,
        }
    }

    pub(crate) fn show_interaction_marker(
        &self,
        state: &ViewerRenderState,
        interaction: &InteractionEnum,
    ) {
        if let Some(scene_focus) = interaction.maybe_scene_focus() {
            *self.point_renderer.show_interaction_marker.lock().unwrap() =
                if interaction.is_active() {
                    let mut vertex_data = vec![];

                    for _i in 0..6 {
                        vertex_data.push(PointVertex2 {
                            _pos: [
                                scene_focus.uv_in_virtual_camera[0] as f32,
                                scene_focus.uv_in_virtual_camera[1] as f32,
                            ],
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

    pub(crate) fn prepare(
        &self,
        state: &ViewerRenderState,
        viewport_size: &ImageSize,
        intrinsics: &DynCamera<f64, 1>,
        zoom_2d: TranslationAndScaling,
    ) {
        let ortho_cam_uniforms = OrthoCam {
            virtual_camera_width: intrinsics.image_size().width as f32,
            virtual_camera_height: intrinsics.image_size().height as f32,
            viewport_scale: intrinsics.image_size().width as f32 / viewport_size.width as f32,
            dummy: 0.0,
        };

        state.queue.write_buffer(
            &self.ortho_cam_buffer,
            0,
            bytemuck::cast_slice(&[ortho_cam_uniforms]),
        );

        let zoom_uniform = Zoom2d {
            translation_x: zoom_2d.translation[0] as f32,
            translation_y: zoom_2d.translation[1] as f32,
            scaling_x: zoom_2d.scaling[0] as f32,
            scaling_y: zoom_2d.scaling[1] as f32,
        };

        state
            .queue
            .write_buffer(&self.zoom_buffer, 0, bytemuck::cast_slice(&[zoom_uniform]));
    }

    pub(crate) fn paint<'rp>(
        &'rp self,
        command_encoder: &'rp mut wgpu::CommandEncoder,
        texture_view: &'rp wgpu::TextureView,
        depth: &ZBufferTexture,
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
