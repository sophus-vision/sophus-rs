pub mod buffers;
pub mod depth_renderer;
pub mod interaction;
pub mod line;
pub mod mesh;
pub mod point;
pub mod textured_mesh;

use eframe::egui;
use wgpu::DepthStencilState;

use self::buffers::SceneRenderBuffers;
use self::interaction::Interaction;
use self::mesh::MeshRenderer;
use self::point::ScenePointRenderer;
use super::actor::ViewerBuilder;
use super::DepthRenderer;
use super::ViewerRenderState;
use crate::calculus::region::IsRegion;
use crate::image::arc_image::ArcImageF32;
use crate::image::image_view::IsImageView;
use crate::sensor::perspective_camera::KannalaBrandtCamera;
use crate::tensor::view::IsTensorLike;

pub struct SceneRenderer {
    pub buffers: SceneRenderBuffers,
    pub mesh_renderer: MeshRenderer,
    pub textured_mesh_renderer: textured_mesh::TexturedMeshRenderer,
    pub point_renderer: ScenePointRenderer,
    pub line_renderer: line::SceneLineRenderer,
    pub depth_renderer: DepthRenderer,
    pub interaction: Interaction,
}

impl SceneRenderer {
    pub fn new(
        wgpu_render_state: &ViewerRenderState,
        builder: &ViewerBuilder,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let device = &wgpu_render_state.device;

        let depth_renderer = DepthRenderer::new(
            wgpu_render_state,
            &builder.config.camera.intrinsics,
            depth_stencil.clone(),
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                }],
                label: Some("texture_bind_group_layout"),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene pipeline"),
            bind_group_layouts: &[&bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });
        let buffers = SceneRenderBuffers::new(wgpu_render_state, builder);

        Self {
            buffers,
            depth_renderer,
            mesh_renderer: MeshRenderer::new(
                wgpu_render_state,
                &pipeline_layout,
                depth_stencil.clone(),
            ),
            textured_mesh_renderer: textured_mesh::TexturedMeshRenderer::new(
                wgpu_render_state,
                &pipeline_layout,
                depth_stencil.clone(),
            ),
            point_renderer: ScenePointRenderer::new(
                wgpu_render_state,
                &pipeline_layout,
                depth_stencil.clone(),
            ),
            line_renderer: line::SceneLineRenderer::new(
                wgpu_render_state,
                &pipeline_layout,
                depth_stencil,
            ),
            interaction: Interaction {
                maybe_state: None,
                scene_from_camera: builder.config.camera.scene_from_camera,
                clipping_planes: builder.config.camera.clipping_planes,
            },
        }
    }

    pub fn process_event(
        &mut self,
        cam: &KannalaBrandtCamera<f64>,
        response: &egui::Response,
        z_buffer: ArcImageF32,
    ) {
        self.interaction.process_event(cam, response, z_buffer);
    }

    pub fn paint<'rp>(
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
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
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
        self.mesh_renderer.paint(
            &mut render_pass,
            &self.buffers.bind_group,
            &self.buffers.dist_bind_group,
        );
        self.textured_mesh_renderer.paint(
            &mut render_pass,
            &self.buffers.bind_group,
            &self.buffers.dist_bind_group,
        );
        self.point_renderer.paint(
            &mut render_pass,
            &self.buffers.bind_group,
            &self.buffers.dist_bind_group,
        );
        self.line_renderer.paint(
            &mut render_pass,
            &self.buffers.bind_group,
            &self.buffers.dist_bind_group,
        );
    }

    pub fn depth_paint<'rp>(
        &'rp self,
        command_encoder: &'rp mut wgpu::CommandEncoder,
        depth_texture_view: &'rp wgpu::TextureView,
        depth: &DepthRenderer,
    ) {
        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: depth_texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
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
        self.mesh_renderer.depth_paint(
            &mut render_pass,
            &self.buffers.bind_group,
            &self.buffers.dist_bind_group,
        );
        self.textured_mesh_renderer.depth_paint(
            &mut render_pass,
            &self.buffers.bind_group,
            &self.buffers.dist_bind_group,
        );
        self.line_renderer.depth_paint(
            &mut render_pass,
            &self.buffers.bind_group,
            &self.buffers.dist_bind_group,
        );
    }

    pub fn clear_vertex_data(&mut self) {
        self.line_renderer.vertex_data.clear();
        self.point_renderer.vertex_data.clear();
        self.mesh_renderer.vertices.clear();
        self.textured_mesh_renderer.vertices.clear();
    }

    pub fn prepare(&self, state: &ViewerRenderState, intrinsics: &KannalaBrandtCamera<f64>) {
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
        state.queue.write_buffer(
            &self.mesh_renderer.vertex_buffer,
            0,
            bytemuck::cast_slice(self.mesh_renderer.vertices.as_slice()),
        );
        state.queue.write_buffer(
            &self.textured_mesh_renderer.vertex_buffer,
            0,
            bytemuck::cast_slice(self.textured_mesh_renderer.vertices.as_slice()),
        );

        let mut scene_from_camera_uniform: [[f32; 4]; 4] = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                scene_from_camera_uniform[j][i] =
                    self.interaction.scene_from_camera.inverse().matrix()[(i, j)] as f32;
            }
        }
        state.queue.write_buffer(
            &self.buffers.view_uniform_buffer,
            0,
            bytemuck::cast_slice(&[scene_from_camera_uniform]),
        );

        // distortion table
        let mut maybe_dist_lut = self.buffers.distortion_lut.lock().unwrap();
        if maybe_dist_lut.is_none() {
            let distort_lut = &intrinsics.distort_table();
            *maybe_dist_lut = Some(distort_lut.clone());

            state.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.buffers.dist_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(distort_lut.table.tensor.scalar_view().as_slice().unwrap()),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(8 * distort_lut.table.image_size().width as u32),
                    rows_per_image: None,
                },
                self.buffers.dist_texture.size(),
            );
            state.queue.write_buffer(
                &self.buffers.lut_buffer,
                0,
                bytemuck::cast_slice(&[
                    distort_lut.offset().x as f32,
                    distort_lut.offset().y as f32,
                    distort_lut.region.range().x as f32,
                    distort_lut.region.range().y as f32,
                ]),
            );
        }
    }
}
