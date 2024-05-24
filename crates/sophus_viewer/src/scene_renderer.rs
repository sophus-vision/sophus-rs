/// buffers for rendering a scene
pub mod buffers;

/// depth renderer
pub mod depth_renderer;

/// interaction state
pub mod interaction;

/// line renderer
pub mod line;

/// mesh renderer
pub mod mesh;

/// point renderer
pub mod point;

/// textured mesh renderer
pub mod textured_mesh;
use self::buffers::SceneRenderBuffers;
use self::interaction::Interaction;
use self::mesh::MeshRenderer;
use self::point::ScenePointRenderer;
use crate::actor::ViewerBuilder;
use crate::DepthRenderer;
use crate::ViewerRenderState;
use eframe::egui;
use sophus_core::calculus::region::IsRegion;
use sophus_core::tensor::tensor_view::IsTensorLike;
use sophus_image::arc_image::ArcImageF32;
use sophus_image::image_view::IsImageView;
use sophus_sensor::distortion_table::distort_table;
use sophus_sensor::dyn_camera::DynCamera;
use wgpu::DepthStencilState;

/// Scene renderer
pub struct SceneRenderer {
    /// Buffers of the scene
    pub buffers: SceneRenderBuffers,
    /// Mesh renderer
    pub mesh_renderer: MeshRenderer,
    /// Textured mesh renderer
    pub textured_mesh_renderer: textured_mesh::TexturedMeshRenderer,
    /// Point renderer
    pub point_renderer: ScenePointRenderer,
    /// Line renderer
    pub line_renderer: line::SceneLineRenderer,
    /// Depth renderer
    pub depth_renderer: DepthRenderer,
    /// Interaction state
    pub interaction: Interaction,
}

impl SceneRenderer {
    /// Create a new scene renderer
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
                maybe_pointer_state: None,
                maybe_scroll_state: None,
                maybe_scene_focus: None,
                scene_from_camera: builder.config.camera.scene_from_camera,
                clipping_planes: builder.config.camera.clipping_planes,
            },
        }
    }

    pub(crate) fn process_event(
        &mut self,
        cam: &DynCamera<f64, 1>,
        response: &egui::Response,
        z_buffer: ArcImageF32,
    ) {
        self.interaction.process_event(cam, response, &z_buffer);
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

    pub(crate) fn depth_paint<'rp>(
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

    /// Clear the vertex data
    pub fn clear_vertex_data(&mut self) {
        self.line_renderer.vertex_data.clear();
        self.point_renderer.vertex_data.clear();
        self.mesh_renderer.vertices.clear();
        self.textured_mesh_renderer.vertices.clear();
    }

    pub(crate) fn prepare(&self, state: &ViewerRenderState, intrinsics: &DynCamera<f64, 1>) {
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
            let distort_lut = distort_table(intrinsics);
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
