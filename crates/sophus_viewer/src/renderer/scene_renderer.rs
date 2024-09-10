/// buffers for rendering a scene
pub mod buffers;

/// line renderer
pub mod line;

/// mesh renderer
pub mod mesh;

/// point renderer
pub mod point;

/// textured mesh renderer
pub mod textured_mesh;

use sophus_core::calculus::region::IsRegion;
use sophus_core::tensor::tensor_view::IsTensorLike;
use sophus_image::image_view::IsImageView;
use sophus_lie::Isometry3F64;
use sophus_sensor::distortion_table::distort_table;
use sophus_sensor::dyn_camera::DynCamera;
use wgpu::DepthStencilState;

use crate::renderer::scene_renderer::buffers::Frustum;
use crate::renderer::scene_renderer::buffers::SceneRenderBuffers;
use crate::renderer::scene_renderer::mesh::MeshRenderer;
use crate::renderer::scene_renderer::point::ScenePointRenderer;
use crate::renderer::textures::ZBufferTexture;
use crate::renderer::types::ClippingPlanesF64;
use crate::renderer::types::TranslationAndScaling;
use crate::renderer::types::Zoom2d;
use crate::RenderContext;

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
    clipping_planes: ClippingPlanesF64,
}

impl SceneRenderer {
    /// Create a new scene renderer
    pub fn new(
        wgpu_render_state: &RenderContext,
        intrinsics: DynCamera<f64, 1>,
        clipping_planes: ClippingPlanesF64,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let device = &wgpu_render_state.wgpu_device;

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene group layout"),
                entries: &[
                    // frustum uniform
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
                    // zoom table uniform
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
                    // distortion table uniform
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
                    // view-transform uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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

        let distortion_texture_bind_group_layout =
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
                label: Some("dist_texture_bind_group_layout"),
            });

        let background_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("background_texture_bind_group_layout"),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene pipeline"),
            bind_group_layouts: &[
                &uniform_bind_group_layout,
                &distortion_texture_bind_group_layout,
                &background_texture_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let mesh_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene pipeline"),
            bind_group_layouts: &[
                &uniform_bind_group_layout,
                &distortion_texture_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let buffers = SceneRenderBuffers::new(wgpu_render_state, &intrinsics, clipping_planes);

        Self {
            buffers,
            mesh_renderer: MeshRenderer::new(
                wgpu_render_state,
                &mesh_pipeline_layout,
                depth_stencil.clone(),
            ),
            textured_mesh_renderer: textured_mesh::TexturedMeshRenderer::new(
                wgpu_render_state,
                &pipeline_layout,
                depth_stencil.clone(),
            ),
            point_renderer: ScenePointRenderer::new(
                wgpu_render_state,
                &mesh_pipeline_layout,
                depth_stencil.clone(),
            ),
            line_renderer: line::SceneLineRenderer::new(
                wgpu_render_state,
                &mesh_pipeline_layout,
                depth_stencil,
            ),
            clipping_planes,
        }
    }

    pub(crate) fn paint<'rp>(
        &'rp self,
        state: &RenderContext,
        scene_from_camera: &Isometry3F64,
        command_encoder: &'rp mut wgpu::CommandEncoder,
        texture_view: &'rp wgpu::TextureView,
        depth: &ZBufferTexture,
        backface_culling: bool,
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
            state,
            scene_from_camera,
            &self.buffers,
            &mut render_pass,
            backface_culling,
        );
        self.textured_mesh_renderer.paint(
            state,
            scene_from_camera,
            &self.buffers,
            &mut render_pass,
        );
        self.point_renderer
            .paint(state, scene_from_camera, &self.buffers, &mut render_pass);
        self.line_renderer
            .paint(state, scene_from_camera, &self.buffers, &mut render_pass);
    }

    pub(crate) fn depth_paint<'rp>(
        &'rp self,
        state: &RenderContext,
        scene_from_camera: &Isometry3F64,
        command_encoder: &'rp mut wgpu::CommandEncoder,
        depth_texture_view: &'rp wgpu::TextureView,
        depth: &ZBufferTexture,
        backface_culling: bool,
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
            state,
            scene_from_camera,
            &self.buffers,
            &mut render_pass,
            backface_culling,
        );
        self.textured_mesh_renderer.depth_paint(
            state,
            scene_from_camera,
            &self.buffers,
            &mut render_pass,
        );
        self.line_renderer
            .depth_paint(state, scene_from_camera, &self.buffers, &mut render_pass);
        self.point_renderer
            .depth_paint(state, scene_from_camera, &self.buffers, &mut render_pass);
    }

    pub(crate) fn prepare(
        &self,
        state: &RenderContext,
        zoom_2d: TranslationAndScaling,
        intrinsics: &DynCamera<f64, 1>,
    ) {
        let frustum_uniforms = Frustum {
            camera_image_width: intrinsics.image_size().width as f32,
            camera_image_height: intrinsics.image_size().height as f32,
            near: self.clipping_planes.near as f32,
            far: self.clipping_planes.far as f32,
            fx: intrinsics.pinhole_params()[0] as f32,
            fy: intrinsics.pinhole_params()[1] as f32,
            px: intrinsics.pinhole_params()[2] as f32,
            py: intrinsics.pinhole_params()[3] as f32,
        };

        state.wgpu_queue.write_buffer(
            &self.buffers.frustum_uniform_buffer,
            0,
            bytemuck::cast_slice(&[frustum_uniforms]),
        );

        let zoom_uniform = Zoom2d {
            translation_x: zoom_2d.translation[0] as f32,
            translation_y: zoom_2d.translation[1] as f32,
            scaling_x: zoom_2d.scaling[0] as f32,
            scaling_y: zoom_2d.scaling[1] as f32,
        };

        state.wgpu_queue.write_buffer(
            &self.buffers.zoom_buffer,
            0,
            bytemuck::cast_slice(&[zoom_uniform]),
        );

        // distortion table
        let mut maybe_dist_lut = self.buffers.distortion_lut.lock().unwrap();
        if maybe_dist_lut.is_none() {
            let distort_lut = distort_table(intrinsics);
            *maybe_dist_lut = Some(distort_lut.clone());

            state.wgpu_queue.write_texture(
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
                    rows_per_image: Some(distort_lut.table.image_size().height as u32),
                },
                self.buffers.dist_texture.size(),
            );
            state.wgpu_queue.write_buffer(
                &self.buffers.camara_params_buffer,
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
