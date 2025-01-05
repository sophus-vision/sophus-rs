use crate::pipeline_builder::PipelineBuilder;
use crate::pipeline_builder::TexturedMeshVertex3;
use crate::prelude::*;
use crate::renderables::scene_renderable::TexturedTriangleMesh3;
use crate::uniform_buffers::VertexShaderUniformBuffers;
use crate::RenderContext;
use sophus_image::arc_image::ArcImage4U8;
use sophus_image::image_view::IsImageView;
use sophus_lie::Isometry3F64;
use wgpu::util::DeviceExt;

/// mesh entity
pub struct TexturedMeshEntity {
    pub(crate) _vertex_data: Vec<TexturedMeshVertex3>,
    pub(crate) _vertex_buffer: wgpu::Buffer,
    pub(crate) _texture: wgpu::Texture,
    pub(crate) _texture_bind_group: wgpu::BindGroup,
    pub(crate) _scene_from_entity: Isometry3F64,
}

impl TexturedMeshEntity {
    /// new
    pub fn new(
        render_context: &RenderContext,
        mesh: &TexturedTriangleMesh3,
        image: ArcImage4U8,
    ) -> Self {
        let vertex_data: Vec<TexturedMeshVertex3> = mesh
            .triangles
            .iter()
            .flat_map(|trig| {
                vec![
                    TexturedMeshVertex3 {
                        _pos: [trig.p0[0], trig.p0[1], trig.p0[2]],
                        _tex: [trig.tex0[0], trig.tex0[1]],
                    },
                    TexturedMeshVertex3 {
                        _pos: [trig.p1[0], trig.p1[1], trig.p1[2]],
                        _tex: [trig.tex1[0], trig.tex1[1]],
                    },
                    TexturedMeshVertex3 {
                        _pos: [trig.p2[0], trig.p2[1], trig.p2[2]],
                        _tex: [trig.tex2[0], trig.tex2[1]],
                    },
                ]
            })
            .collect();

        let vertex_buffer =
            render_context
                .wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("3D mesh vertex buffer: {}", mesh.name)),
                    contents: bytemuck::cast_slice(&vertex_data),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        let device = &render_context.wgpu_device;

        let background_bind_group_layout =
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

        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let texture_size = wgpu::Extent3d {
            width: image.image_size().width as u32,
            height: image.image_size().height as u32,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("dist_texture"),
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &background_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
            ],
            label: Some("background_bind_group"),
        });

        render_context.wgpu_queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(image.tensor.scalar_view().as_slice().unwrap()),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * image.image_size().width as u32),
                rows_per_image: Some(image.image_size().height as u32),
            },
            texture.size(),
        );

        Self {
            _vertex_data: vertex_data,
            _vertex_buffer: vertex_buffer,
            _texture: texture,
            _texture_bind_group: texture_bind_group,
            _scene_from_entity: mesh.world_from_entity,
        }
    }
}

/// Scene textured mesh renderer
pub struct TexturedMeshRenderer {
    /// pipeline
    pub pipeline: wgpu::RenderPipeline,
    /// table
    pub mesh_table: BTreeMap<String, TexturedMeshEntity>,
}

impl TexturedMeshRenderer {
    /// new
    pub fn new(render_context: &RenderContext, scene_pipelines: &PipelineBuilder) -> Self {
        let device = &render_context.wgpu_device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("texture scene mesh shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{} {}",
                    include_str!("./../shaders/utils.wgsl"),
                    include_str!("./../shaders/scene_textured_mesh.wgsl")
                )
                .into(),
            ),
        });

        Self {
            pipeline: scene_pipelines.create::<TexturedMeshVertex3>(
                "textured-mesh".to_string(),
                &shader,
                Some(wgpu::Face::Back),
            ),
            mesh_table: BTreeMap::new(),
        }
    }

    /// paint
    pub fn paint<'rp>(
        &'rp self,
        render_context: &RenderContext,
        scene_from_camera: &Isometry3F64,
        world_from_scene: &Isometry3F64,
        uniforms: &'rp VertexShaderUniformBuffers,
        render_pass: &mut wgpu::RenderPass<'rp>,
    ) {
        render_pass.set_pipeline(&self.pipeline);

        for mesh in self.mesh_table.values() {
            uniforms
                .camera_from_entity_pose_buffer
                .update_given_camera_and_entity(
                    &render_context.wgpu_queue,
                    &(world_from_scene * scene_from_camera),
                    &mesh._scene_from_entity,
                );
            render_pass.set_bind_group(1, &mesh._texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, mesh._vertex_buffer.slice(..));
            render_pass.draw(0..mesh._vertex_data.len() as u32, 0..1);
        }
    }
}
