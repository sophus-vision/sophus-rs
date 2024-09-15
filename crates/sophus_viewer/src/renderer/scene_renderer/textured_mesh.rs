use std::collections::BTreeMap;

use bytemuck::Pod;
use bytemuck::Zeroable;
use eframe::egui_wgpu::wgpu::util::DeviceExt;
use sophus_core::IsTensorLike;
use sophus_image::arc_image::ArcImage4U8;
use sophus_image::image_view::IsImageView;
use sophus_lie::Isometry3F64;
use wgpu::DepthStencilState;

use crate::renderables::renderable3d::TexturedTriangleMesh3;
use crate::renderer::scene_renderer::buffers::SceneRenderBuffers;
use crate::RenderContext;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct TexturedMeshVertex3 {
    pub(crate) _pos: [f32; 3],
    pub(crate) _tex: [f32; 2],
}

pub(crate) struct TexturedMeshEntity {
    pub(crate) vertex_data: Vec<TexturedMeshVertex3>,
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) _texture: wgpu::Texture,
    pub(crate) texture_bind_group: wgpu::BindGroup,
    pub(crate) scene_from_entity: Isometry3F64,
}

impl TexturedMeshEntity {
    pub(crate) fn new(
        wgpu_render_state: &RenderContext,
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
            wgpu_render_state
                .wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("3D mesh vertex buffer: {}", mesh.name)),
                    contents: bytemuck::cast_slice(&vertex_data),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        let device = &wgpu_render_state.wgpu_device;

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

        wgpu_render_state.wgpu_queue.write_texture(
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
            vertex_data,
            vertex_buffer,
            _texture: texture,
            texture_bind_group,
            scene_from_entity: mesh.scene_from_entity,
        }
    }
}

/// Scene textured mesh renderer
pub struct TexturedMeshRenderer {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) mesh_table: BTreeMap<String, TexturedMeshEntity>,
}

impl TexturedMeshRenderer {
    pub(crate) fn new(
        wgpu_render_state: &RenderContext,
        pipeline_layout: &wgpu::PipelineLayout,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let device = &wgpu_render_state.wgpu_device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("texture scene mesh shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{} {}",
                    include_str!("./scene_utils.wgsl"),
                    include_str!("./textured_mesh_scene_shader.wgsl")
                )
                .into(),
            ),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("textured mesh scene pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<TexturedMeshVertex3>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::TextureFormat::Rgba8UnormSrgb.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: depth_stencil.clone(),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            mesh_table: BTreeMap::new(),
        }
    }
    pub(crate) fn paint<'rp>(
        &'rp self,
        wgpu_render_state: &RenderContext,
        scene_from_camera: &Isometry3F64,
        buffers: &'rp SceneRenderBuffers,
        render_pass: &mut wgpu::RenderPass<'rp>,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &buffers.bind_group, &[]);

        for mesh in self.mesh_table.values() {
            buffers.view_uniform.update_given_camera_and_entity(
                &wgpu_render_state.wgpu_queue,
                scene_from_camera,
                &mesh.scene_from_entity,
            );
            render_pass.set_bind_group(1, &mesh.texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            render_pass.draw(0..mesh.vertex_data.len() as u32, 0..1);
        }
    }
}
