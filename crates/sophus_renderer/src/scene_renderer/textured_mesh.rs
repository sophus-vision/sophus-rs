use eframe::wgpu;
use sophus_image::ArcImage4U8;
use sophus_lie::Isometry3F64;
use wgpu::util::DeviceExt;

use crate::{
    RenderContext,
    pipeline_builder::{
        PipelineBuilder,
        TexturedMeshVertex3,
    },
    prelude::*,
    renderables::TexturedTriangleMesh3,
    scene_renderer::SceneView,
    uniform_buffers::{
        MAX_SCENE_ENTITIES,
        VertexShaderUniformBuffers,
    },
};

/// mesh entity
pub struct TexturedMeshEntity {
    /// drawn as edges, whatever the view says
    pub(crate) wireframe: bool,
    pub(crate) vertex_data: Vec<TexturedMeshVertex3>,
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) _texture: wgpu::Texture,
    pub(crate) texture_bind_group: wgpu::BindGroup,
    pub(crate) world_from_entity: Isometry3F64,
}

impl TexturedMeshEntity {
    /// new
    pub fn new(
        render_context: &RenderContext,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
        mesh: &TexturedTriangleMesh3,
    ) -> Self {
        let image: &ArcImage4U8 = &mesh.texture;
        let vertex_data: Vec<TexturedMeshVertex3> = mesh
            .triangles
            .iter()
            .flat_map(|trig| {
                // flat-shading normal, as for an untextured mesh
                let n = (trig.p1 - trig.p0).cross(&(trig.p2 - trig.p0)).normalize();
                vec![
                    TexturedMeshVertex3 {
                        _pos: [trig.p0[0], trig.p0[1], trig.p0[2]],
                        _normal: n.into(),
                        _tex: [trig.tex0[0], trig.tex0[1]],
                    },
                    TexturedMeshVertex3 {
                        _pos: [trig.p1[0], trig.p1[1], trig.p1[2]],
                        _normal: n.into(),
                        _tex: [trig.tex1[0], trig.tex1[1]],
                    },
                    TexturedMeshVertex3 {
                        _pos: [trig.p2[0], trig.p2[1], trig.p2[2]],
                        _normal: n.into(),
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

        // Filtered, and with a chain of smaller copies to filter between. Point sampling a
        // texture on a surface which rakes away from the camera picks one texel of the many a
        // pixel covers, arbitrarily, so the surface seethes as the camera moves - which is what a
        // texture on a floor mostly does.
        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let texture_size = wgpu::Extent3d {
            width: image.image_size().width as u32,
            height: image.image_size().height as u32,
            depth_or_array_layers: 1,
        };
        // one level per halving, down to a single texel
        let mip_level_count = 32 - texture_size.width.max(texture_size.height).leading_zeros();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some(&format!("textured mesh texture: {}", mesh.name)),
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: texture_bind_group_layout,
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
            label: Some(&format!("textured mesh bind group: {}", mesh.name)),
        });

        // The chain is built here rather than on the gpu: the texture is uploaded once when the
        // mesh is, so there is nothing to be gained from a pass per level, and a box average is
        // what a mip level is.
        let mut level_size = (texture_size.width as usize, texture_size.height as usize);
        let mut texels: Vec<u8> = bytemuck::cast_slice(
            image
                .tensor
                .scalar_view()
                .as_slice()
                .expect("the texture is contiguous"),
        )
        .to_vec();
        for level in 0..mip_level_count {
            if level > 0 {
                let (width, height) = (level_size.0.div_ceil(2), level_size.1.div_ceil(2));
                let mut smaller = vec![0u8; width * height * 4];
                for v in 0..height {
                    for u in 0..width {
                        for channel in 0..4 {
                            // the four texels above, or fewer along an odd edge
                            let mut sum = 0u32;
                            let mut count = 0u32;
                            for (du, dv) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                                let (su, sv) = (2 * u + du, 2 * v + dv);
                                if su < level_size.0 && sv < level_size.1 {
                                    sum += texels[(sv * level_size.0 + su) * 4 + channel] as u32;
                                    count += 1;
                                }
                            }
                            smaller[(v * width + u) * 4 + channel] = (sum / count) as u8;
                        }
                    }
                }
                texels = smaller;
                level_size = (width, height);
            }

            render_context.wgpu_queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &texels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * level_size.0 as u32),
                    rows_per_image: Some(level_size.1 as u32),
                },
                wgpu::Extent3d {
                    width: level_size.0 as u32,
                    height: level_size.1 as u32,
                    depth_or_array_layers: 1,
                },
            );
        }

        Self {
            wireframe: mesh.wireframe,
            vertex_data,
            vertex_buffer,
            _texture: texture,
            texture_bind_group,
            world_from_entity: mesh.world_from_entity,
        }
    }
}

/// Scene textured mesh renderer
pub struct TexturedMeshRenderer {
    /// pipeline, with back faces dropped
    pub pipeline_with_culling: wgpu::RenderPipeline,
    /// pipeline, drawing both sides
    pub pipeline_without_culling: wgpu::RenderPipeline,
    /// layout of the per-entity texture bind group
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
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

        let texture_bind_group_layout =
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
                label: Some("textured mesh bind group layout"),
            });

        Self {
            pipeline_with_culling: scene_pipelines
                .create_with_bind_group_layouts::<TexturedMeshVertex3>(
                    "textured-mesh with culling".to_string(),
                    &shader,
                    Some(wgpu::Face::Back),
                    &[&texture_bind_group_layout],
                ),
            pipeline_without_culling: scene_pipelines
                .create_with_bind_group_layouts::<TexturedMeshVertex3>(
                    "textured-mesh".to_string(),
                    &shader,
                    None,
                    &[&texture_bind_group_layout],
                ),
            texture_bind_group_layout,
            mesh_table: BTreeMap::new(),
        }
    }

    /// paint
    pub(crate) fn paint<'rp>(
        &'rp self,
        render_context: &RenderContext,
        view: &SceneView,
        uniforms: &'rp VertexShaderUniformBuffers,
        render_pass: &mut wgpu::RenderPass<'rp>,
        entity_slot: &mut u32,
        backface_culling: bool,
    ) {
        render_pass.set_pipeline(match backface_culling {
            true => &self.pipeline_with_culling,
            false => &self.pipeline_without_culling,
        });

        for mesh in self.mesh_table.values() {
            if *entity_slot >= MAX_SCENE_ENTITIES {
                log::warn!("more than {MAX_SCENE_ENTITIES} scene entities - skipping the rest");
                break;
            }
            let pose_offset = uniforms
                .camera_from_entity_pose_buffer
                .update_given_camera_and_entity(
                    &render_context.wgpu_queue,
                    *entity_slot,
                    &view.world_from_camera,
                    &mesh.world_from_entity,
                    view.light_in_world,
                    mesh.wireframe,
                );
            *entity_slot += 1;

            render_pass.set_bind_group(0, &uniforms.render_bind_group, &[pose_offset]);
            render_pass.set_bind_group(1, &mesh.texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            render_pass.draw(0..mesh.vertex_data.len() as u32, 0..1);
        }
    }
}
