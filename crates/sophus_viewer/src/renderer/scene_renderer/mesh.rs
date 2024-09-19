use std::collections::BTreeMap;

use bytemuck::Pod;
use bytemuck::Zeroable;
use eframe::egui_wgpu::wgpu::util::DeviceExt;
use sophus_lie::Isometry3F64;
use wgpu::DepthStencilState;

use crate::renderables::renderable3d::TriangleMesh3;
use crate::renderer::uniform_buffers::VertexShaderUniformBuffers;
use crate::RenderContext;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct MeshVertex3 {
    pub(crate) _pos: [f32; 3],
    pub(crate) _color: [f32; 4],
}

pub(crate) struct Mesh3dEntity {
    pub(crate) vertex_data: Vec<MeshVertex3>,
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) scene_from_entity: Isometry3F64,
}

impl Mesh3dEntity {
    /// Create a new 3D mesh entity
    pub fn new(wgpu_render_state: &RenderContext, mesh: &TriangleMesh3) -> Self {
        let vertex_data: Vec<MeshVertex3> = mesh
            .triangles
            .iter()
            .flat_map(|trig| {
                vec![
                    MeshVertex3 {
                        _pos: [trig.p0[0], trig.p0[1], trig.p0[2]],
                        _color: [trig.color0.r, trig.color0.g, trig.color0.b, trig.color0.a],
                    },
                    MeshVertex3 {
                        _pos: [trig.p1[0], trig.p1[1], trig.p1[2]],
                        _color: [trig.color1.r, trig.color1.g, trig.color1.b, trig.color1.a],
                    },
                    MeshVertex3 {
                        _pos: [trig.p2[0], trig.p2[1], trig.p2[2]],
                        _color: [trig.color2.r, trig.color2.g, trig.color2.b, trig.color2.a],
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

        Self {
            vertex_data,
            vertex_buffer,
            scene_from_entity: mesh.scene_from_entity,
        }
    }
}

/// Scene mesh renderer
pub struct MeshRenderer {
    pub(crate) pipeline_without_culling: wgpu::RenderPipeline,
    pub(crate) pipeline_with_culling: wgpu::RenderPipeline,
    pub(crate) mesh_table: BTreeMap<String, Mesh3dEntity>,
}

impl MeshRenderer {
    // Define a function to create a render pipeline with specified cull mode
    fn create_render_pipeline(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
        depth_stencil: Option<wgpu::DepthStencilState>,
        cull_mode: Option<wgpu::Face>,
    ) -> wgpu::RenderPipeline {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh scene pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertex3>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::TextureFormat::Rgba32Float.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode,
                ..Default::default()
            },
            depth_stencil: depth_stencil.clone(),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        })
    }

    /// Create a new scene mesh renderer
    pub fn new(
        wgpu_render_state: &RenderContext,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let device = &wgpu_render_state.wgpu_device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene mesh shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{} {}",
                    include_str!("./shaders/scene_utils.wgsl"),
                    include_str!("./shaders/mesh.wgsl")
                )
                .into(),
            ),
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene render layout"),
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

        // let texture_bind_group_layout =
        //     device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        //         label: Some("scene render texture layout"),
        //         entries: &[wgpu::BindGroupLayoutEntry {
        //             binding: 0,
        //             visibility: wgpu::ShaderStages::FRAGMENT,
        //             ty: wgpu::BindingType::StorageTexture {
        //                 access: wgpu::StorageTextureAccess::WriteOnly,
        //                 format: wgpu::TextureFormat::R32Float,
        //                 view_dimension: wgpu::TextureViewDimension::D2,
        //             },
        //             count: None,
        //         }],
        //     });

        // Create a new pipeline layout that includes both bind group layouts
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh Renderer Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline_with_culling = Self::create_render_pipeline(
            device,
            &shader,
            &pipeline_layout,
            depth_stencil.clone(),
            Some(wgpu::Face::Back),
        );

        let pipeline_without_culling = Self::create_render_pipeline(
            device,
            &shader,
            &pipeline_layout,
            depth_stencil.clone(),
            None,
        );

        Self {
            pipeline_with_culling,
            pipeline_without_culling,
            mesh_table: BTreeMap::new(),
        }
    }

    pub(crate) fn paint<'rp>(
        &'rp self,
        wgpu_render_state: &RenderContext,
        scene_from_camera: &Isometry3F64,
        buffers: &'rp VertexShaderUniformBuffers,
        render_pass: &mut wgpu::RenderPass<'rp>,
        backface_culling: bool,
    ) {
        let pipeline = if backface_culling {
            &self.pipeline_with_culling
        } else {
            &self.pipeline_without_culling
        };
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &buffers.bind_group, &[]);
        //  render_pass.set_bind_group(1, &buffers.texture_bind_group, &[]);

        for mesh in self.mesh_table.values() {
            buffers
                .camera_from_entity_pose_buffer
                .update_given_camera_and_entity(
                    &wgpu_render_state.wgpu_queue,
                    scene_from_camera,
                    &mesh.scene_from_entity,
                );
            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            render_pass.draw(0..mesh.vertex_data.len() as u32, 0..1);
        }
    }
}
