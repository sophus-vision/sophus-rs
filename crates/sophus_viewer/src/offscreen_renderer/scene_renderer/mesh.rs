use std::collections::BTreeMap;

use bytemuck::Pod;
use bytemuck::Zeroable;
use eframe::egui_wgpu::wgpu::util::DeviceExt;
use wgpu::DepthStencilState;

use crate::renderables::renderable3d::Triangle3;
use crate::ViewerRenderState;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
/// 3D mesh vertex
pub struct MeshVertex3 {
    pub(crate) _pos: [f32; 3],
    pub(crate) _color: [f32; 4],
}

/// Scene mesh renderer
pub struct MeshRenderer {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) depth_pipeline: wgpu::RenderPipeline,
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) mesh_table: BTreeMap<String, Vec<Triangle3>>,
    pub(crate) vertices: Vec<MeshVertex3>,
}

impl MeshRenderer {
    /// Create a new scene mesh renderer
    pub fn new(
        wgpu_render_state: &ViewerRenderState,
        pipeline_layout: &wgpu::PipelineLayout,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let device = &wgpu_render_state.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene mesh shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{} {}",
                    include_str!("./scene_utils.wgsl"),
                    include_str!("./mesh_scene_shader.wgsl")
                )
                .into(),
            ),
        });

        // hack: generate a buffer of 1000 points, because vertex buffer cannot be resized
        let mut vertex_data = vec![];
        for _i in 0..1000 {
            vertex_data.push(MeshVertex3 {
                _pos: [0.0, 0.0, 0.0],
                _color: [1.0, 0.0, 0.0, 1.0],
            });
            vertex_data.push(MeshVertex3 {
                _pos: [0.0, 0.0, 0.0],
                _color: [1.0, 0.0, 0.0, 1.0],
            });
            vertex_data.push(MeshVertex3 {
                _pos: [0.0, 0.0, 0.0],
                _color: [1.0, 0.0, 0.0, 1.0],
            });
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh scene pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertex3>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
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

        let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("depth_mesh scene pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertex3>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "depth_fs_main",
                targets: &[Some(wgpu::TextureFormat::R32Float.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),

            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            depth_pipeline,
            vertex_buffer,
            vertices: vec![],
            mesh_table: BTreeMap::new(),
        }
    }

    pub(crate) fn paint<'rp>(
        &'rp self,
        render_pass: &mut wgpu::RenderPass<'rp>,
        bind_group: &'rp wgpu::BindGroup,
        dist_bind_group: &'rp wgpu::BindGroup,
        background_bind_group: &'rp wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_bind_group(1, dist_bind_group, &[]);
        render_pass.set_bind_group(2, background_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.vertices.len() as u32, 0..1);
    }

    pub(crate) fn depth_paint<'rp>(
        &'rp self,
        depth_render_pass: &mut wgpu::RenderPass<'rp>,
        bind_group: &'rp wgpu::BindGroup,
        dist_bind_group: &'rp wgpu::BindGroup,
        background_bind_group: &'rp wgpu::BindGroup,
    ) {
        depth_render_pass.set_pipeline(&self.depth_pipeline);
        depth_render_pass.set_bind_group(0, bind_group, &[]);
        depth_render_pass.set_bind_group(1, dist_bind_group, &[]);
        depth_render_pass.set_bind_group(2, background_bind_group, &[]);
        depth_render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        depth_render_pass.draw(0..self.vertices.len() as u32, 0..1);
    }
}
