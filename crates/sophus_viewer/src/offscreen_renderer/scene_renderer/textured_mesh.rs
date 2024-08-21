use std::collections::BTreeMap;

use bytemuck::Pod;
use bytemuck::Zeroable;
use eframe::egui_wgpu::wgpu::util::DeviceExt;
use wgpu::DepthStencilState;

use crate::renderables::renderable3d::TexturedTriangle3;
use crate::ViewerRenderState;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
/// 3D textured mesh vertex
pub struct TexturedMeshVertex3 {
    pub(crate) _pos: [f32; 3],
    pub(crate) _tex: [f32; 2],
}

/// Scene textured mesh renderer
pub struct TexturedMeshRenderer {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) depth_pipeline: wgpu::RenderPipeline,
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) mesh_table: BTreeMap<String, Vec<TexturedTriangle3>>,
    pub(crate) vertices: Vec<TexturedMeshVertex3>,
}

impl TexturedMeshRenderer {
    /// Create a new scene textured mesh renderer
    pub fn new(
        wgpu_render_state: &ViewerRenderState,
        pipeline_layout: &wgpu::PipelineLayout,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let device = &wgpu_render_state.device;

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

        // hack: generate a buffer of 1000 points, because vertex buffer cannot be resized
        let mut vertex_data = vec![];
        for _i in 0..1000 {
            vertex_data.push(TexturedMeshVertex3 {
                _pos: [0.0, 0.0, 0.0],
                _tex: [0.0, 0.0],
            });
            vertex_data.push(TexturedMeshVertex3 {
                _pos: [0.0, 0.0, 0.0],
                _tex: [0.0, 0.0],
            });
            vertex_data.push(TexturedMeshVertex3 {
                _pos: [0.0, 0.0, 0.0],
                _tex: [0.0, 0.0],
            });
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
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

        let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("depth textured mesh scene pipeline"),
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
                entry_point: "depth_fs_main",
                targets: &[Some(wgpu::TextureFormat::R32Float.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: depth_stencil.clone(),
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
        uniform_bind_group: &'rp wgpu::BindGroup,
        distortion_bind_group: &'rp wgpu::BindGroup,
        texture_bind_group: &'rp wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, uniform_bind_group, &[]);
        render_pass.set_bind_group(1, distortion_bind_group, &[]);
        render_pass.set_bind_group(2, texture_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.vertices.len() as u32, 0..1);
    }

    pub(crate) fn depth_paint<'rp>(
        &'rp self,
        depth_render_pass: &mut wgpu::RenderPass<'rp>,
        bind_group: &'rp wgpu::BindGroup,
        dist_bind_group: &'rp wgpu::BindGroup,
        texture_bind_group: &'rp wgpu::BindGroup,
    ) {
        depth_render_pass.set_pipeline(&self.depth_pipeline);
        depth_render_pass.set_bind_group(0, bind_group, &[]);
        depth_render_pass.set_bind_group(1, dist_bind_group, &[]);
        depth_render_pass.set_bind_group(2, texture_bind_group, &[]);

        depth_render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        depth_render_pass.draw(0..self.vertices.len() as u32, 0..1);
    }
}
