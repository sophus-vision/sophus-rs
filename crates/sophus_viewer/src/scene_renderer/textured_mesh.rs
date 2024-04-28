use crate::renderable::TexturedTriangle3;
use crate::ViewerRenderState;
use bytemuck::Pod;
use bytemuck::Zeroable;
use eframe::egui_wgpu::wgpu::util::DeviceExt;
use std::collections::BTreeMap;
use wgpu::DepthStencilState;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TexturedMeshVertex3 {
    pub _pos: [f32; 3],
    pub _tex: [f32; 2],
}

// pub(crate) struct TexturedMeshes {
//     pub(crate) start_idx: u32,
//     pub(crate) end_idx: u32,

//     pub texture: wgpu::Texture,
//     pub bind_group: wgpu::BindGroup,
// }

pub struct TexturedMeshRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub depth_pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub mesh_table: BTreeMap<String, Vec<TexturedTriangle3>>,
    pub vertices: Vec<TexturedMeshVertex3>,
    //pub meshes: BTreeMap<String, TexturedMeshes>,
}

impl TexturedMeshRenderer {
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
                    include_str!("./utils.wgsl"),
                    include_str!("./mesh_scene_shader.wgsl")
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
            label: Some("mesh scene pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<TexturedMeshVertex3>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::TextureFormat::Rgba8UnormSrgb.into())],
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
                    array_stride: std::mem::size_of::<TexturedMeshVertex3>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "depth_fs_main",
                targets: &[Some(wgpu::TextureFormat::R32Float.into())],
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
            //meshes: BTreeMap::new(),
        }
    }

    pub fn paint<'rp>(
        &'rp self,
        render_pass: &mut wgpu::RenderPass<'rp>,
        bind_group: &'rp wgpu::BindGroup,
        dist_bind_group: &'rp wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_bind_group(1, dist_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.vertices.len() as u32, 0..1);
    }

    pub fn depth_paint<'rp>(
        &'rp self,
        depth_render_pass: &mut wgpu::RenderPass<'rp>,
        bind_group: &'rp wgpu::BindGroup,
        dist_bind_group: &'rp wgpu::BindGroup,
    ) {
        depth_render_pass.set_pipeline(&self.depth_pipeline);
        depth_render_pass.set_bind_group(0, bind_group, &[]);
        depth_render_pass.set_bind_group(1, dist_bind_group, &[]);
        depth_render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        depth_render_pass.draw(0..self.vertices.len() as u32, 0..1);
    }
}
