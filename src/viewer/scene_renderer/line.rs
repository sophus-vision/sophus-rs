use bytemuck::Pod;
use bytemuck::Zeroable;
use eframe::egui_wgpu::wgpu::util::DeviceExt;
use std::collections::BTreeMap;
use wgpu::DepthStencilState;

use crate::viewer::renderable::Line3;
use crate::viewer::ViewerRenderState;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LineVertex3 {
    pub _p0: [f32; 3],
    pub _p1: [f32; 3],
    pub _color: [f32; 4],
    pub _line_width: f32,
}

pub struct SceneLineRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub depth_pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub vertex_data: Vec<LineVertex3>,
    pub line_table: BTreeMap<String, Vec<Line3>>,
}

impl SceneLineRenderer {
    pub fn new(
        wgpu_render_state: &ViewerRenderState,
        pipeline_layout: &wgpu::PipelineLayout,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let device = &wgpu_render_state.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene line shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{} {}",
                    include_str!("./utils.wgsl"),
                    include_str!("./line_scene_shader.wgsl")
                )
                .into(),
            ),
        });

        // hack: generate a buffer of 1000 lines, because vertex buffer cannot be resized
        let mut vertex_data = vec![];
        for _i in 0..1000 {
            let v = LineVertex3 {
                _p0: [0.0, 0.0, 0.0],
                _p1: [0.0, 0.0, 0.0],
                _color: [1.0, 0.0, 0.0, 1.0],
                _line_width: 1.0,
            };

            for _i in 0..6 {
                vertex_data.push(v);
            }
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line scene pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex3>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1=>Float32x3, 2 => Float32x4, 3 => Float32],

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
            label: Some("depth line scene pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex3>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1=>Float32x3, 2 => Float32x4, 3 => Float32],
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
            vertex_data: vec![],
            line_table: BTreeMap::new(),
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
        render_pass.draw(0..self.vertex_data.len() as u32, 0..1);
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
        depth_render_pass.draw(0..self.vertex_data.len() as u32, 0..1);
    }
}
