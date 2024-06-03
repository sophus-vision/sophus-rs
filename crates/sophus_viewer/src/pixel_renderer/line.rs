use std::collections::BTreeMap;

use eframe::egui_wgpu::wgpu::util::DeviceExt;
use nalgebra::SVector;
use wgpu::DepthStencilState;

use crate::pixel_renderer::LineVertex2;
use crate::renderables::renderable2d::Line2;
use crate::ViewerRenderState;

/// Pixel line renderer
pub struct PixelLineRenderer {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) lines_table: BTreeMap<String, Vec<Line2>>,
    pub(crate) vertex_data: Vec<LineVertex2>,
}

impl PixelLineRenderer {
    /// Create a new pixel line renderer
    pub fn new(
        wgpu_render_state: &ViewerRenderState,
        pipeline_layout: &wgpu::PipelineLayout,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let device = &wgpu_render_state.device;

        let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pixel line shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./line_pixel_shader.wgsl").into()),
        });

        // hack: generate a buffer of 1000 lines, because vertex buffer cannot be resized
        let mut line_vertex_data = vec![];
        for i in 0..1000 {
            let p0 = SVector::<f32, 2>::new(i as f32, 0.0);
            let p1 = SVector::<f32, 2>::new(i as f32, 1000.0);
            let d = (p0 - p1).normalize();
            let normal = [d[1], -d[0]];

            let v0 = LineVertex2 {
                _pos: [p0[0], p0[1]],
                _normal: normal,
                _color: [1.0, 0.0, 0.0, 1.0],
                _line_width: 5.0,
            };
            let v1 = LineVertex2 {
                _pos: [p1[0], p1[1]],
                _normal: normal,
                _color: [1.0, 0.0, 0.0, 1.0],
                _line_width: 5.0,
            };
            line_vertex_data.push(v0);
            line_vertex_data.push(v0);
            line_vertex_data.push(v1);
            line_vertex_data.push(v0);
            line_vertex_data.push(v1);
            line_vertex_data.push(v1);
        }

        let line_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pixel line vertex buffer"),
            contents: bytemuck::cast_slice(&line_vertex_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pixel line pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &line_shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex2>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1=>Float32x4, 2 => Float32x2, 3 => Float32],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &line_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::TextureFormat::Rgba8UnormSrgb.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline: line_pipeline,
            vertex_buffer: line_vertex_buffer,
            lines_table: BTreeMap::new(),
            vertex_data: vec![],
        }
    }

    pub(crate) fn paint<'rp>(
        &'rp self,
        render_pass: &mut wgpu::RenderPass<'rp>,
        uniform_bind_group: &'rp wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.vertex_data.len() as u32, 0..1);
    }
}
