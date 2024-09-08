use std::collections::BTreeMap;

use eframe::egui_wgpu::wgpu::util::DeviceExt;
use wgpu::DepthStencilState;

use crate::offscreen_renderer::pixel_renderer::LineVertex2;
use crate::renderables::renderable2d::LineSegments2;
use crate::ViewerRenderState;

pub(crate) struct Line2dEntity {
    pub(crate) vertex_data: Vec<LineVertex2>,
    pub(crate) vertex_buffer: wgpu::Buffer,
}

impl Line2dEntity {
    pub(crate) fn new(wgpu_render_state: &ViewerRenderState, lines: &LineSegments2) -> Self {
        let mut vertex_data = vec![];
        for line in lines.segments.iter() {
            let p0 = line.p0;
            let p1 = line.p1;
            let d = (p0 - p1).normalize();
            let normal = [d[1], -d[0]];

            let v0 = LineVertex2 {
                _pos: [p0[0], p0[1]],
                _normal: normal,
                _color: [line.color.r, line.color.g, line.color.b, line.color.a],
                _line_width: line.line_width,
            };
            let v1 = LineVertex2 {
                _pos: [p1[0], p1[1]],
                _normal: normal,
                _color: [line.color.r, line.color.g, line.color.b, line.color.a],
                _line_width: line.line_width,
            };
            vertex_data.push(v0);
            vertex_data.push(v0);
            vertex_data.push(v1);
            vertex_data.push(v0);
            vertex_data.push(v1);
            vertex_data.push(v1);
        }

        let vertex_buffer =
            wgpu_render_state
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Pixel line vertex buffer: {}", lines.name)),
                    contents: bytemuck::cast_slice(&vertex_data),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        Self {
            vertex_data,
            vertex_buffer,
        }
    }
}

/// Pixel line renderer
pub struct PixelLineRenderer {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) lines_table: BTreeMap<String, Line2dEntity>,
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
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{} {}",
                    include_str!("./pixel_utils.wgsl"),
                    include_str!("./line_pixel_shader.wgsl")
                )
                .into(),
            ),
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
                compilation_options: Default::default(),

            },
            fragment: Some(wgpu::FragmentState {
                module: &line_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::TextureFormat::Rgba8UnormSrgb.into())],
                compilation_options: Default::default(),
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
            lines_table: BTreeMap::new(),
        }
    }

    pub(crate) fn paint<'rp>(
        &'rp self,
        render_pass: &mut wgpu::RenderPass<'rp>,
        uniform_bind_group: &'rp wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, uniform_bind_group, &[]);
        for (_name, line) in self.lines_table.iter() {
            render_pass.set_vertex_buffer(0, line.vertex_buffer.slice(..));
            render_pass.draw(0..line.vertex_data.len() as u32, 0..1);
        }
    }
}
