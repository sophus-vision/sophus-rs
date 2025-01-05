use crate::pipeline_builder::LineVertex2;
use crate::pipeline_builder::PipelineBuilder;
use crate::prelude::*;
use crate::renderables::pixel_renderable::LineSegments2;
use crate::RenderContext;
use wgpu::util::DeviceExt;
pub(crate) struct Line2dEntity {
    pub(crate) vertex_data: Vec<LineVertex2>,
    pub(crate) vertex_buffer: wgpu::Buffer,
}

impl Line2dEntity {
    pub(crate) fn new(render_context: &RenderContext, lines: &LineSegments2) -> Self {
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
            render_context
                .wgpu_device
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
    pub fn new(render_context: &RenderContext, pixel_pipelines: &PipelineBuilder) -> Self {
        let device = &render_context.wgpu_device;

        let line_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pixel line shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{} {}",
                    include_str!("./../shaders/utils.wgsl"),
                    include_str!("./../shaders/pixel_line.wgsl")
                )
                .into(),
            ),
        });

        Self {
            lines_table: BTreeMap::new(),
            pipeline: pixel_pipelines.create::<LineVertex2>("line".to_string(), &line_shader, None),
        }
    }

    pub(crate) fn paint<'rp>(&'rp self, render_pass: &mut wgpu::RenderPass<'rp>) {
        render_pass.set_pipeline(&self.pipeline);
        for (_name, line) in self.lines_table.iter() {
            render_pass.set_vertex_buffer(0, line.vertex_buffer.slice(..));
            render_pass.draw(0..line.vertex_data.len() as u32, 0..1);
        }
    }
}
