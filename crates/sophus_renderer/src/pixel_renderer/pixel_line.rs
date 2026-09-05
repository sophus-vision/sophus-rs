use eframe::wgpu;
use wgpu::util::DeviceExt;

use crate::{
    RenderContext,
    pipeline_builder::{
        LineVertex2,
        PipelineBuilder,
    },
    prelude::*,
    renderables::LineSegments2,
};
pub(crate) struct Line2dEntity {
    pub(crate) instance_count: u32,
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

            vertex_data.push(LineVertex2 {
                _p0: [p0[0], p0[1]],
                _p1: [p1[0], p1[1]],
                _normal: normal,
                _color: [line.color.r, line.color.g, line.color.b, line.color.a],
                _line_width: line.line_width,
            });
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
            instance_count: vertex_data.len() as u32,
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
        for line in self.lines_table.values() {
            render_pass.set_vertex_buffer(0, line.vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..line.instance_count);
        }
    }
}
