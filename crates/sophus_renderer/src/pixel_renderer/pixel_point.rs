use eframe::wgpu;
use wgpu::util::DeviceExt;

use crate::{
    RenderContext,
    pipeline_builder::{
        PipelineBuilder,
        PointVertex2,
    },
    prelude::*,
    renderables::PointCloud2,
};

pub(crate) struct Point2dEntity {
    pub(crate) instance_count: u32,
    pub(crate) vertex_buffer: wgpu::Buffer,
}

impl Point2dEntity {
    pub(crate) fn new(render_context: &RenderContext, points: &PointCloud2) -> Self {
        let mut vertex_data = vec![];

        for point in points.points.iter() {
            let v = PointVertex2 {
                _pos: [point.p[0], point.p[1]],
                _color: [point.color.r, point.color.g, point.color.b, point.color.a],
                _point_size: point.point_size,
            };
            vertex_data.push(v);
        }

        let vertex_buffer =
            render_context
                .wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Pixel point vertex buffer: {}", points.name)),
                    contents: bytemuck::cast_slice(&vertex_data),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        Self {
            instance_count: vertex_data.len() as u32,
            vertex_buffer,
        }
    }
}

/// Pixel point renderer
pub struct PixelPointRenderer {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) points_table: BTreeMap<String, Point2dEntity>,
}

impl PixelPointRenderer {
    /// Create a new pixel point renderer
    pub fn new(render_context: &RenderContext, pixel_pipelines: &PipelineBuilder) -> Self {
        let device = &render_context.wgpu_device;

        let point_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pixel point shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{} {}",
                    include_str!("./../shaders/utils.wgsl"),
                    include_str!("./../shaders/pixel_point.wgsl")
                )
                .into(),
            ),
        });

        Self {
            pipeline: pixel_pipelines.create::<PointVertex2>(
                "point".to_string(),
                &point_shader,
                None,
            ),
            points_table: BTreeMap::new(),
        }
    }

    pub(crate) fn paint<'rp>(&'rp self, render_pass: &mut wgpu::RenderPass<'rp>) {
        render_pass.set_pipeline(&self.pipeline);
        for point in self.points_table.values() {
            render_pass.set_vertex_buffer(0, point.vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..point.instance_count);
        }
    }
}
