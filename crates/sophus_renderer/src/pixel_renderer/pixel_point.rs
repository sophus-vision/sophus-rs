use crate::pipeline_builder::PipelineBuilder;
use crate::pipeline_builder::PointVertex2;
use crate::preludes::*;
use crate::renderables::pixel_renderable::PointCloud2;
use crate::RenderContext;
use eframe::egui::mutex::Mutex;
use wgpu::util::DeviceExt;

pub(crate) struct Point2dEntity {
    pub(crate) vertex_data: Vec<PointVertex2>,
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
            for _i in 0..6 {
                vertex_data.push(v);
            }
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
            vertex_data,
            vertex_buffer,
        }
    }
}

/// Pixel point renderer
pub struct PixelPointRenderer {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) points_table: BTreeMap<String, Point2dEntity>,
    pub(crate) show_interaction_marker: Mutex<bool>,
    pub(crate) interaction_vertex_buffer: wgpu::Buffer,
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

        let mut interaction_vertices = vec![];
        for _i in 0..6 {
            interaction_vertices.push(PointVertex2 {
                _pos: [0.0, 0.0],
                _color: [1.0, 0.0, 0.0, 1.0],
                _point_size: 5.0,
            });
        }

        let interaction_vertex_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("interaction vertex buffer"),
                contents: bytemuck::cast_slice(&interaction_vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            pipeline: pixel_pipelines.create::<PointVertex2>(
                "point".to_string(),
                &point_shader,
                None,
            ),
            interaction_vertex_buffer,
            show_interaction_marker: Mutex::new(false),
            points_table: BTreeMap::new(),
        }
    }

    pub(crate) fn paint<'rp>(&'rp self, render_pass: &mut wgpu::RenderPass<'rp>) {
        render_pass.set_pipeline(&self.pipeline);
        for (_name, point) in self.points_table.iter() {
            render_pass.set_vertex_buffer(0, point.vertex_buffer.slice(..));
            render_pass.draw(0..point.vertex_data.len() as u32, 0..1);
        }

        let show_interaction_marker = self.show_interaction_marker.lock();

        if *show_interaction_marker {
            render_pass.set_vertex_buffer(0, self.interaction_vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..1);
        }
    }
}
