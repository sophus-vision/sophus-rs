use eframe::{
    egui::mutex::Mutex,
    wgpu,
};
use wgpu::util::DeviceExt;

use crate::{
    RenderContext,
    pipeline_builder::{
        EllipseVertex2,
        PipelineBuilder,
    },
    prelude::*,
    renderables::{
        Ellipse2,
        EllipseCloud2,
    },
};

/// The vertex of one ellipse, or [None] for a degenerate one - which has no interior to draw and
/// no inverse to draw it with.
pub(crate) fn ellipse_vertex(ellipse: &Ellipse2) -> Option<EllipseVertex2> {
    let to_unit_circle = ellipse.shape.try_inverse()?;
    Some(EllipseVertex2 {
        _center: [ellipse.center[0], ellipse.center[1]],
        // the bounding box of `{ c + A u : |u| <= 1 }` is the row norms of `A`
        _half_extent: [
            ellipse.shape.row(0).norm() as f32,
            ellipse.shape.row(1).norm() as f32,
        ],
        _to_unit_circle: [
            to_unit_circle[(0, 0)] as f32,
            to_unit_circle[(0, 1)] as f32,
            to_unit_circle[(1, 0)] as f32,
            to_unit_circle[(1, 1)] as f32,
        ],
        _color: [
            ellipse.color.r,
            ellipse.color.g,
            ellipse.color.b,
            ellipse.color.a,
        ],
        _line_width: ellipse.line_width,
        _padding: [0.0; 3],
    })
}

/// How many ellipses the interaction pivot is drawn from: a ring about each axis, a bar along
/// each, and the dot at the middle - each with a shape of grey behind it.
pub(crate) const PIVOT_ELLIPSES: usize = 14;

pub(crate) struct Ellipse2dEntity {
    pub(crate) instance_count: u32,
    pub(crate) vertex_buffer: wgpu::Buffer,
}

impl Ellipse2dEntity {
    pub(crate) fn new(render_context: &RenderContext, ellipses: &EllipseCloud2) -> Self {
        let mut vertex_data = vec![];

        for ellipse in ellipses.ellipses.iter() {
            if let Some(vertex) = ellipse_vertex(ellipse) {
                vertex_data.push(vertex);
            }
        }

        let vertex_buffer =
            render_context
                .wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Pixel ellipse vertex buffer: {}", ellipses.name)),
                    contents: bytemuck::cast_slice(&vertex_data),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        Self {
            instance_count: vertex_data.len() as u32,
            vertex_buffer,
        }
    }
}

/// Pixel ellipse renderer
pub struct PixelEllipseRenderer {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) ellipses_table: BTreeMap<String, Ellipse2dEntity>,
    /// The pivot an interaction turns about, drawn as rings - written every frame it is shown,
    /// rather than being an entity of the scene.
    pub(crate) show_interaction_marker: Mutex<bool>,
    pub(crate) interaction_vertex_buffer: wgpu::Buffer,
}

impl PixelEllipseRenderer {
    /// Create a new pixel ellipse renderer
    pub fn new(render_context: &RenderContext, pixel_pipelines: &PipelineBuilder) -> Self {
        let device = &render_context.wgpu_device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pixel ellipse shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{} {}",
                    include_str!("./../shaders/utils.wgsl"),
                    include_str!("./../shaders/pixel_ellipse.wgsl")
                )
                .into(),
            ),
        });

        let interaction_vertex_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("interaction ellipse vertex buffer"),
                contents: bytemuck::cast_slice(
                    &[EllipseVertex2 {
                        _center: [0.0, 0.0],
                        _half_extent: [0.0, 0.0],
                        _to_unit_circle: [0.0; 4],
                        _color: [0.0; 4],
                        _line_width: 0.0,
                        _padding: [0.0; 3],
                    }; PIVOT_ELLIPSES],
                ),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            pipeline: pixel_pipelines.create::<EllipseVertex2>(
                "ellipse".to_string(),
                &shader,
                None,
            ),
            ellipses_table: BTreeMap::new(),
            show_interaction_marker: Mutex::new(false),
            interaction_vertex_buffer,
        }
    }

    pub(crate) fn paint<'rp>(&'rp self, render_pass: &mut wgpu::RenderPass<'rp>) {
        render_pass.set_pipeline(&self.pipeline);
        for ellipse in self.ellipses_table.values() {
            render_pass.set_vertex_buffer(0, ellipse.vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..ellipse.instance_count);
        }

        if *self.show_interaction_marker.lock() {
            render_pass.set_vertex_buffer(0, self.interaction_vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..PIVOT_ELLIPSES as u32);
        }
    }
}
