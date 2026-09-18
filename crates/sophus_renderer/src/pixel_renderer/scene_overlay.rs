use eframe::wgpu;
use wgpu::util::DeviceExt;

use crate::{
    RenderContext,
    pipeline_builder::{
        LineVertex3,
        PipelineBuilder,
        PointVertex3,
    },
    prelude::*,
    renderables::{
        LineSegments3,
        PointCloud3,
    },
};

/// How many vertices one segment is drawn with: six for each piece of the strip which follows the
/// curve the projection bends it into. Must match `PIECES` in `overlay_line3.wgsl`.
const LINE_VERTICES: u32 = 6 * 24;

pub(crate) struct OverlayEntity {
    pub(crate) instance_count: u32,
    pub(crate) vertex_buffer: wgpu::Buffer,
}

/// The scene's lines and points, drawn over the finished image rather than into it.
///
/// These are the two renderables with a width in *pixels* rather than in metres: a landmark is not
/// a sphere of any radius and a bearing is not a rod, and both should stay legible however far
/// away they are. Everything else in a scene has a size, and is rasterized into the undistorted
/// intermediate or traced against the ray of each pixel.
///
/// A width in pixels is the reason they are drawn here rather than there. The intermediate is a
/// plane fitted to the view, or one of five ninety degree faces, and a pixel of it is not a pixel
/// of the image - so a width asked for in image pixels came out as something else, differently at
/// different parts of the image, and was then resampled by the warp on top of that. Drawn after
/// the warp, they are the width they were asked to be, and stay crisp.
///
/// What they give up by leaving the scene pass is the depth buffer, so they take the depth the
/// distortion pass leaves behind instead: it holds the inverse distance along the ray of every
/// pixel of the finished image, which is all an occlusion test needs.
pub(crate) struct SceneOverlayRenderer {
    pub(crate) line_pipeline: wgpu::RenderPipeline,
    pub(crate) point_pipeline: wgpu::RenderPipeline,
    pub(crate) depth_bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) lines: BTreeMap<String, OverlayEntity>,
    pub(crate) points: BTreeMap<String, OverlayEntity>,
}

impl SceneOverlayRenderer {
    pub(crate) fn new(render_context: &RenderContext, pixel_pipelines: &PipelineBuilder) -> Self {
        let device = &render_context.wgpu_device;

        let depth_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene overlay depth layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        // the inverse distance the distortion pass wrote, which is `r32float` and
                        // therefore never filtered
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let shader = |name: &str, body: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(
                    format!("{} {}", include_str!("./../shaders/utils.wgsl"), body).into(),
                ),
            })
        };

        Self {
            line_pipeline: pixel_pipelines.create_writing_inverse_depth::<LineVertex3>(
                "scene overlay line".to_string(),
                &shader(
                    "scene overlay line shader",
                    include_str!("./../shaders/overlay_line3.wgsl"),
                ),
                &[&depth_bind_group_layout],
            ),
            point_pipeline: pixel_pipelines.create_writing_inverse_depth::<PointVertex3>(
                "scene overlay point".to_string(),
                &shader(
                    "scene overlay point shader",
                    include_str!("./../shaders/overlay_point3.wgsl"),
                ),
                &[&depth_bind_group_layout],
            ),
            depth_bind_group_layout,
            lines: BTreeMap::new(),
            points: BTreeMap::new(),
        }
    }

    /// The entity's pose is applied here rather than carried into the shader, since these are
    /// drawn together from the camera's own pose - one uniform for all of them - and a renderable
    /// is uploaded again whenever it moves anyway.
    pub(crate) fn insert_lines(&mut self, render_context: &RenderContext, lines: &LineSegments3) {
        let world_from_entity = lines.world_from_entity;
        let vertex_data = lines
            .segments
            .iter()
            .map(|segment| LineVertex3 {
                _p0: world_from_entity.transform(segment.p0.cast()).cast().into(),
                _p1: world_from_entity.transform(segment.p1.cast()).cast().into(),
                _color: [
                    segment.color.r,
                    segment.color.g,
                    segment.color.b,
                    segment.color.a,
                ],
                _line_width: segment.line_width,
            })
            .collect::<Vec<_>>();

        self.lines.insert(
            lines.name.clone(),
            OverlayEntity {
                instance_count: vertex_data.len() as u32,
                vertex_buffer: render_context.wgpu_device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("scene overlay line buffer: {}", lines.name)),
                        contents: bytemuck::cast_slice(&vertex_data),
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                ),
            },
        );
    }

    pub(crate) fn insert_points(&mut self, render_context: &RenderContext, points: &PointCloud3) {
        let world_from_entity = points.world_from_entity;
        let vertex_data = points
            .points
            .iter()
            .map(|point| PointVertex3 {
                _pos: world_from_entity.transform(point.p.cast()).cast().into(),
                _point_size: point.point_size,
                _color: [point.color.r, point.color.g, point.color.b, point.color.a],
            })
            .collect::<Vec<_>>();

        self.points.insert(
            points.name.clone(),
            OverlayEntity {
                instance_count: vertex_data.len() as u32,
                vertex_buffer: render_context.wgpu_device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("scene overlay point buffer: {}", points.name)),
                        contents: bytemuck::cast_slice(&vertex_data),
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                ),
            },
        );
    }

    /// Whether there is anything to draw, and so whether the pass is worth running at all - it
    /// costs a copy of the scene's depth to read from.
    pub(crate) fn is_empty(&self) -> bool {
        self.lines.values().all(|line| line.instance_count == 0)
            && self.points.values().all(|point| point.instance_count == 0)
    }

    pub(crate) fn paint<'rp>(&'rp self, render_pass: &mut wgpu::RenderPass<'rp>) {
        render_pass.set_pipeline(&self.line_pipeline);
        for line in self.lines.values() {
            render_pass.set_vertex_buffer(0, line.vertex_buffer.slice(..));
            render_pass.draw(0..LINE_VERTICES, 0..line.instance_count);
        }

        render_pass.set_pipeline(&self.point_pipeline);
        for point in self.points.values() {
            render_pass.set_vertex_buffer(0, point.vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..point.instance_count);
        }
    }
}
