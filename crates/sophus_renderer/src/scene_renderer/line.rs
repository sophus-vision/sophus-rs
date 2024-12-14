use crate::pipeline_builder::LineVertex3;
use crate::pipeline_builder::PipelineBuilder;
use crate::preludes::*;
use crate::renderables::scene_renderable::LineSegments3;
use crate::uniform_buffers::VertexShaderUniformBuffers;
use crate::RenderContext;
use sophus_lie::Isometry3F64;
use wgpu::util::DeviceExt;

pub(crate) struct Line3dEntity {
    pub(crate) vertex_data: Vec<LineVertex3>,
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) world_from_entity: Isometry3F64,
}

impl Line3dEntity {
    /// Create a new 3d line entity
    pub fn new(render_context: &RenderContext, lines: &LineSegments3) -> Self {
        let mut vertex_data = vec![];
        for line in lines.segments.iter() {
            let p0 = line.p0;
            let p1 = line.p1;

            let v0 = LineVertex3 {
                _p0: [p0[0], p0[1], p0[2]],
                _p1: [p1[0], p1[1], p1[2]],
                _color: [line.color.r, line.color.g, line.color.b, line.color.a],
                _line_width: line.line_width,
            };
            let v1 = LineVertex3 {
                _p0: [p0[0], p0[1], p0[2]],
                _p1: [p1[0], p1[1], p1[2]],
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
                    label: Some(format!("3D line vertex buffer: {}", lines.name).as_str()),
                    contents: bytemuck::cast_slice(&vertex_data),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        Self {
            vertex_data,
            vertex_buffer,
            world_from_entity: lines.world_from_entity,
        }
    }
}

/// Scene line renderer
pub struct SceneLineRenderer {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) line_table: BTreeMap<String, Line3dEntity>,
}

impl SceneLineRenderer {
    /// Create a new scene line renderer
    pub fn new(render_context: &RenderContext, scene_pipelines: &PipelineBuilder) -> Self {
        let shader =
            render_context
                .wgpu_device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("scene line shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        format!(
                            "{} {}",
                            include_str!("./../shaders/utils.wgsl"),
                            include_str!("./../shaders/scene_line.wgsl")
                        )
                        .into(),
                    ),
                });

        Self {
            pipeline: scene_pipelines.create::<LineVertex3>("line".to_string(), &shader, None),
            line_table: BTreeMap::new(),
        }
    }

    pub(crate) fn paint<'rp>(
        &'rp self,
        render_context: &RenderContext,
        scene_from_camera: &Isometry3F64,
        world_from_scene: &Isometry3F64,
        uniforms: &'rp VertexShaderUniformBuffers,
        render_pass: &mut wgpu::RenderPass<'rp>,
    ) {
        render_pass.set_pipeline(&self.pipeline);

        for line in self.line_table.values() {
            uniforms
                .camera_from_entity_pose_buffer
                .update_given_camera_and_entity(
                    &render_context.wgpu_queue,
                    &(world_from_scene * scene_from_camera),
                    &line.world_from_entity,
                );
            render_pass.set_vertex_buffer(0, line.vertex_buffer.slice(..));
            render_pass.draw(0..line.vertex_data.len() as u32, 0..1);
        }
    }
}
