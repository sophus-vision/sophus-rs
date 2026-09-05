use eframe::wgpu;
use sophus_lie::Isometry3F64;
use wgpu::util::DeviceExt;

use crate::{
    RenderContext,
    pipeline_builder::{
        LineVertex3,
        PipelineBuilder,
    },
    prelude::*,
    renderables::LineSegments3,
    scene_renderer::SceneView,
    uniform_buffers::{
        MAX_SCENE_ENTITIES,
        VertexShaderUniformBuffers,
    },
};

pub(crate) struct Line3dEntity {
    pub(crate) instance_count: u32,
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
            vertex_data.push(v0);
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
            instance_count: vertex_data.len() as u32,
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
        view: &SceneView,
        uniforms: &'rp VertexShaderUniformBuffers,
        render_pass: &mut wgpu::RenderPass<'rp>,
        entity_slot: &mut u32,
    ) {
        render_pass.set_pipeline(&self.pipeline);

        for line in self.line_table.values() {
            if *entity_slot >= MAX_SCENE_ENTITIES {
                log::warn!("more than {MAX_SCENE_ENTITIES} scene entities - skipping the rest");
                break;
            }
            let pose_offset = uniforms
                .camera_from_entity_pose_buffer
                .update_given_camera_and_entity(
                    &render_context.wgpu_queue,
                    *entity_slot,
                    &view.world_from_camera,
                    &line.world_from_entity,
                    view.light_in_world,
                    // a line and a point are already edges
                    false,
                );
            *entity_slot += 1;

            render_pass.set_bind_group(0, &uniforms.render_bind_group, &[pose_offset]);
            render_pass.set_vertex_buffer(0, line.vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..line.instance_count);
        }
    }
}
