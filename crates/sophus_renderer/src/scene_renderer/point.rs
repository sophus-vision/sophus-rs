use eframe::wgpu;
use sophus_lie::Isometry3F64;
use wgpu::util::DeviceExt;

use crate::{
    RenderContext,
    pipeline_builder::{
        PipelineBuilder,
        PointVertex3,
    },
    prelude::*,
    renderables::PointCloud3,
    scene_renderer::SceneView,
    uniform_buffers::{
        MAX_SCENE_ENTITIES,
        VertexShaderUniformBuffers,
    },
};
pub(crate) struct Point3dEntity {
    pub(crate) instance_count: u32,
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) world_from_entity: Isometry3F64,
}

impl Point3dEntity {
    /// Create a new 2d line entity
    pub fn new(render_context: &RenderContext, points: &PointCloud3) -> Self {
        let mut vertex_data = vec![];
        for point in points.points.iter() {
            let v = PointVertex3 {
                _pos: [point.p[0], point.p[1], point.p[2]],
                _color: [point.color.r, point.color.g, point.color.b, point.color.a],
                _point_size: point.point_size,
            };
            vertex_data.push(v);
        }

        let vertex_buffer =
            render_context
                .wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("3d point vertex buffer: {}", points.name)),
                    contents: bytemuck::cast_slice(&vertex_data),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        Self {
            instance_count: vertex_data.len() as u32,
            vertex_buffer,
            world_from_entity: points.world_from_entity,
        }
    }
}

/// Scene point renderer
pub struct ScenePointRenderer {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) point_table: BTreeMap<String, Point3dEntity>,
}

impl ScenePointRenderer {
    /// Create a new scene point renderer
    pub fn new(render_context: &RenderContext, scene_pipelines: &PipelineBuilder) -> Self {
        let device = &render_context.wgpu_device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene point shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{} {}",
                    include_str!("./../shaders/utils.wgsl"),
                    include_str!("./../shaders/scene_point.wgsl")
                )
                .into(),
            ),
        });

        Self {
            pipeline: scene_pipelines.create::<PointVertex3>("point".to_string(), &shader, None),
            point_table: BTreeMap::new(),
        }
    }

    pub(crate) fn paint<'rp>(
        &'rp self,
        render_context: &RenderContext,
        view: &SceneView,
        buffers: &'rp VertexShaderUniformBuffers,
        render_pass: &mut wgpu::RenderPass<'rp>,
        entity_slot: &mut u32,
    ) {
        render_pass.set_pipeline(&self.pipeline);

        for point in self.point_table.values() {
            if *entity_slot >= MAX_SCENE_ENTITIES {
                log::warn!("more than {MAX_SCENE_ENTITIES} scene entities - skipping the rest");
                break;
            }
            let pose_offset = buffers
                .camera_from_entity_pose_buffer
                .update_given_camera_and_entity(
                    &render_context.wgpu_queue,
                    *entity_slot,
                    &view.world_from_camera,
                    &point.world_from_entity,
                    view.light_in_world,
                    // a line and a point are already edges
                    false,
                );
            *entity_slot += 1;

            render_pass.set_bind_group(0, &buffers.render_bind_group, &[pose_offset]);
            render_pass.set_vertex_buffer(0, point.vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..point.instance_count);
        }
    }
}
