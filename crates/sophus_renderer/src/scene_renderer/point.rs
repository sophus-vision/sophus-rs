use crate::pipeline_builder::PipelineBuilder;
use crate::pipeline_builder::PointVertex3;
use crate::preludes::*;
use crate::renderables::scene_renderable::PointCloud3;
use crate::uniform_buffers::VertexShaderUniformBuffers;
use crate::RenderContext;
use sophus_lie::Isometry3F64;
use wgpu::util::DeviceExt;
pub(crate) struct Point3dEntity {
    pub(crate) vertex_data: Vec<PointVertex3>,
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
            for _i in 0..6 {
                vertex_data.push(v);
            }
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
            vertex_data,
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
        scene_from_camera: &Isometry3F64,
        world_from_scene: &Isometry3F64,
        buffers: &'rp VertexShaderUniformBuffers,
        render_pass: &mut wgpu::RenderPass<'rp>,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &buffers.render_bind_group, &[]);

        for point in self.point_table.values() {
            buffers
                .camera_from_entity_pose_buffer
                .update_given_camera_and_entity(
                    &render_context.wgpu_queue,
                    &(world_from_scene * scene_from_camera),
                    &point.world_from_entity,
                );
            render_pass.set_vertex_buffer(0, point.vertex_buffer.slice(..));
            render_pass.draw(0..point.vertex_data.len() as u32, 0..1);
        }
    }
}
