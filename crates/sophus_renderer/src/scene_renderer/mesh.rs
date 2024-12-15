use crate::pipeline_builder::MeshVertex3;
use crate::pipeline_builder::PipelineBuilder;
use crate::preludes::*;
use crate::renderables::scene_renderable::TriangleMesh3;
use crate::uniform_buffers::VertexShaderUniformBuffers;
use crate::RenderContext;
use sophus_lie::Isometry3F64;
use wgpu::util::DeviceExt;

pub(crate) struct Mesh3dEntity {
    pub(crate) vertex_data: Vec<MeshVertex3>,
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) world_from_entity: Isometry3F64,
}

impl Mesh3dEntity {
    /// Create a new 3D mesh entity
    pub fn new(render_context: &RenderContext, mesh: &TriangleMesh3) -> Self {
        let vertex_data: Vec<MeshVertex3> = mesh
            .triangles
            .iter()
            .flat_map(|trig| {
                vec![
                    MeshVertex3 {
                        _pos: [trig.p0[0], trig.p0[1], trig.p0[2]],
                        _color: [trig.color0.r, trig.color0.g, trig.color0.b, trig.color0.a],
                    },
                    MeshVertex3 {
                        _pos: [trig.p1[0], trig.p1[1], trig.p1[2]],
                        _color: [trig.color1.r, trig.color1.g, trig.color1.b, trig.color1.a],
                    },
                    MeshVertex3 {
                        _pos: [trig.p2[0], trig.p2[1], trig.p2[2]],
                        _color: [trig.color2.r, trig.color2.g, trig.color2.b, trig.color2.a],
                    },
                ]
            })
            .collect();

        let vertex_buffer =
            render_context
                .wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("3D mesh vertex buffer: {}", mesh.name)),
                    contents: bytemuck::cast_slice(&vertex_data),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        Self {
            vertex_data,
            vertex_buffer,
            world_from_entity: mesh.world_from_entity,
        }
    }
}

/// Scene mesh renderer
pub struct MeshRenderer {
    pub(crate) pipeline_without_culling: wgpu::RenderPipeline,
    pub(crate) pipeline_with_culling: wgpu::RenderPipeline,
    pub(crate) mesh_table: BTreeMap<String, Mesh3dEntity>,
}

impl MeshRenderer {
    /// Create a new scene mesh renderer
    pub fn new(render_context: &RenderContext, scene_pipelines: &PipelineBuilder) -> Self {
        let device = &render_context.wgpu_device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene mesh shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{} {}",
                    include_str!("./../shaders/utils.wgsl"),
                    include_str!("./../shaders/scene_mesh.wgsl")
                )
                .into(),
            ),
        });

        Self {
            pipeline_with_culling: scene_pipelines.create::<MeshVertex3>(
                "mesh with culling".to_string(),
                &shader,
                Some(wgpu::Face::Back),
            ),
            pipeline_without_culling: scene_pipelines.create::<MeshVertex3>(
                "mesh".to_string(),
                &shader,
                Some(wgpu::Face::Back),
            ),
            mesh_table: BTreeMap::new(),
        }
    }

    pub(crate) fn paint<'rp>(
        &'rp self,
        render_context: &RenderContext,
        scene_from_camera: &Isometry3F64,
        world_from_scene: &Isometry3F64,
        uniforms: &'rp VertexShaderUniformBuffers,
        render_pass: &mut wgpu::RenderPass<'rp>,
        backface_culling: bool,
    ) {
        let pipeline = if backface_culling {
            &self.pipeline_with_culling
        } else {
            &self.pipeline_without_culling
        };
        render_pass.set_pipeline(pipeline);

        for mesh in self.mesh_table.values() {
            uniforms
                .camera_from_entity_pose_buffer
                .update_given_camera_and_entity(
                    &render_context.wgpu_queue,
                    &(world_from_scene * scene_from_camera),
                    &mesh.world_from_entity,
                );
            render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            render_pass.draw(0..mesh.vertex_data.len() as u32, 0..1);
        }
    }
}
