use std::collections::BTreeMap;

use bytemuck::Pod;
use bytemuck::Zeroable;
use eframe::egui_wgpu::wgpu::util::DeviceExt;
use sophus_lie::Isometry3F64;
use wgpu::DepthStencilState;

use crate::renderables::renderable3d::PointCloud3;
use crate::renderer::scene_renderer::buffers::SceneRenderBuffers;
use crate::RenderContext;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct PointVertex3 {
    pub(crate) _pos: [f32; 3],
    pub(crate) _point_size: f32,
    pub(crate) _color: [f32; 4],
}

pub(crate) struct Point3dEntity {
    pub(crate) vertex_data: Vec<PointVertex3>,
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) scene_from_entity: Isometry3F64,
}

impl Point3dEntity {
    /// Create a new 2d line entity
    pub fn new(wgpu_render_state: &RenderContext, points: &PointCloud3) -> Self {
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
            wgpu_render_state
                .wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("3d point vertex buffer: {}", points.name)),
                    contents: bytemuck::cast_slice(&vertex_data),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        Self {
            vertex_data,
            vertex_buffer,
            scene_from_entity: points.scene_from_entity,
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
    pub fn new(
        wgpu_render_state: &RenderContext,
        pipeline_layout: &wgpu::PipelineLayout,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let device = &wgpu_render_state.wgpu_device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene point shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{} {}",
                    include_str!("./scene_utils.wgsl"),
                    include_str!("./point_scene_shader.wgsl")
                )
                .into(),
            ),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("point scene pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PointVertex3>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1=>Float32, 2 => Float32x4],

                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::TextureFormat::Rgba8UnormSrgb.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: depth_stencil.clone(),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            point_table: BTreeMap::new(),
        }
    }

    pub(crate) fn paint<'rp>(
        &'rp self,
        wgpu_render_state: &RenderContext,
        scene_from_camera: &Isometry3F64,
        buffers: &'rp SceneRenderBuffers,
        render_pass: &mut wgpu::RenderPass<'rp>,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &buffers.bind_group, &[]);

        for point in self.point_table.values() {
            buffers.view_uniform.update_given_camera_and_entity(
                &wgpu_render_state.wgpu_queue,
                scene_from_camera,
                &point.scene_from_entity,
            );
            render_pass.set_vertex_buffer(0, point.vertex_buffer.slice(..));
            render_pass.draw(0..point.vertex_data.len() as u32, 0..1);
        }
    }
}
