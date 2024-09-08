use std::collections::BTreeMap;

use bytemuck::Pod;
use bytemuck::Zeroable;
use eframe::egui_wgpu::wgpu::util::DeviceExt;
use sophus_lie::Isometry3F64;
use wgpu::DepthStencilState;

use crate::offscreen_renderer::scene_renderer::buffers::SceneRenderBuffers;
use crate::renderables::renderable3d::LineSegments3;
use crate::ViewerRenderState;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct LineVertex3 {
    pub(crate) _p0: [f32; 3],
    pub(crate) _p1: [f32; 3],
    pub(crate) _color: [f32; 4],
    pub(crate) _line_width: f32,
}

pub(crate) struct Line3dEntity {
    pub(crate) vertex_data: Vec<LineVertex3>,
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) scene_from_entity: Isometry3F64,
}

impl Line3dEntity {
    /// Create a new 3d line entity
    pub fn new(wgpu_render_state: &ViewerRenderState, lines: &LineSegments3) -> Self {
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
            wgpu_render_state
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(format!("3D line vertex buffer: {}", lines.name).as_str()),
                    contents: bytemuck::cast_slice(&vertex_data),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        Self {
            vertex_data,
            vertex_buffer,
            scene_from_entity: lines.scene_from_entity,
        }
    }
}

/// Scene line renderer
pub struct SceneLineRenderer {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) depth_pipeline: wgpu::RenderPipeline,
    pub(crate) line_table: BTreeMap<String, Line3dEntity>,
}

impl SceneLineRenderer {
    /// Create a new scene line renderer
    pub fn new(
        wgpu_render_state: &ViewerRenderState,
        pipeline_layout: &wgpu::PipelineLayout,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let device = &wgpu_render_state.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene line shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{} {}",
                    include_str!("./scene_utils.wgsl"),
                    include_str!("./line_scene_shader.wgsl")
                )
                .into(),
            ),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line scene pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex3>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1=>Float32x3, 2 => Float32x4, 3 => Float32],

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

        let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("depth line scene pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<LineVertex3>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1=>Float32x3, 2 => Float32x4, 3 => Float32],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "depth_fs_main",
                targets: &[Some(wgpu::TextureFormat::R32Float.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),

            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            depth_pipeline,
            line_table: BTreeMap::new(),
        }
    }

    pub(crate) fn paint<'rp>(
        &'rp self,
        wgpu_render_state: &ViewerRenderState,
        scene_from_camera: &Isometry3F64,
        buffers: &'rp SceneRenderBuffers,
        render_pass: &mut wgpu::RenderPass<'rp>,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &buffers.bind_group, &[]);
        render_pass.set_bind_group(1, &buffers.dist_bind_group, &[]);

        for line in self.line_table.values() {
            buffers.view_uniform.update_given_camera_and_entity(
                &wgpu_render_state.queue,
                scene_from_camera,
                &line.scene_from_entity,
            );
            render_pass.set_vertex_buffer(0, line.vertex_buffer.slice(..));
            render_pass.draw(0..line.vertex_data.len() as u32, 0..1);
        }
    }

    pub(crate) fn depth_paint<'rp>(
        &'rp self,
        wgpu_render_state: &ViewerRenderState,
        scene_from_camera: &Isometry3F64,
        buffers: &'rp SceneRenderBuffers,
        depth_render_pass: &mut wgpu::RenderPass<'rp>,
    ) {
        depth_render_pass.set_pipeline(&self.depth_pipeline);
        depth_render_pass.set_bind_group(0, &buffers.bind_group, &[]);
        depth_render_pass.set_bind_group(1, &buffers.dist_bind_group, &[]);
        for line in self.line_table.values() {
            buffers.view_uniform.update_given_camera_and_entity(
                &wgpu_render_state.queue,
                scene_from_camera,
                &line.scene_from_entity,
            );
            depth_render_pass.set_vertex_buffer(0, line.vertex_buffer.slice(..));
            depth_render_pass.draw(0..line.vertex_data.len() as u32, 0..1);
        }
    }
}
