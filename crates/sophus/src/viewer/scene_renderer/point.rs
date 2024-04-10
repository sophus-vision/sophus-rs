use crate::viewer::renderable::Point3;
use crate::viewer::ViewerRenderState;
use bytemuck::Pod;
use bytemuck::Zeroable;
use eframe::egui_wgpu::wgpu::util::DeviceExt;
use std::collections::BTreeMap;
use wgpu::DepthStencilState;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PointVertex3 {
    pub _pos: [f32; 3],
    pub _point_size: f32,
    pub _color: [f32; 4],
}

pub struct ScenePointRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub point_table: BTreeMap<String, Vec<Point3>>,
    pub vertex_data: Vec<PointVertex3>,
}

impl ScenePointRenderer {
    pub fn new(
        wgpu_render_state: &ViewerRenderState,
        pipeline_layout: &wgpu::PipelineLayout,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let device = &wgpu_render_state.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scene point shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{} {}",
                    include_str!("./utils.wgsl"),
                    include_str!("./point_scene_shader.wgsl")
                )
                .into(),
            ),
        });

        // hack: generate a buffer of 1000 points, because vertex buffer cannot be resized
        let mut vertex_data = vec![];
        for _i in 0..1000 {
            for _i in 0..6 {
                vertex_data.push(PointVertex3 {
                    _pos: [0.0, 0.0, 0.0],
                    _color: [1.0, 0.0, 0.0, 1.0],
                    _point_size: 5.0,
                });
            }
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene point vertex buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh scene pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PointVertex3>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1=>Float32, 2 => Float32x4],

                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::TextureFormat::Rgba8UnormSrgb.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            vertex_buffer,
            vertex_data: vec![],
            point_table: BTreeMap::new(),
        }
    }

    pub fn paint<'rp>(
        &'rp self,
        render_pass: &mut wgpu::RenderPass<'rp>,
        bind_group: &'rp wgpu::BindGroup,
        dist_bind_group: &'rp wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.set_bind_group(1, dist_bind_group, &[]); // NEW!
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.vertex_data.len() as u32, 0..1);
    }
}
