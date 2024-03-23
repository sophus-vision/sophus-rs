use crate::viewer::renderable::Point2;
use crate::viewer::PointVertex2;
use crate::viewer::ViewerRenderState;

use eframe::egui_wgpu::wgpu::util::DeviceExt;
use std::collections::BTreeMap;
use std::sync::Mutex;
use wgpu::DepthStencilState;

pub struct PixelPointRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub points_table: BTreeMap<String, Vec<Point2>>,
    pub vertex_data: Vec<PointVertex2>,

    pub interaction_vertex_buffer: wgpu::Buffer,
    pub show_interaction_marker: Mutex<bool>,
}

impl PixelPointRenderer {
    pub fn new(
        wgpu_render_state: &ViewerRenderState,
        pipeline_layout: &wgpu::PipelineLayout,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let device = &wgpu_render_state.device;

        let point_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pixel point shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("./point_pixel_shader.wgsl").into()),
        });

        // hack: generate a buffer of 1000 points, because vertex buffer cannot be resized
        let mut point_vertex_data = vec![];
        for _i in 0..1000 {
            for _i in 0..6 {
                point_vertex_data.push(PointVertex2 {
                    _pos: [0.0, 0.0],
                    _color: [1.0, 0.0, 0.0, 1.0],
                    _point_size: 5.0,
                });
            }
        }
        let point_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pixel point vertex buffer"),
            contents: bytemuck::cast_slice(&point_vertex_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let mut interaction_vertices = vec![];
        for _i in 0..6 {
            interaction_vertices.push(PointVertex2 {
                _pos: [0.0, 0.0],
                _color: [1.0, 0.0, 0.0, 1.0],
                _point_size: 5.0,
            });
        }

        let interaction_vertex_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("interaction vertex buffer"),
                contents: bytemuck::cast_slice(&interaction_vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

        let point_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pixel point pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &point_shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PointVertex2>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1=>Float32, 2 => Float32x4],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &point_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::TextureFormat::Rgba8UnormSrgb.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline: point_pipeline,
            vertex_buffer: point_vertex_buffer,
            vertex_data: vec![],
            interaction_vertex_buffer,
            show_interaction_marker: Mutex::new(false),
            points_table: BTreeMap::new(),
        }
    }

    pub fn paint<'rp>(
        &'rp self,
        render_pass: &mut wgpu::RenderPass<'rp>,
        uniform_bind_group: &'rp wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.vertex_data.len() as u32, 0..1);

        let show_interaction_marker = self.show_interaction_marker.lock().unwrap();

        if show_interaction_marker.to_owned() {
            render_pass.set_bind_group(0, uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.interaction_vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..1);
        }
    }
}
