use std::collections::BTreeMap;
use std::sync::Mutex;

use eframe::egui_wgpu::wgpu::util::DeviceExt;
use wgpu::DepthStencilState;

use crate::offscreen_renderer::pixel_renderer::PointVertex2;
use crate::renderables::renderable2d::PointCloud2;
use crate::ViewerRenderState;

pub(crate) struct Point2dEntity {
    pub(crate) vertex_data: Vec<PointVertex2>,
    pub(crate) vertex_buffer: wgpu::Buffer,
}

impl Point2dEntity {
    pub(crate) fn new(wgpu_render_state: &ViewerRenderState, points: &PointCloud2) -> Self {
        let mut vertex_data = vec![];

        for point in points.points.iter() {
            let v = PointVertex2 {
                _pos: [point.p[0], point.p[1]],
                _color: [point.color.r, point.color.g, point.color.b, point.color.a],
                _point_size: point.point_size,
            };
            for _i in 0..6 {
                vertex_data.push(v);
            }
        }

        let vertex_buffer =
            wgpu_render_state
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Pixel point vertex buffer: {}", points.name)),
                    contents: bytemuck::cast_slice(&vertex_data),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        Self {
            vertex_data,
            vertex_buffer,
        }
    }
}

/// Pixel point renderer
pub struct PixelPointRenderer {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) points_table: BTreeMap<String, Point2dEntity>,
    pub(crate) interaction_vertex_buffer: wgpu::Buffer,
    pub(crate) show_interaction_marker: Mutex<bool>,
}

impl PixelPointRenderer {
    /// Create a new pixel point renderer
    pub fn new(
        wgpu_render_state: &ViewerRenderState,
        pipeline_layout: &wgpu::PipelineLayout,
        depth_stencil: Option<DepthStencilState>,
    ) -> Self {
        let device = &wgpu_render_state.device;

        let point_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pixel point shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{} {}",
                    include_str!("./pixel_utils.wgsl"),
                    include_str!("./point_pixel_shader.wgsl")
                )
                .into(),
            ),
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
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &point_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::TextureFormat::Rgba8UnormSrgb.into())],
                compilation_options: Default::default(),
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
            interaction_vertex_buffer,
            show_interaction_marker: Mutex::new(false),
            points_table: BTreeMap::new(),
        }
    }

    pub(crate) fn paint<'rp>(
        &'rp self,
        render_pass: &mut wgpu::RenderPass<'rp>,
        uniform_bind_group: &'rp wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, uniform_bind_group, &[]);
        for (_name, point) in self.points_table.iter() {
            render_pass.set_vertex_buffer(0, point.vertex_buffer.slice(..));
            render_pass.draw(0..point.vertex_data.len() as u32, 0..1);
        }

        let show_interaction_marker = self.show_interaction_marker.lock().unwrap();

        if show_interaction_marker.to_owned() {
            render_pass.set_vertex_buffer(0, self.interaction_vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..1);
        }
    }
}
