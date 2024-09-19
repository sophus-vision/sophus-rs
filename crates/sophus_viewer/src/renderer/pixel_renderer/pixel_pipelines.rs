use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use bytemuck::Pod;
use bytemuck::Zeroable;
use wgpu::BufferAddress;
use wgpu::ShaderModule;

use crate::renderer::pixel_renderer::LineVertex2;
use crate::RenderContext;

struct Builder {}

struct TargetTexture {
    rgba_output_format: wgpu::TextureFormat,
}

struct PipelineAndPass {
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    render_pipeline: wgpu::RenderPipeline,
}

struct PixelPipelines {
    context: RenderContext,
    rgba_target: Arc<TargetTexture>,
    pipeline_and_passes: HashMap<String, PipelineAndPass>,
}

trait PixelVertex {
    fn array_stride() -> BufferAddress;

    fn step_mode() -> wgpu::VertexStepMode {
        wgpu::VertexStepMode::Vertex
    }

    fn attr() -> Vec<wgpu::VertexAttribute>;
}

impl PixelVertex for LineVertex2 {
    fn array_stride() -> BufferAddress {
        std::mem::size_of::<LineVertex2>() as wgpu::BufferAddress
    }

    fn attr() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![0 => Float32x2, 1=>Float32x4, 2 => Float32x2, 3 => Float32]
            .to_vec()
    }
}

impl<'a> PixelPipelines {
    pub fn new(context: &RenderContext, target: &Arc<TargetTexture>) -> Self {
        PixelPipelines {
            context: context.clone(),
            rgba_target: target.clone(),
            pipeline_and_passes: HashMap::new(),
        }
    }

    pub fn add<Vertex: PixelVertex>(&mut self, name: String, shader: ShaderModule) {
        let device = self.context.wgpu_device.clone();

        let hardcoded_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("`{}` pixel uniform render layout", name)),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(16),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(16),
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("`{}` pixel pipeline layout", name)),
            bind_group_layouts: &[&hardcoded_uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("`{}` pixel pipeline", name)),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: Vertex::array_stride(),
                    step_mode: Vertex::step_mode(),
                    attributes: &Vertex::attr(),
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(self.rgba_target.rgba_output_format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        self.pipeline_and_passes.insert(
            name,
            PipelineAndPass {
                uniform_bind_group_layout: hardcoded_uniform_bind_group_layout,
                render_pipeline: pipeline,
            },
        );
    }
}

// let pipeline =        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {

//     label: Some("pixel line pipeline"),
//     layout: Some(pipeline_layout),
//     vertex: wgpu::VertexState {
//         module: &line_shader,
//         entry_point: "vs_main",
//         buffers: &[wgpu::VertexBufferLayout {
//             array_stride: std::mem::size_of::<LineVertex2>() as wgpu::BufferAddress,
//             step_mode: wgpu::VertexStepMode::Vertex,
//             attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1=>Float32x4, 2 => Float32x2, 3 => Float32],
//         }],
//         compilation_options: Default::default(),

//     },
//     fragment: Some(wgpu::FragmentState {
//         module: &line_shader,
//         entry_point: "fs_main",
//         targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
//         compilation_options: Default::default(),
//     }),
//     primitive: wgpu::PrimitiveState {
//         topology: wgpu::PrimitiveTopology::TriangleList,
//         front_face: wgpu::FrontFace::Ccw,
//         ..Default::default()
//     },
//     depth_stencil: None,
//     multisample: wgpu::MultisampleState::default(),
//     multiview: None,
// // });
//     });

// device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
//     label: Some("pixel line pipeline"),
//     layout: Some(pipeline_layout),
//     vertex: wgpu::VertexState {
//         module: &line_shader,
//         entry_point: "vs_main",
//         buffers: &[wgpu::VertexBufferLayout {
//             array_stride: std::mem::size_of::<LineVertex2>() as wgpu::BufferAddress,
//             step_mode: wgpu::VertexStepMode::Vertex,
//             attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1=>Float32x4, 2 => Float32x2, 3 => Float32],
//         }],
//         compilation_options: Default::default(),

//     },
//     fragment: Some(wgpu::FragmentState {
//         module: &line_shader,
//         entry_point: "fs_main",
//         targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
//         compilation_options: Default::default(),
//     }),
//     primitive: wgpu::PrimitiveState {
//         topology: wgpu::PrimitiveTopology::TriangleList,
//         front_face: wgpu::FrontFace::Ccw,
//         ..Default::default()
//     },
//     depth_stencil: None,
//     multisample: wgpu::MultisampleState::default(),
//     multiview: None,
// });
//     }
// }
