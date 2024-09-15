// use std::num::NonZeroU64;

// // Add this to the existing imports
// use wgpu::util::DeviceExt;
// use wgpu::DepthStencilState;

// use crate::renderer::camera::properties::RenderCameraProperties;
// use crate::RenderContext;

// // Add this struct to store distortion parameters
// #[repr(C)]
// #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
// struct DistortionParams {
//     k1: f32,
//     k2: f32,
//     p1: f32,
//     p2: f32,
// }

// // Add these fields to the SceneRenderer struct
// pub struct DistortionRenderer {
//     distortion_pipeline: wgpu::RenderPipeline,
//     distortion_bind_group: wgpu::BindGroup,
//     distortion_params_buffer: wgpu::Buffer,
// }

// impl DistortionRenderer {
//     pub fn new(
//         wgpu_render_state: &RenderContext,
//         camera_properties: &RenderCameraProperties,
//         depth_stencil: Option<DepthStencilState>,
//     ) -> Self {
//         let distortion_shader =
//             wgpu_render_state
//                 .wgpu_device
//                 .create_shader_module(wgpu::ShaderModuleDescriptor {
//                     label: Some("distortion shader"),
//                     source: wgpu::ShaderSource::Wgsl(
//                         include_str!("./distortion_shader.wgsl").into(),
//                     ),
//                 });

//         let distortion_bind_group_layout = wgpu_render_state.wgpu_device.create_bind_group_layout(
//             &wgpu::BindGroupLayoutDescriptor {
//                 entries: &[
//                     wgpu::BindGroupLayoutEntry {
//                         binding: 0,
//                         visibility: wgpu::ShaderStages::FRAGMENT,
//                         ty: wgpu::BindingType::Texture {
//                             sample_type: wgpu::TextureSampleType::Float { filterable: true },
//                             view_dimension: wgpu::TextureViewDimension::D2,
//                             multisampled: false,
//                         },
//                         count: None,
//                     },
//                     wgpu::BindGroupLayoutEntry {
//                         binding: 1,
//                         visibility: wgpu::ShaderStages::FRAGMENT,
//                         ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
//                         count: None,
//                     },
//                     wgpu::BindGroupLayoutEntry {
//                         binding: 2,
//                         visibility: wgpu::ShaderStages::FRAGMENT,
//                         ty: wgpu::BindingType::Buffer {
//                             ty: wgpu::BufferBindingType::Uniform,
//                             has_dynamic_offset: false,
//                             min_binding_size: None,
//                         },
//                         count: None,
//                     },
//                 ],
//                 label: Some("distortion bind group layout"),
//             },
//         );

//         let distortion_pipeline_layout =
//             wgpu_render_state
//                 .wgpu_device
//                 .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
//                     label: Some("distortion pipeline layout"),
//                     bind_group_layouts: &[&distortion_bind_group_layout],
//                     push_constant_ranges: &[],
//                 });

//         let distortion_pipeline =
//             wgpu_render_state
//                 .wgpu_device
//                 .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
//                     label: Some("distortion pipeline"),
//                     layout: Some(&distortion_pipeline_layout),
//                     vertex: wgpu::VertexState {
//                         module: &distortion_shader,
//                         entry_point: "vs_main",
//                         buffers: &[],
//                         compilation_options: Default::default(),
//                     },
//                     fragment: Some(wgpu::FragmentState {
//                         module: &distortion_shader,
//                         entry_point: "fs_main",
//                         targets: &[Some(wgpu::ColorTargetState {
//                             format: wgpu::TextureFormat::Rgba8UnormSrgb,
//                             blend: Some(wgpu::BlendState::REPLACE),
//                             write_mask: wgpu::ColorWrites::ALL,
//                         })],
//                         compilation_options: Default::default(),
//                     }),
//                     primitive: wgpu::PrimitiveState {
//                         topology: wgpu::PrimitiveTopology::TriangleList,
//                         strip_index_format: None,
//                         front_face: wgpu::FrontFace::Ccw,
//                         cull_mode: Some(wgpu::Face::Back),
//                         polygon_mode: wgpu::PolygonMode::Fill,
//                         unclipped_depth: false,
//                         conservative: false,
//                     },
//                     depth_stencil: None,
//                     multisample: wgpu::MultisampleState::default(),
//                     multiview: None,
//                 });

//         let distortion_params = DistortionParams {
//             k1: 0.1,
//             k2: 0.05,
//             p1: 0.01,
//             p2: 0.01,
//         };

//         let distortion_params_buffer =
//             wgpu_render_state
//                 .wgpu_device
//                 .create_buffer_init(&wgpu::util::BufferInitDescriptor {
//                     label: Some("Distortion params buffer"),
//                     contents: bytemuck::cast_slice(&[distortion_params]),
//                     usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
//                 });

//         let distortion_bind_group =
//             wgpu_render_state
//                 .wgpu_device
//                 .create_bind_group(&wgpu::BindGroupDescriptor {
//                     label: Some("Distortion bind group"),
//                     layout: &distortion_bind_group_layout,
//                     entries: &[wgpu::BindGroupEntry {
//                         binding: 0,
//                         resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
//                             buffer: &distortion_params_buffer,
//                             offset: 0,
//                             size: Some(NonZeroU64::new(std::mem::size_of::<DistortionParams>() as u64).unwrap()),,
//                         }),
//                     }],
//                 });

//         Self {
//             // ... existing fields ...
//             distortion_pipeline,
//             distortion_bind_group,
//             distortion_params_buffer,
//         }
//     }

//     // Modify the paint method to apply the distortion effect
//     pub(crate) fn paint<'rp>(
//         &'rp self,
//         state: &RenderContext,
//         scene_from_camera: &Isometry3F64,
//         command_encoder: &'rp mut wgpu::CommandEncoder,
//         texture_view: &'rp wgpu::TextureView,
//         depth: &DepthTextures,
//         backface_culling: bool,
//     ) {
//         // ... existing rendering code ...

//         // Apply distortion effect
//         let mut distortion_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
//             label: Some("Distortion pass"),
//             color_attachments: &[Some(wgpu::RenderPassColorAttachment {
//                 view: texture_view,
//                 resolve_target: None,
//                 ops: wgpu::Operations {
//                     load: wgpu::LoadOp::Load,
//                     store: wgpu::StoreOp::Store,
//                 },
//             })],
//             depth_stencil_attachment: None,
//             occlusion_query_set: None,
//             timestamp_writes: None,
//         });

//         distortion_pass.set_pipeline(&self.distortion_pipeline);
//         distortion_pass.set_bind_group(0, &self.buffers.bind_group, &[]);
//         distortion_pass.set_bind_group(2, &self.distortion_bind_group, &[]);
//         distortion_pass.draw(0..3, 0..1);
//     }
// }
