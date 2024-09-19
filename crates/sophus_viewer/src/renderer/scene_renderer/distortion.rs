// Strategy:
// 1. Render 3d scene against alpha=0 background, with depth in w, as 4f32 rgba_texture
// 2. Copy z-buffer to ndc_depth texture
// 2. In compute shader
//    a) load background image
//    b) lookup depth from ndc_depth texture using distortion model
//    c) lookup rgba from rgbd_texture using distortion model
//    d) mix with background image and store rgba in rgba_texture_distorted, 4u8
//    e) convert ndc_depth to metric depth and store in depth_texture_distorted, f32
// 3. Render 2d overlay on top of the rgba_texture_distorted

use eframe::egui_wgpu::wgpu::util::DeviceExt;
use wgpu::DepthStencilState;

use crate::renderer::camera::properties::RenderCameraProperties;
use crate::renderer::textures::depth::DepthTextures;
use crate::renderer::textures::rgba::RgbdTexture;
use crate::RenderContext;
use sophus_image::ImageSize;

/// Scene line renderer
pub struct DistortionRenderer {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_background: wgpu::ComputePipeline,
    bind_group_layout_background: wgpu::BindGroupLayout,
    pub(crate) frustum_uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    uniform_bind_group_layout: wgpu::BindGroupLayout,
}

impl DistortionRenderer {
    /// Create a new scene line renderer
    pub fn new(
        wgpu_render_state: &RenderContext,
        depth_stencil: Option<DepthStencilState>,
        rgba: &RgbdTexture,
        camera_properties: &RenderCameraProperties,
    ) -> Self {
        let device = &wgpu_render_state.wgpu_device;
        let frustum_uniforms = camera_properties.to_uniform();

        let frustum_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("frustum buffer"),
            contents: bytemuck::cast_slice(&[frustum_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("uniform layout"),
                entries: &[
                    // frustum uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frustum_uniform_buffer.as_entire_binding(),
            }],
        });
        let (pipeline_background, bind_group_layout_background) = {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("distortion shader with background"),
                source: wgpu::ShaderSource::Wgsl(
                    format!(
                        "{} {}",
                        include_str!("./shaders/scene_utils.wgsl"),
                        include_str!("./shaders/distortion.wgsl")
                    )
                    .into(),
                ),
            });
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("distortion bind group layout with background"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::R32Float,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("distortion pipeline layout with background"),
                bind_group_layouts: &[&uniform_bind_group_layout, &bind_group_layout],
                push_constant_ranges: &[],
            });

            (
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("distortion pipeline with background"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: "distort_with_background",
                    compilation_options: Default::default(),
                }),
                bind_group_layout,
            )
        };
        let (pipeline, bind_group_layout) = {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("distortion shader"),
                source: wgpu::ShaderSource::Wgsl(
                    format!(
                        "{} {}",
                        include_str!("./shaders/scene_utils.wgsl"),
                        include_str!("./shaders/distortion.wgsl")
                    )
                    .into(),
                ),
            });
            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("distortion bind group layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::R32Float,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("distortion pipeline layout"),
                bind_group_layouts: &[(&uniform_bind_group_layout), &bind_group_layout],
                push_constant_ranges: &[],
            });

            (
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("distortion pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: "distort",
                    compilation_options: Default::default(),
                }),
                bind_group_layout,
            )
        };

        Self {
            uniform_bind_group,
            pipeline,
            bind_group_layout,
            pipeline_background,
            bind_group_layout_background,
            frustum_uniform_buffer,
            uniform_bind_group_layout,
        }
    }

    pub(crate) fn run(
        &self,
        context: &RenderContext,
        mut command_encoder: wgpu::CommandEncoder,
        rgba: &RgbdTexture,
        depth: &DepthTextures,
        background_texture: &Option<wgpu::Texture>,
        view_port_size: &ImageSize,
    ) {
        println!("distorting image: {:?}", view_port_size);

        // let input_image = ArcImage4F32::from_image_size_and_val(
        //     *view_port_size,
        //     VecF32::<4>::new(1.0, 0.0, 0.0, 1.0),
        // );
        // let input_texture = context.wgpu_device.create_texture_with_data(
        //     &context.wgpu_queue,
        //     &wgpu::TextureDescriptor {
        //         label: Some("Input Texture"),
        //         size: wgpu::Extent3d {
        //             width: view_port_size.width as u32,
        //             height: view_port_size.height as u32,
        //             depth_or_array_layers: 1,
        //         },
        //         mip_level_count: 1,
        //         sample_count: 1,
        //         dimension: wgpu::TextureDimension::D2,
        //         format: wgpu::TextureFormat::Rgba32Float,
        //         usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        //         view_formats: &[],
        //     },
        //     wgpu::util::TextureDataOrder::LayerMajor,
        //     bytemuck::cast_slice(input_image.tensor.scalar_view().as_slice().unwrap()),
        // );

        println!(
            "rgba texture: {:?} {:?}",
            rgba.rgba_texture.width(),
            rgba.rgba_texture.height()
        );

        match background_texture {
            Some(background_texture) => {
                let bind_group =
                    context
                        .wgpu_device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Compute Bind Group with background"),
                            layout: &self.bind_group_layout_background,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &rgba.rgba_texture_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(
                                        &rgba.rgba_texture_view_distorted,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: wgpu::BindingResource::TextureView(
                                        &background_texture
                                            .create_view(&wgpu::TextureViewDescriptor::default()),
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::TextureView(
                                        &depth.main_render_ndc_z_texture.ndc_z_texture_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: wgpu::BindingResource::TextureView(
                                        &depth.main_render_ndc_z_texture.z_distorted_texture_view,
                                    ),
                                },
                            ],
                        });

                {
                    let mut compute_pass =
                        command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                    const WORKGROUP_SIZE: u32 = 16;
                    compute_pass.set_pipeline(&self.pipeline_background);
                    compute_pass.set_bind_group(1, &bind_group, &[]);
                    compute_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

                    compute_pass.dispatch_workgroups(
                        (view_port_size.width as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
                        (view_port_size.height as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
                        1,
                    );
                }
                context
                    .wgpu_queue
                    .submit(std::iter::once(command_encoder.finish()));
            }
            None => {
                let bind_group =
                    context
                        .wgpu_device
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("Compute Bind Group"),
                            layout: &self.bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(
                                        &rgba.rgba_texture_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(
                                        &rgba.rgba_texture_view_distorted,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: wgpu::BindingResource::TextureView(
                                        &depth.main_render_ndc_z_texture.ndc_z_texture_view,
                                    ),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 4,
                                    resource: wgpu::BindingResource::TextureView(
                                        &depth.main_render_ndc_z_texture.z_distorted_texture_view,
                                    ),
                                },
                            ],
                        });

                {
                    let mut compute_pass =
                        command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                    const WORKGROUP_SIZE: u32 = 16;
                    compute_pass.set_pipeline(&self.pipeline);
                    compute_pass.set_bind_group(1, &bind_group, &[]);
                    compute_pass.set_bind_group(0, &self.uniform_bind_group, &[]);

                    compute_pass.dispatch_workgroups(
                        (view_port_size.width as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
                        (view_port_size.height as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
                        1,
                    );
                }
                context
                    .wgpu_queue
                    .submit(std::iter::once(command_encoder.finish()));
            }
        }

        // let rgba_u8;
        // {
        //     let buffer_slice = output_buffer.slice(..);
        //     let (tx, rx) = std::sync::mpsc::channel();
        //     buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        //         tx.send(result).unwrap();
        //     });
        //     context.wgpu_device.poll(wgpu::Maintain::Wait);

        //     if rx.recv().unwrap().is_err() {
        //         panic!("Failed to map buffer");
        //     }
        //     let data = buffer_slice.get_mapped_range();

        //     let view = ImageView4U8::from_stride_and_slice(
        //         ImageSize::new(
        //             view_port_size.width as usize,
        //             view_port_size.height as usize,
        //         ),
        //         view_port_size.width as usize,
        //         bytemuck::cast_slice(&data[..]),
        //     );

        //     rgba_u8 = ArcImage4U8::make_copy_from(&view);

        //     //vec = buffer_slice.get_mapped_range().to_vec();
        // }
        // println!("distorted image size: {:?}", rgba_u8);
    }
}
