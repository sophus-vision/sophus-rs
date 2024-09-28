use crate::renderer::textures::depth::DepthTextures;
use crate::renderer::textures::rgba::RgbdTexture;
use crate::renderer::uniform_buffers::VertexShaderUniformBuffers;
use crate::RenderContext;
use sophus_image::ImageSize;
use std::sync::Arc;
use wgpu::BindGroup;
use wgpu::BindGroupLayout;

/// Scene line renderer
pub struct DistortionRenderer {
    uniforms: Arc<VertexShaderUniformBuffers>,
    pipeline: wgpu::ComputePipeline,
    pipeline_background: wgpu::ComputePipeline,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group_layout_background: wgpu::BindGroupLayout,
}

impl DistortionRenderer {
    fn make_texture_bind_group_layout(
        render_context: &RenderContext,
        background_image: bool,
    ) -> BindGroupLayout {
        let mut vec = vec![
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
        ];

        if background_image {
            vec.push(wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }

        vec.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        vec.push(wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::R32Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        });

        render_context
            .wgpu_device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("distortion bind group layout with background"),
                entries: &vec,
            })
    }

    fn create_bind_group(
        &self,
        render_context: &RenderContext,
        rgba: &RgbdTexture,
        depth: &DepthTextures,
        background_texture: &Option<wgpu::Texture>,
    ) -> BindGroup {
        let mut vec = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&rgba.rgba_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&rgba.rgba_texture_view_distorted),
            },
        ];

        if let Some(background_texture) = background_texture {
            let view = background_texture.create_view(&wgpu::TextureViewDescriptor::default());
            vec.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&view),
            });
            vec.push(wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(
                    &depth.main_render_ndc_z_texture.ndc_z_texture_view,
                ),
            });
            vec.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(
                    &depth.main_render_ndc_z_texture.z_distorted_texture_view,
                ),
            });

            return render_context
                .wgpu_device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("distortion compute bindgroup with"),
                    layout: &self.texture_bind_group_layout_background,
                    entries: &vec,
                });
        }

        vec.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::TextureView(
                &depth.main_render_ndc_z_texture.ndc_z_texture_view,
            ),
        });
        vec.push(wgpu::BindGroupEntry {
            binding: 4,
            resource: wgpu::BindingResource::TextureView(
                &depth.main_render_ndc_z_texture.z_distorted_texture_view,
            ),
        });

        render_context
            .wgpu_device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("distortion compute bindgroup"),
                layout: &self.texture_bind_group_layout,
                entries: &vec,
            })
    }

    /// Create a new scene line renderer
    pub fn new(render_context: &RenderContext, uniforms: Arc<VertexShaderUniformBuffers>) -> Self {
        let device = &render_context.wgpu_device;
        let (pipeline_background, texture_bind_group_layout_background) = {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("distortion shader with background"),
                source: wgpu::ShaderSource::Wgsl(
                    format!(
                        "{} {}",
                        include_str!("../shaders/utils.wgsl"),
                        include_str!("../shaders/distortion.wgsl")
                    )
                    .into(),
                ),
            });

            let texture_bind_group = Self::make_texture_bind_group_layout(render_context, true);

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("distortion pipeline layout with background"),
                bind_group_layouts: &[&uniforms.compute_bind_group_layout, &texture_bind_group],
                push_constant_ranges: &[],
            });

            (
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("distortion pipeline with background"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: "distort_with_background",
                    compilation_options: Default::default(),
                    cache: None,
                }),
                texture_bind_group,
            )
        };
        let (pipeline, texture_bind_group_layout) = {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("distortion shader with background"),
                source: wgpu::ShaderSource::Wgsl(
                    format!(
                        "{} {}",
                        include_str!("../shaders/utils.wgsl"),
                        include_str!("../shaders/distortion.wgsl")
                    )
                    .into(),
                ),
            });
            let texture_bind_group_layout =
                Self::make_texture_bind_group_layout(render_context, false);

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("distortion pipeline layout"),
                bind_group_layouts: &[
                    &uniforms.compute_bind_group_layout,
                    &texture_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

            (
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("distortion pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: "distort",
                    compilation_options: Default::default(),
                    cache: None,
                }),
                texture_bind_group_layout,
            )
        };

        Self {
            uniforms,
            pipeline,
            pipeline_background,
            texture_bind_group_layout,
            texture_bind_group_layout_background,
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
        match background_texture {
            Some(_) => {
                let bind_group = self.create_bind_group(context, rgba, depth, background_texture);
                {
                    let mut compute_pass =
                        command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                    const WORKGROUP_SIZE: u32 = 16;
                    compute_pass.set_pipeline(&self.pipeline_background);
                    compute_pass.set_bind_group(1, &bind_group, &[]);
                    compute_pass.set_bind_group(0, &self.uniforms.compute_bind_group, &[]);

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
                let bind_group = self.create_bind_group(context, rgba, depth, &None);

                {
                    let mut compute_pass =
                        command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                    const WORKGROUP_SIZE: u32 = 16;
                    compute_pass.set_pipeline(&self.pipeline);
                    compute_pass.set_bind_group(1, &bind_group, &[]);
                    compute_pass.set_bind_group(0, &self.uniforms.compute_bind_group, &[]);

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
    }
}
