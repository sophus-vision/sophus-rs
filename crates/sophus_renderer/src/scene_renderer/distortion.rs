use eframe::wgpu;
use sophus_image::ImageSize;
use wgpu::{
    BindGroup,
    BindGroupLayout,
};

use crate::{
    RenderContext,
    prelude::*,
    textures::{
        DepthTextures,
        RgbdTexture,
    },
    uniform_buffers::VertexShaderUniformBuffers,
};

/// Scene line renderer
pub struct DistortionRenderer {
    uniforms: Arc<VertexShaderUniformBuffers>,
    pipeline: wgpu::ComputePipeline,
    pipeline_background: wgpu::ComputePipeline,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group_layout_background: wgpu::BindGroupLayout,
    /// The bind group only changes when the textures do, so it is kept rather than rebuilt for
    /// every frame - see [Self::invalidate_bind_group].
    cached_bind_group: Option<BindGroup>,
    face_pipeline: wgpu::ComputePipeline,
    face_pipeline_background: wgpu::ComputePipeline,
    face_bind_group_layout: wgpu::BindGroupLayout,
    face_bind_group_layout_background: wgpu::BindGroupLayout,
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
                multisampled: true,
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

    /// Layout for the multi-frustum variants: they read the face atlas instead of a single
    /// plane, and have no multisampled depth to resolve.
    fn make_face_bind_group_layout(
        render_context: &RenderContext,
        background_image: bool,
    ) -> BindGroupLayout {
        let mut vec = vec![
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
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
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
        render_context
            .wgpu_device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("distortion face bind group layout"),
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
                resource: wgpu::BindingResource::TextureView(&rgba.resolved_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&rgba.final_texture_view),
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
                    &depth.main_render_ndc_z_texture.multisample_texture_view,
                ),
            });
            vec.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(
                    &depth.main_render_ndc_z_texture.final_texture_view,
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
                &depth.main_render_ndc_z_texture.multisample_texture_view,
            ),
        });
        vec.push(wgpu::BindGroupEntry {
            binding: 4,
            resource: wgpu::BindingResource::TextureView(
                &depth.main_render_ndc_z_texture.final_texture_view,
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

        // one module, two entry points
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("distortion shader"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{} {}",
                    include_str!("../shaders/utils.wgsl"),
                    include_str!("../shaders/distortion.wgsl")
                )
                .into(),
            ),
        });

        let compute_pipeline = |entry_point: &str, background: bool| {
            let texture_bind_group_layout =
                Self::make_texture_bind_group_layout(render_context, background);
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("`{entry_point}` pipeline layout")),
                bind_group_layouts: &[
                    &uniforms.compute_bind_group_layout,
                    &texture_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("`{entry_point}` pipeline")),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });
            (pipeline, texture_bind_group_layout)
        };

        let (pipeline_background, texture_bind_group_layout_background) =
            compute_pipeline("distort_with_background", true);
        let (pipeline, texture_bind_group_layout) = compute_pipeline("distort", false);

        let face_pipeline = |entry_point: &str, background: bool| {
            let layout = Self::make_face_bind_group_layout(render_context, background);
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("`{entry_point}` pipeline layout")),
                bind_group_layouts: &[&uniforms.compute_bind_group_layout, &layout],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("`{entry_point}` pipeline")),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });
            (pipeline, layout)
        };
        let (face_pipeline_background, face_bind_group_layout_background) =
            face_pipeline("distort_faces_with_background", true);
        let (face_pipeline, face_bind_group_layout) = face_pipeline("distort_faces", false);

        Self {
            face_pipeline,
            face_pipeline_background,
            face_bind_group_layout,
            face_bind_group_layout_background,
            uniforms,
            pipeline,
            pipeline_background,
            texture_bind_group_layout,
            texture_bind_group_layout_background,
            cached_bind_group: None,
        }
    }

    /// Drops the cached bind group; call whenever the textures it refers to are replaced.
    pub(crate) fn invalidate_bind_group(&mut self) {
        self.cached_bind_group = None;
    }

    /// Composites the scene out of the multi-frustum intermediate.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_faces(
        &self,
        context: &RenderContext,
        command_encoder: &mut wgpu::CommandEncoder,
        rgba: &RgbdTexture,
        depth: &DepthTextures,
        faces: &crate::textures::FaceTextures,
        background_texture: &Option<wgpu::Texture>,
        view_port_size: &ImageSize,
    ) {
        const WORKGROUP_SIZE: u32 = 16;

        let background_view = background_texture
            .as_ref()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));

        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&rgba.final_texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(
                    &depth.main_render_ndc_z_texture.final_texture_view,
                ),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&faces.atlas_view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&faces.depth_atlas_view),
            },
        ];
        if let Some(view) = background_view.as_ref() {
            entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        let (layout, pipeline) = match background_texture {
            Some(_) => (
                &self.face_bind_group_layout_background,
                &self.face_pipeline_background,
            ),
            None => (&self.face_bind_group_layout, &self.face_pipeline),
        };
        let bind_group = context
            .wgpu_device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("distortion face bind group"),
                layout,
                entries: &entries,
            });

        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("distortion (faces)"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &self.uniforms.compute_bind_group, &[0]);
        compute_pass.set_bind_group(1, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            (view_port_size.width as u32).div_ceil(WORKGROUP_SIZE),
            (view_port_size.height as u32).div_ceil(WORKGROUP_SIZE),
            1,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &mut self,
        context: &RenderContext,
        command_encoder: &mut wgpu::CommandEncoder,
        rgba: &RgbdTexture,
        depth: &DepthTextures,
        background_texture: &Option<wgpu::Texture>,
        view_port_size: &ImageSize,
    ) {
        const WORKGROUP_SIZE: u32 = 16;

        let bind_group = match self.cached_bind_group.take() {
            Some(bind_group) => bind_group,
            None => self.create_bind_group(context, rgba, depth, background_texture),
        };
        let pipeline = match background_texture {
            Some(_) => &self.pipeline_background,
            None => &self.pipeline,
        };

        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("distortion"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &self.uniforms.compute_bind_group, &[0]);
        compute_pass.set_bind_group(1, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            (view_port_size.width as u32).div_ceil(WORKGROUP_SIZE),
            (view_port_size.height as u32).div_ceil(WORKGROUP_SIZE),
            1,
        );
        drop(compute_pass);

        self.cached_bind_group = Some(bind_group);
    }
}
