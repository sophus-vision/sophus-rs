use core::num::NonZeroU64;

use eframe::wgpu;
use sophus_image::ImageSize;
use wgpu::util::DeviceExt;

use crate::{
    RenderContext,
    camera::HEMISPHERE,
    types::SOPHUS_RENDER_MULTISAMPLE_COUNT,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FaceSlotPod {
    offset_x: u32,
    _padding: [u32; 3],
}

/// Render targets for the multi-frustum intermediate.
///
/// The faces are rendered one at a time into a shared square target, and copied into an atlas
/// which the distortion pass samples - laid out side by side, face `i` occupying the columns
/// `[i * face_size, (i + 1) * face_size)`.
#[derive(Debug)]
pub(crate) struct FaceTextures {
    /// side length of one face, in pixels
    pub(crate) face_size: u32,
    multisample_view: wgpu::TextureView,
    resolved: wgpu::Texture,
    resolved_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    atlas: wgpu::Texture,
    pub(crate) atlas_view: wgpu::TextureView,
    pub(crate) depth_atlas_view: wgpu::TextureView,
    depth_resolve_pipeline: wgpu::ComputePipeline,
    depth_resolve_bind_group: wgpu::BindGroup,
    /// Distance between the slot uniforms of two faces.
    slot_stride: u32,
}

impl FaceTextures {
    /// The side of one face, for a view which is rendered into a view port of this size: as wide
    /// as the longer side, so that neither axis of the view is sampled below its own resolution.
    pub(crate) fn face_size_for(view_port_size: ImageSize) -> u32 {
        view_port_size.width.max(view_port_size.height) as u32
    }

    pub(crate) fn new(render_context: &RenderContext, face_size: u32) -> Self {
        let device = &render_context.wgpu_device;
        let square = wgpu::Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: 1,
        };

        let multisample = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frustum face multisample target"),
            size: square,
            mip_level_count: 1,
            sample_count: SOPHUS_RENDER_MULTISAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let resolved = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frustum face"),
            size: square,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frustum face depth"),
            size: square,
            mip_level_count: 1,
            sample_count: SOPHUS_RENDER_MULTISAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            // also read by the depth resolve, which folds it into the depth atlas
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frustum face atlas"),
            size: wgpu::Extent3d {
                width: face_size * HEMISPHERE.len() as u32,
                height: face_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let depth_atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frustum face depth atlas"),
            size: wgpu::Extent3d {
                width: face_size * HEMISPHERE.len() as u32,
                height: face_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let depth_atlas_view = depth_atlas.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());

        // One slot per face, written once here - selecting it with a dynamic offset rather than
        // rewriting a uniform between passes, which would not interleave with them.
        let slot_size = core::mem::size_of::<FaceSlotPod>() as u32;
        let alignment = device.limits().min_uniform_buffer_offset_alignment;
        let slot_stride = slot_size.div_ceil(alignment) * alignment;
        let mut slots = vec![0u8; (slot_stride * HEMISPHERE.len() as u32) as usize];
        for face in 0..HEMISPHERE.len() as u32 {
            let pod = FaceSlotPod {
                offset_x: face * face_size,
                _padding: [0; 3],
            };
            let at = (face * slot_stride) as usize;
            slots[at..at + slot_size as usize].copy_from_slice(bytemuck::bytes_of(&pod));
        }
        let slot_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("frustum face slots"),
            contents: &slots,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("face depth resolve layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(slot_size as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        // as `texture_multisampled_2d<f32>`, matching the single-plane path
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
        let depth_resolve_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("face depth resolve"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &slot_buffer,
                        offset: 0,
                        size: NonZeroU64::new(slot_size as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_atlas_view),
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("face depth resolve shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/face_depth.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("face depth resolve pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let depth_resolve_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("face depth resolve pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("resolve_face_depth"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            face_size,
            multisample_view: multisample.create_view(&Default::default()),
            resolved_view: resolved.create_view(&Default::default()),
            resolved,
            depth_view,
            atlas_view: atlas.create_view(&Default::default()),
            atlas,
            depth_atlas_view,
            depth_resolve_pipeline,
            depth_resolve_bind_group,
            slot_stride,
        }
    }

    /// Resolves the depth of the face which was just rendered into its slot of the depth atlas.
    pub(crate) fn resolve_face_depth(
        &self,
        command_encoder: &mut wgpu::CommandEncoder,
        face_index: u32,
    ) {
        const WORKGROUP_SIZE: u32 = 16;
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("face depth resolve"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.depth_resolve_pipeline);
        compute_pass.set_bind_group(
            0,
            &self.depth_resolve_bind_group,
            &[face_index * self.slot_stride],
        );
        compute_pass.dispatch_workgroups(
            self.face_size.div_ceil(WORKGROUP_SIZE),
            self.face_size.div_ceil(WORKGROUP_SIZE),
            1,
        );
    }

    /// Attachments to render one face into.
    pub(crate) fn face_attachments(
        &self,
    ) -> (&wgpu::TextureView, &wgpu::TextureView, &wgpu::TextureView) {
        (
            &self.multisample_view,
            &self.resolved_view,
            &self.depth_view,
        )
    }

    /// Moves the face which was just rendered into its slot of the atlas.
    pub(crate) fn copy_face_into_atlas(
        &self,
        command_encoder: &mut wgpu::CommandEncoder,
        face_index: u32,
    ) {
        command_encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.resolved,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.atlas,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: face_index * self.face_size,
                    y: 0,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.face_size,
                height: self.face_size,
                depth_or_array_layers: 1,
            },
        );
    }
}
