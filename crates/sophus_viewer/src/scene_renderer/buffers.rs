use std::sync::Mutex;

use sophus_sensor::distortion_table::DistortTable;
use sophus_sensor::DynCamera;
use wgpu::util::DeviceExt;

use crate::ViewerRenderState;

#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Transforms {
    width: f32,
    height: f32,
    near: f32,
    far: f32,
    fx: f32,
    fy: f32,
    px: f32,
    py: f32,
}

#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Lut {
    lut_offset_x: f32,
    lut_offset_y: f32,
    lut_range_x: f32,
    lut_range_y: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct View {
    scene_from_camera: [[f32; 4]; 4],
}

/// Buffers for rendering a scene
pub struct SceneRenderBuffers {
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) _uniform_buffer: wgpu::Buffer,
    pub(crate) view_uniform_buffer: wgpu::Buffer,
    pub(crate) lut_buffer: wgpu::Buffer,
    pub(crate) dist_texture: wgpu::Texture,
    pub(crate) dist_bind_group: wgpu::BindGroup,
    pub(crate) distortion_lut: Mutex<Option<DistortTable>>,
    pub(crate) background_texture: wgpu::Texture,
    pub(crate) background_bind_group: wgpu::BindGroup,
}

impl SceneRenderBuffers {
    pub(crate) fn new(
        wgpu_render_state: &ViewerRenderState,
        intrinsics: &DynCamera<f64, 1>,
    ) -> Self {
        let device = &wgpu_render_state.device;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("custom3d"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let identity = [
            [1.0, 0.0, 0.0, 0.0], // 1.
            [0.0, 1.0, 0.0, 0.0], // 2.
            [0.0, 0.0, 1.0, 0.0], // 3.
            [0.0, 0.0, 0.0, 1.0], // 4.
        ];

        let transform_uniforms = Transforms {
            width: intrinsics.image_size().width as f32,
            height: intrinsics.image_size().height as f32,
            near: 0.1,
            far: 1000.0,
            fx: intrinsics.pinhole_params()[0] as f32,
            fy: intrinsics.pinhole_params()[1] as f32,
            px: intrinsics.pinhole_params()[2] as f32,
            py: intrinsics.pinhole_params()[3] as f32,
        };

        let transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[transform_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let lut_uniforms = Lut {
            lut_offset_x: 0.0,
            lut_offset_y: 0.0,
            lut_range_x: 0.0,
            lut_range_y: 0.0,
        };

        let lut_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lut Buffer"),
            contents: bytemuck::cast_slice(&[lut_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let view_uniforms = View {
            scene_from_camera: identity,
        };

        let view_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer2"),
            contents: bytemuck::cast_slice(&[view_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("custom3d"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: transform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: view_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lut_buffer.as_entire_binding(),
                },
            ],
        });

        let texture_size = wgpu::Extent3d {
            width: intrinsics.image_size().width as u32,
            height: intrinsics.image_size().height as u32,
            depth_or_array_layers: 1,
        };

        let background_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("background_texture_bind_group_layout"),
            });

        let background_texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let background_texture_size = wgpu::Extent3d {
            width: intrinsics.image_size().width as u32,
            height: intrinsics.image_size().height as u32,
            depth_or_array_layers: 1,
        };
        let background_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: background_texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("dist_texture"),
            view_formats: &[],
        });

        let background_texture_view =
            background_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let background_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &background_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&background_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&background_texture_sampler),
                },
            ],
            label: Some("background_bind_group"),
        });

        let dist_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("dist_texture"),
            view_formats: &[],
        });

        let dist_texture_view = dist_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let dist_texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    },
                    count: None,
                }],
                label: Some("dist_texture_bind_group_layout"),
            });

        let dist_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &dist_texture_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&dist_texture_view),
            }],
            label: Some("diffuse_bind_group"),
        });

        Self {
            bind_group,
            _uniform_buffer: transform_buffer,
            view_uniform_buffer: view_buffer,
            lut_buffer,
            dist_texture,
            dist_bind_group,
            distortion_lut: Mutex::new(None),
            background_bind_group,
            background_texture,
        }
    }
}
