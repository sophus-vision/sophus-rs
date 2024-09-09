use std::sync::Mutex;

use sophus_lie::Isometry3F64;
use sophus_sensor::distortion_table::DistortTable;
use sophus_sensor::DynCamera;
use wgpu::util::DeviceExt;

use crate::offscreen_renderer::ClippingPlanes;
use crate::offscreen_renderer::Zoom2d;
use crate::ViewerRenderState;

#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Frustum {
    pub(crate) camera_image_width: f32, // <= this is NOT the view-port width
    pub(crate) camera_image_height: f32, // <= this is NOT the view-port height
    pub(crate) near: f32,
    pub(crate) far: f32,
    // pinhole parameters for debugging only
    pub(crate) fx: f32,
    pub(crate) fy: f32,
    pub(crate) px: f32,
    pub(crate) py: f32,
}

#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DistortionLut {
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

pub(crate) struct ViewUniform {
    pub(crate) camera_from_entity_buffer: wgpu::Buffer,
}

impl ViewUniform {
    pub(crate) fn update_given_camera_and_entity(
        &self,
        queue: &wgpu::Queue,
        scene_from_camera: &Isometry3F64,
        scene_from_entity: &Isometry3F64,
    ) {
        let camera_from_entity_mat4x4 =
            (scene_from_camera.inverse().group_mul(scene_from_entity)).matrix();

        let mut camera_from_entity_uniform: [[f32; 4]; 4] = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                camera_from_entity_uniform[j][i] = camera_from_entity_mat4x4[(i, j)] as f32;
            }
        }
        queue.write_buffer(
            &self.camera_from_entity_buffer,
            0,
            bytemuck::cast_slice(&[camera_from_entity_uniform]),
        );
    }
}

/// Buffers for rendering a scene
pub struct SceneRenderBuffers {
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) frustum_uniform_buffer: wgpu::Buffer,
    pub(crate) view_uniform: ViewUniform,
    pub(crate) camara_params_buffer: wgpu::Buffer,
    pub(crate) zoom_buffer: wgpu::Buffer,
    pub(crate) dist_texture: wgpu::Texture,
    pub(crate) dist_bind_group: wgpu::BindGroup,
    pub(crate) distortion_lut: Mutex<Option<DistortTable>>,
}

impl SceneRenderBuffers {
    pub(crate) fn new(
        wgpu_render_state: &ViewerRenderState,
        intrinsics: &DynCamera<f64, 1>,
    ) -> Self {
        let device = &wgpu_render_state.device;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene render layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        let frustum_uniforms = Frustum {
            camera_image_width: intrinsics.image_size().width as f32,
            camera_image_height: intrinsics.image_size().height as f32,
            near: ClippingPlanes::DEFAULT_NEAR as f32,
            far: ClippingPlanes::DEFAULT_FAR as f32,
            fx: intrinsics.pinhole_params()[0] as f32,
            fy: intrinsics.pinhole_params()[1] as f32,
            px: intrinsics.pinhole_params()[2] as f32,
            py: intrinsics.pinhole_params()[3] as f32,
        };

        let frustum_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("frustum buffer"),
            contents: bytemuck::cast_slice(&[frustum_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let zoom_uniform = Zoom2d::default();

        let zoom_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("zoom buffer"),
            contents: bytemuck::cast_slice(&[zoom_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let distortion_uniforms = DistortionLut {
            lut_offset_x: 0.0,
            lut_offset_y: 0.0,
            lut_range_x: 0.0,
            lut_range_y: 0.0,
        };

        let distortion_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("distortion uniform buffer"),
            contents: bytemuck::cast_slice(&[distortion_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let view_uniforms = View {
            scene_from_camera: identity,
        };

        let view_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view buffer"),
            contents: bytemuck::cast_slice(&[view_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene render bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frustum_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: zoom_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: distortion_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: view_buffer.as_entire_binding(),
                },
            ],
        });

        let texture_size = wgpu::Extent3d {
            width: intrinsics.image_size().width as u32,
            height: intrinsics.image_size().height as u32,
            depth_or_array_layers: 1,
        };

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
            frustum_uniform_buffer,
            view_uniform: ViewUniform {
                camera_from_entity_buffer: view_buffer,
            },
            camara_params_buffer: distortion_buffer,
            dist_texture,
            dist_bind_group,
            distortion_lut: Mutex::new(None),
            zoom_buffer,
        }
    }
}
