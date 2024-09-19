use sophus_lie::Isometry3F64;
use wgpu::util::DeviceExt;

use crate::renderer::camera::properties::RenderCameraProperties;
use crate::renderer::types::Zoom2dPod;
use crate::RenderContext;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraFromEntityPosePod {
    camera_from_entity: [[f32; 4]; 4],
}

pub(crate) struct CameraFromEntityPoseUniform {
    pub(crate) camera_from_entity_buffer: wgpu::Buffer,
}

impl CameraFromEntityPoseUniform {
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
pub struct VertexShaderUniformBuffers {
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,

    pub(crate) camera_properties_buffer: wgpu::Buffer,
    pub(crate) camera_from_entity_pose_buffer: CameraFromEntityPoseUniform,
    pub(crate) zoom_buffer: wgpu::Buffer,
}

impl VertexShaderUniformBuffers {
    pub(crate) fn new(
        wgpu_render_state: &RenderContext,
        camera_properties: &RenderCameraProperties,
    ) -> Self {
        let device = &wgpu_render_state.wgpu_device;

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene render layout"),
                entries: &[
                    // todo: Add pinhole model uniform
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

        let camera_properties_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("frustum buffer"),
                contents: bytemuck::cast_slice(&[camera_properties.to_uniform()]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let zoom_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("zoom buffer"),
            contents: bytemuck::cast_slice(&[Zoom2dPod::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_from_entity_pose_pod = CameraFromEntityPosePod {
            camera_from_entity: identity,
        };

        let camera_from_entity_pose_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("camera-from-entity pose"),
                contents: bytemuck::cast_slice(&[camera_from_entity_pose_pod]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene render bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_properties_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: zoom_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: camera_from_entity_pose_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            bind_group: uniform_bind_group,
            camera_properties_buffer: camera_properties_uniform_buffer,
            camera_from_entity_pose_buffer: CameraFromEntityPoseUniform {
                camera_from_entity_buffer: camera_from_entity_pose_buffer,
            },
            zoom_buffer,
            bind_group_layout: uniform_bind_group_layout,
        }
    }
}
