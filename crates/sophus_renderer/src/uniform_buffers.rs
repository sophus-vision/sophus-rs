use sophus_image::ImageSize;
use sophus_lie::Isometry3F64;
use wgpu::{
    util::DeviceExt,
    ShaderStages,
};

use crate::{
    camera::RenderCameraProperties,
    prelude::*,
    types::{
        TranslationAndScaling,
        Zoom2dPod,
    },
    RenderContext,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraFromEntityPosePod {
    camera_from_entity: [[f32; 4]; 4],
}

#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PinholeModelPod {
    pub(crate) camera_image_width: f32, // <= this is NOT the view-port width
    pub(crate) camera_image_height: f32, // <= this is NOT the view-port height
    pub(crate) fx: f32,
    pub(crate) fy: f32,
    pub(crate) px: f32,
    pub(crate) py: f32,
    pub(crate) viewport_scale: f32,
    pub(crate) dummy: f32,
}

#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CameraPropertiesUniform {
    pub(crate) camera_image_width: f32, // <= this is NOT the view-port width
    pub(crate) camera_image_height: f32, // <= this is NOT the view-port height
    pub(crate) near: f32,
    pub(crate) far: f32,
    // pinhole parameter
    pub(crate) fx: f32,
    pub(crate) fy: f32,
    pub(crate) px: f32,
    pub(crate) py: f32,
    //
    pub(crate) alpha: f32, // if alpha == 0, then we use the pinhole model
    pub(crate) beta: f32,
}

pub(crate) struct CameraFromEntityPoseUniform {
    pub(crate) camera_from_entity_buffer: wgpu::Buffer,
}

impl CameraFromEntityPoseUniform {
    pub(crate) fn update_given_camera_and_entity(
        &self,
        queue: &wgpu::Queue,
        scene_from_camera: &Isometry3F64,
        world_from_entity: &Isometry3F64,
    ) {
        let camera_from_entity_mat4x4 = (scene_from_camera.inverse() * world_from_entity).matrix();

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
    pub(crate) render_bind_group: wgpu::BindGroup,
    pub(crate) render_bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) compute_bind_group: wgpu::BindGroup,
    pub(crate) compute_bind_group_layout: wgpu::BindGroupLayout,

    pub(crate) camera_properties_buffer: wgpu::Buffer,
    pub(crate) camera_from_entity_pose_buffer: CameraFromEntityPoseUniform,
    pub(crate) pinhole_buffer: wgpu::Buffer,
    pub(crate) zoom_buffer: wgpu::Buffer,
}

impl VertexShaderUniformBuffers {
    pub(crate) fn make_layout(
        render_context: &RenderContext,
        stage: wgpu::ShaderStages,
    ) -> wgpu::BindGroupLayout {
        render_context
            .wgpu_device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("`{:?}` render layout", stage)),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: stage,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: stage,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: stage,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: stage,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
    }

    pub(crate) fn new(
        render_context: &RenderContext,
        camera_properties: &RenderCameraProperties,
    ) -> Self {
        let device = &render_context.wgpu_device;

        let render_uniform_bind_group_layout =
            Self::make_layout(render_context, ShaderStages::VERTEX);
        let compute_uniform_bind_group_layout =
            Self::make_layout(render_context, ShaderStages::COMPUTE);

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

        let pinhole_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pinhole model buffer"),
            contents: bytemuck::cast_slice(&[PinholeModelPod::default()]),
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

        let render_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render bind group"),
            layout: &render_uniform_bind_group_layout,
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
                    binding: 2,
                    resource: pinhole_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: camera_from_entity_pose_buffer.as_entire_binding(),
                },
            ],
        });
        let compute_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render bind group"),
            layout: &compute_uniform_bind_group_layout,
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
                    binding: 2,
                    resource: pinhole_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: camera_from_entity_pose_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            render_bind_group: render_uniform_bind_group,
            render_bind_group_layout: render_uniform_bind_group_layout,
            compute_bind_group: compute_uniform_bind_group,
            compute_bind_group_layout: compute_uniform_bind_group_layout,
            camera_properties_buffer: camera_properties_uniform_buffer,
            zoom_buffer,
            pinhole_buffer,
            camera_from_entity_pose_buffer: CameraFromEntityPoseUniform {
                camera_from_entity_buffer: camera_from_entity_pose_buffer,
            },
        }
    }

    pub(crate) fn update(
        &self,
        state: &RenderContext,
        zoom_2d: TranslationAndScaling,
        camera_properties: &RenderCameraProperties,
        viewport_size: ImageSize,
    ) {
        let frustum_uniforms = camera_properties.to_uniform();

        state.wgpu_queue.write_buffer(
            &self.camera_properties_buffer,
            0,
            bytemuck::cast_slice(&[frustum_uniforms]),
        );

        let zoom_uniform = Zoom2dPod {
            translation_x: zoom_2d.translation[0] as f32,
            translation_y: zoom_2d.translation[1] as f32,
            scaling_x: zoom_2d.scaling[0] as f32,
            scaling_y: zoom_2d.scaling[1] as f32,
        };

        let intrinsics = camera_properties.intrinsics.clone();
        let pinhole_model = intrinsics.pinhole_model();

        let pinhole = PinholeModelPod {
            camera_image_width: intrinsics.image_size().width as f32,
            camera_image_height: intrinsics.image_size().height as f32,
            fx: pinhole_model.params()[0] as f32,
            fy: pinhole_model.params()[1] as f32,
            px: pinhole_model.params()[2] as f32,
            py: pinhole_model.params()[3] as f32,
            viewport_scale: intrinsics.image_size().width as f32 / viewport_size.width as f32,
            dummy: 1.0,
        };

        state
            .wgpu_queue
            .write_buffer(&self.pinhole_buffer, 0, bytemuck::cast_slice(&[pinhole]));

        state
            .wgpu_queue
            .write_buffer(&self.zoom_buffer, 0, bytemuck::cast_slice(&[zoom_uniform]));
    }
}
