use core::num::NonZeroU64;

use eframe::wgpu;
use sophus_autodiff::linalg::VecF64;
use sophus_image::ImageSize;
use sophus_lie::Isometry3F64;
use wgpu::{
    ShaderStages,
    util::DeviceExt,
};

use crate::{
    RenderContext,
    camera::{
        Frustum,
        RenderCameraProperties,
    },
    prelude::*,
    textures::FaceTextures,
    types::{
        TranslationAndScaling,
        Zoom2dPod,
    },
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraFromEntityPosePod {
    camera_from_entity: [[f32; 4]; 4],
    light_in_entity: [f32; 4],
    /// 1 to draw this entity as edges, whatever the view says
    wireframe: f32,
    _padding: [f32; 3],
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
    /// 1 to tint the image by which frustum each pixel came from, 0 otherwise
    pub(crate) debug_frustum_planes: f32,
    /// Bit `i` is set when face `i` of `HEMISPHERE` was rendered into the atlas this frame. The
    /// slots of the other faces hold whatever was left there, so they must not be sampled.
    pub(crate) rendered_faces: u32,
    /// 1 to draw everything as edges rather than as surfaces, 0 otherwise
    pub(crate) wireframe: f32,
    pub(crate) _padding: [u32; 2],
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

/// Maximum number of scene entities per view whose pose fits into the pose uniform buffer.
pub(crate) const MAX_SCENE_ENTITIES: u32 = 1024;

/// The pose slot the overlays are drawn from.
///
/// The scene's entities take slots from zero upwards, and what they hold is the pose the
/// *intermediate* is rendered from - for a frustum face, that is not the camera. Anything drawn
/// over the finished image needs the camera's own pose, so the last slot is kept for it.
pub(crate) const OVERLAY_POSE_SLOT: u32 = MAX_SCENE_ENTITIES - 1;

pub(crate) struct CameraFromEntityPoseUniform {
    pub(crate) camera_from_entity_buffer: wgpu::Buffer,
    /// Distance between two pose slots - a multiple of the device's uniform offset alignment.
    pub(crate) slot_stride: u64,
}

/// How a frame is to be drawn, as against what is in it.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DrawStyle {
    /// tint each pixel by the frustum it was read back from
    pub(crate) debug_frustum_planes: bool,
    /// draw everything as edges rather than as surfaces
    pub(crate) wireframe: bool,
}

impl CameraFromEntityPoseUniform {
    /// Writes the pose of one entity into its own slot and returns the dynamic offset to bind.
    ///
    /// Note: each entity needs its *own* slot. [wgpu::Queue::write_buffer] does not interleave
    /// with the draw calls of a render pass which is being encoded - all queued writes are
    /// applied before the whole pass runs - so writing every entity to one shared slot would
    /// draw all entities at the pose of the entity which happened to be written last.
    pub(crate) fn update_given_camera_and_entity(
        &self,
        queue: &wgpu::Queue,
        slot: u32,
        world_from_camera: &Isometry3F64,
        world_from_entity: &Isometry3F64,
        light_in_world: VecF64<3>,
        wireframe: bool,
    ) -> u32 {
        let camera_from_entity_mat4x4 = (world_from_camera.inverse() * world_from_entity).matrix();

        let mut camera_from_entity_uniform: [[f32; 4]; 4] = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                camera_from_entity_uniform[j][i] = camera_from_entity_mat4x4[(i, j)] as f32;
            }
        }

        // The light reaches the shader in the entity's own frame, which is the frame its normals
        // are given in - so shading needs no transform at all. It has to be rotated here rather
        // than in the shader, because `world_from_camera` above is the pose the *intermediate* is
        // rendered from: for a frustum face that is not the camera, and a light held in that
        // frame would swing from face to face and seam along their boundaries.
        let light_in_entity = world_from_entity
            .rotation()
            .inverse()
            .transform(light_in_world);

        let offset = slot as u64 * self.slot_stride;
        queue.write_buffer(
            &self.camera_from_entity_buffer,
            offset,
            bytemuck::cast_slice(&[CameraFromEntityPosePod {
                camera_from_entity: camera_from_entity_uniform,
                light_in_entity: [
                    light_in_entity[0] as f32,
                    light_in_entity[1] as f32,
                    light_in_entity[2] as f32,
                    0.0,
                ],
                wireframe: wireframe as u8 as f32,
                _padding: [0.0; 3],
            }]),
        );
        offset as u32
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
                label: Some(&format!("`{stage:?}` render layout")),
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
                    // one pose slot per scene entity, selected via a dynamic offset
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: stage,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: NonZeroU64::new(core::mem::size_of::<
                                CameraFromEntityPosePod,
                            >()
                                as u64),
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

        // The fragment stage reads the light out of the pose uniform, so these are visible to
        // both stages, not to the vertex stage alone.
        let render_uniform_bind_group_layout =
            Self::make_layout(render_context, ShaderStages::VERTEX_FRAGMENT);
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
            light_in_entity: [0.0, 0.0, 1.0, 0.0],
            wireframe: 0.0,
            _padding: [0.0; 3],
        };

        let pose_size = core::mem::size_of::<CameraFromEntityPosePod>() as u64;
        let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let slot_stride = pose_size.div_ceil(alignment) * alignment;

        let camera_from_entity_pose_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera-from-entity pose"),
            size: slot_stride * MAX_SCENE_ENTITIES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // slot 0 is bound by the 2d pixel pipelines, which do not read it - keep it well-defined
        render_context.wgpu_queue.write_buffer(
            &camera_from_entity_pose_buffer,
            0,
            bytemuck::cast_slice(&[camera_from_entity_pose_pod]),
        );

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
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &camera_from_entity_pose_buffer,
                        offset: 0,
                        size: NonZeroU64::new(pose_size),
                    }),
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
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &camera_from_entity_pose_buffer,
                        offset: 0,
                        size: NonZeroU64::new(pose_size),
                    }),
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
                slot_stride,
            },
        }
    }

    /// Uniforms for the multi-frustum intermediate: every face is the same square 90 degree
    /// frustum, only the rotation differs - and that rides in the per-entity pose slots, so it
    /// needs no uniform of its own.
    pub(crate) fn update_for_faces(
        &self,
        context: &RenderContext,
        zoom_2d: TranslationAndScaling,
        camera_properties: &RenderCameraProperties,
        viewport_size: ImageSize,
        faces: &[Frustum],
        style: DrawStyle,
    ) {
        let size = FaceTextures::face_size_for(viewport_size) as f32;

        // Note: the near plane is applied in each face's own z, so the faces clip against a box
        // around the camera rather than against a single plane - which shows *more* near
        // geometry at the periphery than one plane would, not less. Scaling the near distance
        // per face was tried and changed nothing measurable, while making the depth written for
        // a face disagree with the camera's own clipping planes, so it is left alone.
        self.write(
            context,
            zoom_2d,
            camera_properties,
            PinholeModelPod {
                camera_image_width: size,
                camera_image_height: size,
                fx: 0.5 * size,
                fy: 0.5 * size,
                px: 0.5 * (size - 1.0),
                py: 0.5 * (size - 1.0),
                viewport_scale: camera_properties.intrinsics.image_size().width as f32
                    / viewport_size.width as f32,
                debug_frustum_planes: style.debug_frustum_planes as u8 as f32,
                rendered_faces: faces.iter().fold(0, |mask, f| mask | (1 << f.index())),
                wireframe: style.wireframe as u8 as f32,
                _padding: [0; 2],
            },
        );
    }

    pub(crate) fn update(
        &self,
        context: &RenderContext,
        zoom_2d: TranslationAndScaling,
        camera_properties: &RenderCameraProperties,
        viewport_size: ImageSize,
        style: DrawStyle,
        // Note: *not* `zoom_2d`. The plane is fitted to the visible region, which is a different
        // transform - see `Intermediate::choose`.
        pinhole_2d: TranslationAndScaling,
    ) {
        let intrinsics = camera_properties.intrinsics.clone();
        let pinhole_model = intrinsics.pinhole_model();

        self.write(
            context,
            zoom_2d,
            camera_properties,
            PinholeModelPod {
                camera_image_width: intrinsics.image_size().width as f32,
                camera_image_height: intrinsics.image_size().height as f32,
                fx: (pinhole_2d.scaling[0] * pinhole_model.params()[0]) as f32,
                fy: (pinhole_2d.scaling[1] * pinhole_model.params()[1]) as f32,
                px: (pinhole_2d.scaling[0] * pinhole_model.params()[2] + pinhole_2d.translation[0])
                    as f32,
                py: (pinhole_2d.scaling[1] * pinhole_model.params()[3] + pinhole_2d.translation[1])
                    as f32,
                viewport_scale: intrinsics.image_size().width as f32 / viewport_size.width as f32,
                debug_frustum_planes: style.debug_frustum_planes as u8 as f32,
                // the plane path samples no atlas at all
                rendered_faces: 0,
                wireframe: style.wireframe as u8 as f32,
                _padding: [0; 2],
            },
        );
    }

    /// Writes the camera model and the zoom, with the given undistorted intermediate.
    fn write(
        &self,
        context: &RenderContext,
        zoom_2d: TranslationAndScaling,
        camera_properties: &RenderCameraProperties,
        pinhole: PinholeModelPod,
    ) {
        let zoom_uniform = Zoom2dPod {
            translation_x: zoom_2d.translation[0] as f32,
            translation_y: zoom_2d.translation[1] as f32,
            scaling_x: zoom_2d.scaling[0] as f32,
            scaling_y: zoom_2d.scaling[1] as f32,
        };

        // The 2d zoom is a scaling and translation of the image plane: `uv -> s * uv + t`.
        // Both camera models end in an affine step - `u = fx * mx + px` for the distorted model
        // and `u = fx * x + px` for the undistorted pinhole model - hence the zoom can be folded
        // into the intrinsics *exactly*: `fx' = s * fx` and `px' = s * px + t`.
        let mut frustum_uniforms = camera_properties.to_uniform();
        frustum_uniforms.fx *= zoom_uniform.scaling_x;
        frustum_uniforms.fy *= zoom_uniform.scaling_y;
        frustum_uniforms.px =
            zoom_uniform.scaling_x * frustum_uniforms.px + zoom_uniform.translation_x;
        frustum_uniforms.py =
            zoom_uniform.scaling_y * frustum_uniforms.py + zoom_uniform.translation_y;

        context.wgpu_queue.write_buffer(
            &self.camera_properties_buffer,
            0,
            bytemuck::cast_slice(&[frustum_uniforms]),
        );
        context
            .wgpu_queue
            .write_buffer(&self.pinhole_buffer, 0, bytemuck::cast_slice(&[pinhole]));
        context.wgpu_queue.write_buffer(
            &self.zoom_buffer,
            0,
            bytemuck::cast_slice(&[zoom_uniform]),
        );
    }
}

/// The 2d zoom is folded into the intrinsics, see [VertexShaderUniformBuffers::update].
///
/// That is only exact because both camera models end in an affine step, so scaling `f` and `p`
/// is equivalent to scaling and translating the image plane *after* the (non-linear) distortion.
/// If this identity broke, the 3d augmentations would drift away from the background image and
/// from the 2d pixel renderables as soon as the view is zoomed.
#[test]
fn zoom_folds_into_intrinsics() {
    use sophus_autodiff::linalg::VecF64;
    use sophus_sensor::{
        EnhancedUnifiedCameraF64,
        PinholeCameraF64,
    };

    let image_size = ImageSize::new(640, 480);
    let zoom = TranslationAndScaling {
        translation: VecF64::<2>::new(-171.5, -93.25),
        scaling: VecF64::<2>::new(3.5, 3.5),
    };
    let (sx, sy) = (zoom.scaling[0], zoom.scaling[1]);
    let (tx, ty) = (zoom.translation[0], zoom.translation[1]);

    let unified = EnhancedUnifiedCameraF64::new(
        VecF64::<6>::from_array([500.0, 501.0, 319.5, 239.5, 0.6, 1.2]),
        image_size,
    );
    let zoomed_unified = EnhancedUnifiedCameraF64::new(
        VecF64::<6>::from_array([
            sx * 500.0,
            sy * 501.0,
            sx * 319.5 + tx,
            sy * 239.5 + ty,
            0.6,
            1.2,
        ]),
        image_size,
    );

    // `RenderIntrinsics::pinhole_model` uses half the focal length for the unified model
    let pinhole = PinholeCameraF64::new(
        VecF64::<4>::from_array([250.0, 250.5, 319.5, 239.5]),
        image_size,
    );
    let zoomed_pinhole = PinholeCameraF64::new(
        VecF64::<4>::from_array([sx * 250.0, sy * 250.5, sx * 319.5 + tx, sy * 239.5 + ty]),
        image_size,
    );

    for point_in_camera in [
        VecF64::<3>::new(0.0, 0.0, 1.0),
        VecF64::<3>::new(0.3, -0.2, 1.0),
        VecF64::<3>::new(-0.7, 0.5, 2.0),
        VecF64::<3>::new(1.2, 0.9, 3.0),
    ] {
        // A 3d augmentation lands where the zoom puts its unzoomed pixel - i.e. exactly where a
        // 2d renderable anchored at that pixel is drawn, and where the background image is
        // sampled from.
        approx::assert_abs_diff_eq!(
            zoomed_unified.cam_proj(point_in_camera),
            zoom.apply(unified.cam_proj(point_in_camera)),
            epsilon = 1e-9
        );
        approx::assert_abs_diff_eq!(
            zoomed_pinhole.cam_proj(point_in_camera),
            zoom.apply(pinhole.cam_proj(point_in_camera)),
            epsilon = 1e-9
        );
    }

    // The distortion pass runs the models backwards: for a view-port pixel it undistorts with the
    // zoomed model and looks the result up in the scene texture, which was rasterized through the
    // zoomed pinhole model. Both must refer to the same ray.
    for uv_in_image in [
        VecF64::<2>::new(319.5, 239.5),
        VecF64::<2>::new(100.0, 80.0),
        VecF64::<2>::new(600.0, 400.0),
    ] {
        let uv_view_port = zoom.apply(uv_in_image);
        approx::assert_abs_diff_eq!(
            zoomed_unified.undistort(uv_view_port),
            unified.undistort(uv_in_image),
            epsilon = 1e-9
        );
        approx::assert_abs_diff_eq!(
            zoomed_pinhole.cam_proj(zoomed_unified.cam_unproj(uv_view_port)),
            zoom.apply(pinhole.cam_proj(unified.cam_unproj(uv_in_image))),
            epsilon = 1e-9
        );
        // and the background image is sampled at the unzoomed pixel
        approx::assert_abs_diff_eq!(
            zoom.apply_inverse(uv_view_port),
            uv_in_image,
            epsilon = 1e-9
        );
    }
}
