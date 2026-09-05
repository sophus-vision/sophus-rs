use eframe::wgpu;
use sophus_autodiff::linalg::{
    MatF64,
    SVec,
    VecF64,
};
use sophus_lie::Isometry3F64;
use wgpu::util::DeviceExt;

use crate::{
    RenderContext,
    prelude::*,
    renderables::{
        Capsule3,
        CapsuleCloud3,
        Cone3,
        ConeCloud3,
        EllipsoidCloud3,
        Planar3,
        PlanarBound,
        PlanarCloud3,
        PlanarPattern,
    },
};

/// The most spheres one view can trace. They are tried one after another for every pixel, so this
/// is a budget on that loop as much as on the buffer.
pub(crate) const MAX_TRACED_ELLIPSOIDS: usize = 256;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct TracedEllipsoidPod {
    /// centre in *camera* coordinates - the pass has no view matrix, and the ray it traces is the
    /// camera's own
    center: [f32; 3],
    /// A radius which encloses the ellipsoid, for a cheap reject before the full intersection.
    bounding_radius: f32,
    /// The rows of the map taking the ellipsoid to the unit sphere - the inverse of the shape.
    /// Padded to four, since a `vec3` in a storage buffer is aligned to sixteen bytes.
    to_unit_sphere: [[f32; 4]; 3],
    color: [f32; 4],
    /// 1 to draw as edges, whatever the view says
    wireframe: f32,
    _padding: [f32; 3],
}

/// The most flat pieces of a plane, capsules and cones one view can trace.
///
/// Room for a scene's worth of poses: a set of coordinate axes is three of each, so a hundred of
/// them - which a bundle adjustment of fifty cameras, drawn twice over as truth and estimate,
/// comes to - is three hundred. These bound the buffer only; the trace loop runs over what is
/// actually there, so the cost of raising them is memory rather than time.
pub(crate) const MAX_TRACED_PLANARS: usize = 512;
pub(crate) const MAX_TRACED_CAPSULES: usize = 512;
pub(crate) const MAX_TRACED_CONES: usize = 512;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct TracedCapsulePod {
    /// one end of the segment, in camera coordinates
    from: [f32; 3],
    radius: f32,
    /// the other end
    to: [f32; 3],
    /// 1 to draw as edges, whatever the view says
    wireframe: f32,
    color: [f32; 4],
    /// 1 to cut the ends flat - a cylinder - rather than rounding them over
    flat_ends: f32,
    _padding: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct TracedConePod {
    /// the tip, in camera coordinates
    apex: [f32; 3],
    /// distance from the apex to the base
    height: f32,
    /// unit vector from the apex towards the base
    axis: [f32; 3],
    /// radius at the base
    radius: f32,
    color: [f32; 4],
    /// 1 to draw as edges, whatever the view says
    wireframe: f32,
    _padding: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct TracedPlanarPod {
    /// centre in camera coordinates
    center: [f32; 3],
    /// 0 unbounded, 1 elliptical, 2 rectangular
    bound: f32,
    /// The rows of the map taking a point of the plane to the coordinates the bound is tested in,
    /// so that an ellipse becomes the unit circle and a rectangle the unit square.
    to_unit_shape: [[f32; 4]; 2],
    /// unit normal in camera coordinates
    normal: [f32; 3],
    /// width of the outline, or of the grid lines, in view-port pixels
    line_width: f32,
    color: [f32; 4],
    /// size of one period of the pattern, in the units of the plane's axes
    pattern_scale: f32,
    /// 0 plain, 1 grid, 2 checker
    pattern: f32,
    /// 1 to draw as edges, whatever the view says
    wireframe: f32,
    _padding: f32,
    /// the other colour of a checkerboard
    pattern_color: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct TracedHeaderPod {
    ellipsoid_count: u32,
    planar_count: u32,
    capsule_count: u32,
    cone_count: u32,
    /// the light, in camera coordinates - the same one the meshes are shaded by
    light_in_camera: [f32; 4],
}

/// One entity's worth of capsules, kept in its own frame until the frame is drawn.
struct CapsuleEntity {
    wireframe: bool,
    capsules: Vec<Capsule3>,
    world_from_entity: Isometry3F64,
}

/// One entity's worth of cones, kept in its own frame until the frame is drawn.
struct ConeEntity {
    wireframe: bool,
    cones: Vec<Cone3>,
    world_from_entity: Isometry3F64,
}

/// One entity's worth of flat pieces, kept in its own frame until the frame is drawn.
struct PlanarEntity {
    wireframe: bool,
    planars: Vec<Planar3>,
    world_from_entity: Isometry3F64,
}

/// One entity's worth of ellipsoids, kept in its own frame until the frame is drawn.
struct EllipsoidEntity {
    wireframe: bool,
    /// centre, the map taking the unit sphere to the ellipsoid, and the colour
    ellipsoids: Vec<(VecF64<3>, MatF64<3, 3>, [f32; 4])>,
    world_from_entity: Isometry3F64,
}

/// The primitives the distortion pass traces, and the buffer it reads them from.
///
/// These are not rasterized: the distortion pass already computes the exact ray through every
/// pixel of the distorted image, so a sphere is intersected with that ray directly. It is
/// therefore exact under the real camera model - no intermediate, no resampling, and no upper
/// limit on the field of view.
pub struct TracedPrimitives {
    entities: BTreeMap<String, EllipsoidEntity>,
    planar_entities: BTreeMap<String, PlanarEntity>,
    capsule_entities: BTreeMap<String, CapsuleEntity>,
    cone_entities: BTreeMap<String, ConeEntity>,
    buffer: wgpu::Buffer,
    /// The bind group holds one buffer which is never reallocated, so it is built once.
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
}

impl TracedPrimitives {
    pub(crate) fn new(render_context: &RenderContext) -> Self {
        let device = &render_context.wgpu_device;

        let header_size = core::mem::size_of::<TracedHeaderPod>();
        let contents = vec![
            0u8;
            header_size
                + MAX_TRACED_ELLIPSOIDS * core::mem::size_of::<TracedEllipsoidPod>()
                + MAX_TRACED_PLANARS * core::mem::size_of::<TracedPlanarPod>()
                + MAX_TRACED_CAPSULES * core::mem::size_of::<TracedCapsulePod>()
                + MAX_TRACED_CONES * core::mem::size_of::<TracedConePod>()
        ];
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("traced primitives"),
            contents: &contents,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("traced primitives layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("traced primitives"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        TracedPrimitives {
            entities: BTreeMap::new(),
            planar_entities: BTreeMap::new(),
            capsule_entities: BTreeMap::new(),
            cone_entities: BTreeMap::new(),
            buffer,
            bind_group,
            bind_group_layout,
        }
    }

    pub(crate) fn insert(&mut self, ellipsoids: &EllipsoidCloud3) {
        self.entities.insert(
            ellipsoids.name.clone(),
            EllipsoidEntity {
                ellipsoids: ellipsoids
                    .ellipsoids
                    .iter()
                    .map(|e| {
                        (
                            VecF64::<3>::new(
                                e.center[0] as f64,
                                e.center[1] as f64,
                                e.center[2] as f64,
                            ),
                            e.shape,
                            [e.color.r, e.color.g, e.color.b, e.color.a],
                        )
                    })
                    .collect(),
                world_from_entity: ellipsoids.world_from_entity,
                wireframe: ellipsoids.wireframe,
            },
        );
    }

    pub(crate) fn insert_planars(&mut self, planars: &PlanarCloud3) {
        self.planar_entities.insert(
            planars.name.clone(),
            PlanarEntity {
                planars: planars.planars.clone(),
                world_from_entity: planars.world_from_entity,
                wireframe: planars.wireframe,
            },
        );
    }

    pub(crate) fn insert_capsules(&mut self, capsules: &CapsuleCloud3) {
        self.capsule_entities.insert(
            capsules.name.clone(),
            CapsuleEntity {
                capsules: capsules.capsules.clone(),
                world_from_entity: capsules.world_from_entity,
                wireframe: capsules.wireframe,
            },
        );
    }

    pub(crate) fn insert_cones(&mut self, cones: &ConeCloud3) {
        self.cone_entities.insert(
            cones.name.clone(),
            ConeEntity {
                cones: cones.cones.clone(),
                world_from_entity: cones.world_from_entity,
                wireframe: cones.wireframe,
            },
        );
    }

    pub(crate) fn clear(&mut self) {
        self.entities.clear();
        self.planar_entities.clear();
        self.capsule_entities.clear();
        self.cone_entities.clear();
    }

    /// Hands the spheres to the gpu in camera coordinates.
    ///
    /// They are transformed here, on the cpu, rather than by a matrix in the shader: the buffer is
    /// written once a frame either way - as the camera properties are - and this leaves the shader
    /// with the camera's own ray, its origin, and nothing to transform.
    pub(crate) fn update(
        &self,
        render_context: &RenderContext,
        world_from_camera: &Isometry3F64,
        light_in_camera: VecF64<3>,
    ) {
        let camera_from_world = world_from_camera.inverse();
        let mut pods = Vec::new();
        for entity in self.entities.values() {
            let camera_from_entity = camera_from_world * entity.world_from_entity;
            for (center, shape, color) in &entity.ellipsoids {
                if pods.len() >= MAX_TRACED_ELLIPSOIDS {
                    log::warn!(
                        "more than {MAX_TRACED_ELLIPSOIDS} traced ellipsoids - skipping the rest"
                    );
                    break;
                }
                // The entity's rotation carries the shape into the camera frame; its translation
                // moves the centre. A shape which cannot be inverted is a flat ellipsoid, which
                // has no interior to hit.
                let shape_in_camera = camera_from_entity.rotation().matrix() * shape;
                let Some(to_unit_sphere) = shape_in_camera.try_inverse() else {
                    continue;
                };
                let center = camera_from_entity.transform(*center);
                pods.push(TracedEllipsoidPod {
                    center: [center[0] as f32, center[1] as f32, center[2] as f32],
                    // conservative: the Frobenius norm is never below the largest semi-axis
                    bounding_radius: shape_in_camera.norm() as f32,
                    to_unit_sphere: [
                        [
                            to_unit_sphere[(0, 0)] as f32,
                            to_unit_sphere[(0, 1)] as f32,
                            to_unit_sphere[(0, 2)] as f32,
                            0.0,
                        ],
                        [
                            to_unit_sphere[(1, 0)] as f32,
                            to_unit_sphere[(1, 1)] as f32,
                            to_unit_sphere[(1, 2)] as f32,
                            0.0,
                        ],
                        [
                            to_unit_sphere[(2, 0)] as f32,
                            to_unit_sphere[(2, 1)] as f32,
                            to_unit_sphere[(2, 2)] as f32,
                            0.0,
                        ],
                    ],
                    color: *color,
                    wireframe: entity.wireframe as u8 as f32,
                    _padding: [0.0; 3],
                });
            }
        }

        // --- flat pieces of a plane ---
        let mut planar_pods: Vec<TracedPlanarPod> = Vec::new();
        for entity in self.planar_entities.values() {
            let camera_from_entity = camera_from_world * entity.world_from_entity;
            for planar in &entity.planars {
                if planar_pods.len() >= MAX_TRACED_PLANARS {
                    log::warn!("more than {MAX_TRACED_PLANARS} traced planars - skipping the rest");
                    break;
                }
                let axis = |a: &SVec<f32, 3>| {
                    camera_from_entity.rotation().transform(VecF64::<3>::new(
                        a[0] as f64,
                        a[1] as f64,
                        a[2] as f64,
                    ))
                };
                let u = axis(&planar.axes[0]);
                let v = axis(&planar.axes[1]);
                let normal = u.cross(&v);
                if normal.norm() < 1e-12 {
                    // the two axes are parallel: no plane to speak of
                    continue;
                }
                let normal = normal.normalize();

                // The rows taking a point of the plane to the coordinates the bound is tested in,
                // which is the pseudo-inverse of `[u v]` - the axes need not be orthogonal.
                let axes = MatF64::<3, 2>::from_columns(&[u, v]);
                let gram = axes.transpose() * axes;
                let Some(gram_inverse) = gram.try_inverse() else {
                    continue;
                };
                let to_unit_shape = gram_inverse * axes.transpose();

                let center = camera_from_entity.transform(VecF64::<3>::new(
                    planar.center[0] as f64,
                    planar.center[1] as f64,
                    planar.center[2] as f64,
                ));
                planar_pods.push(TracedPlanarPod {
                    center: [center[0] as f32, center[1] as f32, center[2] as f32],
                    bound: match planar.bound {
                        PlanarBound::Infinite => 0.0,
                        PlanarBound::Ellipse => 1.0,
                        PlanarBound::Rectangle => 2.0,
                    },
                    to_unit_shape: [
                        [
                            to_unit_shape[(0, 0)] as f32,
                            to_unit_shape[(0, 1)] as f32,
                            to_unit_shape[(0, 2)] as f32,
                            0.0,
                        ],
                        [
                            to_unit_shape[(1, 0)] as f32,
                            to_unit_shape[(1, 1)] as f32,
                            to_unit_shape[(1, 2)] as f32,
                            0.0,
                        ],
                    ],
                    normal: [normal[0] as f32, normal[1] as f32, normal[2] as f32],
                    line_width: planar.line_width,
                    color: [
                        planar.color.r,
                        planar.color.g,
                        planar.color.b,
                        planar.color.a,
                    ],
                    pattern_scale: planar.pattern_scale,
                    pattern: match planar.pattern {
                        PlanarPattern::Plain => 0.0,
                        PlanarPattern::Grid => 1.0,
                        PlanarPattern::Checker => 2.0,
                    },
                    wireframe: entity.wireframe as u8 as f32,
                    _padding: 0.0,
                    pattern_color: [
                        planar.pattern_color.r,
                        planar.pattern_color.g,
                        planar.pattern_color.b,
                        planar.pattern_color.a,
                    ],
                });
            }
        }

        // --- capsules ---
        let mut capsule_pods: Vec<TracedCapsulePod> = Vec::new();
        for entity in self.capsule_entities.values() {
            let camera_from_entity = camera_from_world * entity.world_from_entity;
            for capsule in &entity.capsules {
                if capsule_pods.len() >= MAX_TRACED_CAPSULES {
                    log::warn!(
                        "more than {MAX_TRACED_CAPSULES} traced capsules - skipping the rest"
                    );
                    break;
                }
                let end = |p: &SVec<f32, 3>| {
                    let p = camera_from_entity.transform(VecF64::<3>::new(
                        p[0] as f64,
                        p[1] as f64,
                        p[2] as f64,
                    ));
                    [p[0] as f32, p[1] as f32, p[2] as f32]
                };
                capsule_pods.push(TracedCapsulePod {
                    from: end(&capsule.from),
                    radius: capsule.radius,
                    to: end(&capsule.to),
                    wireframe: entity.wireframe as u8 as f32,
                    color: [
                        capsule.color.r,
                        capsule.color.g,
                        capsule.color.b,
                        capsule.color.a,
                    ],
                    flat_ends: capsule.flat_ends as u8 as f32,
                    _padding: [0.0; 3],
                });
            }
        }

        // --- cones ---
        let mut cone_pods: Vec<TracedConePod> = Vec::new();
        for entity in self.cone_entities.values() {
            let camera_from_entity = camera_from_world * entity.world_from_entity;
            for cone in &entity.cones {
                if cone_pods.len() >= MAX_TRACED_CONES {
                    log::warn!("more than {MAX_TRACED_CONES} traced cones - skipping the rest");
                    break;
                }
                let axis = camera_from_entity.rotation().transform(VecF64::<3>::new(
                    cone.axis[0] as f64,
                    cone.axis[1] as f64,
                    cone.axis[2] as f64,
                ));
                let height = axis.norm();
                if height < 1e-9 {
                    continue;
                }
                let axis = axis / height;
                let apex = camera_from_entity.transform(VecF64::<3>::new(
                    cone.apex[0] as f64,
                    cone.apex[1] as f64,
                    cone.apex[2] as f64,
                ));
                cone_pods.push(TracedConePod {
                    apex: [apex[0] as f32, apex[1] as f32, apex[2] as f32],
                    height: height as f32,
                    axis: [axis[0] as f32, axis[1] as f32, axis[2] as f32],
                    radius: cone.radius,
                    color: [cone.color.r, cone.color.g, cone.color.b, cone.color.a],
                    wireframe: entity.wireframe as u8 as f32,
                    _padding: [0.0; 3],
                });
            }
        }

        let header = TracedHeaderPod {
            ellipsoid_count: pods.len() as u32,
            planar_count: planar_pods.len() as u32,
            capsule_count: capsule_pods.len() as u32,
            cone_count: cone_pods.len() as u32,
            light_in_camera: [
                light_in_camera[0] as f32,
                light_in_camera[1] as f32,
                light_in_camera[2] as f32,
                0.0,
            ],
        };
        render_context
            .wgpu_queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[header]));
        if !pods.is_empty() {
            render_context.wgpu_queue.write_buffer(
                &self.buffer,
                core::mem::size_of::<TracedHeaderPod>() as u64,
                bytemuck::cast_slice(&pods),
            );
        }
        let planar_offset = core::mem::size_of::<TracedHeaderPod>()
            + MAX_TRACED_ELLIPSOIDS * core::mem::size_of::<TracedEllipsoidPod>();
        if !planar_pods.is_empty() {
            render_context.wgpu_queue.write_buffer(
                &self.buffer,
                planar_offset as u64,
                bytemuck::cast_slice(&planar_pods),
            );
        }
        let capsule_offset =
            planar_offset + MAX_TRACED_PLANARS * core::mem::size_of::<TracedPlanarPod>();
        if !capsule_pods.is_empty() {
            render_context.wgpu_queue.write_buffer(
                &self.buffer,
                capsule_offset as u64,
                bytemuck::cast_slice(&capsule_pods),
            );
        }
        if !cone_pods.is_empty() {
            let offset =
                capsule_offset + MAX_TRACED_CAPSULES * core::mem::size_of::<TracedCapsulePod>();
            render_context.wgpu_queue.write_buffer(
                &self.buffer,
                offset as u64,
                bytemuck::cast_slice(&cone_pods),
            );
        }
    }
}
