mod axes;

pub use axes::*;
use sophus_autodiff::linalg::{
    MatF64,
    SVec,
};
use sophus_image::ArcImage4U8;
use sophus_lie::{
    Isometry3,
    Isometry3F64,
};

use crate::{
    prelude::*,
    renderables::{
        color::Color,
        pixel_renderable::HasToVec2F32,
    },
};

/// scene renderable
#[derive(Clone, Debug)]
pub enum SceneRenderable {
    /// 3D line segments
    Line(LineSegments3),
    /// 3D points
    Point(PointCloud3),
    /// 3D mesh
    Mesh3(TriangleMesh3),
    /// 3D texture-mapped mesh
    TexturedMesh3(TexturedTriangleMesh3),
    /// 3D ellipsoids, traced rather than rasterized
    Ellipsoid(EllipsoidCloud3),
    /// Flat pieces of a plane - disks, ellipses, rectangles, or the whole plane - traced rather
    /// than rasterized
    Planar(PlanarCloud3),
    /// Capsules - a segment with a radius - traced rather than rasterized
    Capsule(CapsuleCloud3),
    /// Cones, traced rather than rasterized
    Cone(ConeCloud3),
}

impl SceneRenderable {
    /// Draw this entity as edges rather than as surfaces.
    ///
    /// Per entity, so that one thing can be opened up while the rest of the scene stays solid -
    /// the view's own wireframe toggle turns everything on at once, and either of the two is
    /// enough. Lines and points have no surface to open up, so they are unaffected.
    pub fn wireframe(mut self, wireframe: bool) -> Self {
        match &mut self {
            SceneRenderable::Mesh3(mesh) => mesh.wireframe = wireframe,
            SceneRenderable::TexturedMesh3(mesh) => mesh.wireframe = wireframe,
            SceneRenderable::Ellipsoid(ellipsoids) => ellipsoids.wireframe = wireframe,
            SceneRenderable::Planar(planars) => planars.wireframe = wireframe,
            SceneRenderable::Capsule(capsules) => capsules.wireframe = wireframe,
            SceneRenderable::Cone(cones) => cones.wireframe = wireframe,
            SceneRenderable::Line(_) | SceneRenderable::Point(_) => {}
        }
        self
    }

    /// Get scene from entity
    pub fn world_from_entity(&self) -> Isometry3F64 {
        match self {
            SceneRenderable::Line(lines) => lines.world_from_entity,
            SceneRenderable::Point(points) => points.world_from_entity,
            SceneRenderable::Ellipsoid(ellipsoids) => ellipsoids.world_from_entity,
            SceneRenderable::Planar(planars) => planars.world_from_entity,
            SceneRenderable::Capsule(capsules) => capsules.world_from_entity,
            SceneRenderable::Cone(cones) => cones.world_from_entity,
            SceneRenderable::Mesh3(mesh) => mesh.world_from_entity,
            SceneRenderable::TexturedMesh3(mesh) => mesh.world_from_entity,
        }
    }
}

/// creates a named line segment at a given pose
pub fn named_line3_at(
    name: impl ToString,
    line_segments: Vec<LineSegment3>,
    world_from_entity: Isometry3F64,
) -> SceneRenderable {
    let lines = LineSegments3 {
        name: name.to_string(),
        segments: line_segments,
        world_from_entity,
    };

    SceneRenderable::Line(lines)
}

/// creates a named line segment
pub fn named_line3(name: impl ToString, line_segments: Vec<LineSegment3>) -> SceneRenderable {
    named_line3_at(name, line_segments, Isometry3::identity())
}

/// creates a named point cloud at a given pose
pub fn named_point3_at(
    name: impl ToString,
    points: Vec<Point3>,
    world_from_entity: Isometry3F64,
) -> SceneRenderable {
    let points = PointCloud3 {
        name: name.to_string(),
        points,
        world_from_entity,
    };

    SceneRenderable::Point(points)
}

/// creates a named point cloud
pub fn named_point3(name: impl ToString, points: Vec<Point3>) -> SceneRenderable {
    named_point3_at(name, points, Isometry3::identity())
}

/// creates a named mesh at a given pose
pub fn named_mesh3_at(
    name: impl ToString,
    mesh: TriangleMesh3,
    world_from_entity: Isometry3F64,
) -> SceneRenderable {
    let mesh = TriangleMesh3 {
        name: name.to_string(),
        triangles: mesh.triangles,
        world_from_entity,
        wireframe: false,
    };

    SceneRenderable::Mesh3(mesh)
}

/// creates a named mesh
pub fn named_mesh3(name: impl ToString, mesh: TriangleMesh3) -> SceneRenderable {
    named_mesh3_at(name, mesh, Isometry3::identity())
}

/// The three principal rings of each ellipsoid: outlines of where it meets its own xy, yz and zx
/// planes, in lines of `line_width` view-port pixels.
///
/// A field of solid ellipsoids is an opaque mass which hides the scene and each other, which is
/// no way to look at a set of covariances. Rings show the same shape and leave everything behind
/// them visible. The alternative is a shell - the same ellipsoid with an alpha below one - which
/// keeps the surface but tints whatever is behind it.
pub fn make_ellipsoid_rings3_at(
    name: impl ToString,
    ellipsoids: Vec<Ellipsoid3>,
    line_width: f32,
    world_from_entity: Isometry3F64,
) -> SceneRenderable {
    let mut rings = vec![];
    for ellipsoid in ellipsoids {
        // the semi-axes of the ellipsoid, which are the columns of the map from the unit sphere
        let axis = |i: usize| {
            let column = ellipsoid.shape.column(i);
            SVec::<f32, 3>::new(column[0] as f32, column[1] as f32, column[2] as f32)
        };
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            rings.push(
                Planar3::ellipse(ellipsoid.center, [axis(i), axis(j)], ellipsoid.color)
                    .outlined(line_width),
            );
        }
    }
    make_planar3_at(name, rings, world_from_entity)
}

/// The three principal rings of each ellipsoid, at the origin of the scene.
pub fn make_ellipsoid_rings3(
    name: impl ToString,
    ellipsoids: Vec<Ellipsoid3>,
    line_width: f32,
) -> SceneRenderable {
    make_ellipsoid_rings3_at(name, ellipsoids, line_width, Isometry3F64::identity())
}

/// Ellipsoids at a pose of their own.
pub fn make_ellipsoid3_at(
    name: impl ToString,
    ellipsoids: Vec<Ellipsoid3>,
    world_from_entity: Isometry3F64,
) -> SceneRenderable {
    SceneRenderable::Ellipsoid(EllipsoidCloud3 {
        name: name.to_string(),
        ellipsoids,
        world_from_entity,
        wireframe: false,
    })
}

/// Ellipsoids, at the origin of the scene.
pub fn make_ellipsoid3(name: impl ToString, ellipsoids: Vec<Ellipsoid3>) -> SceneRenderable {
    make_ellipsoid3_at(name, ellipsoids, Isometry3F64::identity())
}

/// Spheres at a pose of their own, each with a radius in scene units.
///
/// A sphere is an ellipsoid whose shape is a scaled identity, so this is a constructor rather
/// than a primitive of its own - one intersection routine serves both.
pub fn make_sphere3_at(
    name: impl ToString,
    arr: &[(impl HasToVec3F32, f32)],
    color: &Color,
    world_from_entity: Isometry3F64,
) -> SceneRenderable {
    make_ellipsoid3_at(
        name,
        arr.iter()
            .map(|(center, radius)| Ellipsoid3::sphere(center.to_vec3(), *radius as f64, *color))
            .collect(),
        world_from_entity,
    )
}

/// Spheres, at the origin of the scene.
pub fn make_sphere3(
    name: impl ToString,
    arr: &[(impl HasToVec3F32, f32)],
    color: &Color,
) -> SceneRenderable {
    make_sphere3_at(name, arr, color, Isometry3F64::identity())
}

/// make 3d points at a given pose
pub fn make_point3_at(
    name: impl ToString,
    arr: &[impl HasToVec3F32],
    color: &Color,
    point_size: f32,
    world_from_entity: Isometry3F64,
) -> SceneRenderable {
    let mut points = PointCloud3 {
        name: name.to_string(),
        points: vec![],
        world_from_entity,
    };

    for p in arr {
        points.points.push(Point3 {
            p: p.to_vec3(),
            color: *color,
            point_size,
        });
    }
    SceneRenderable::Point(points)
}

/// make 3d points at a given pose
pub fn make_point3(
    name: impl ToString,
    arr: &[impl HasToVec3F32],
    color: &Color,
    point_size: f32,
) -> SceneRenderable {
    make_point3_at(name, arr, color, point_size, Isometry3::identity())
}

/// makes 3d line segments at a given pose
pub fn make_line3_at(
    name: impl ToString,
    arr: &[[impl HasToVec3F32; 2]],
    color: &Color,
    line_width: f32,
    world_from_entity: Isometry3F64,
) -> SceneRenderable {
    let mut lines = LineSegments3 {
        name: name.to_string(),
        segments: vec![],
        world_from_entity,
    };

    for tuple in arr {
        lines.segments.push(LineSegment3 {
            p0: tuple[0].to_vec3(),
            p1: tuple[1].to_vec3(),
            color: *color,
            line_width,
        });
    }

    SceneRenderable::Line(lines)
}

/// makes 3d line segments
pub fn make_line3(
    name: impl ToString,
    arr: &[[impl HasToVec3F32; 2]],
    color: &Color,
    line_width: f32,
) -> SceneRenderable {
    make_line3_at(name, arr, color, line_width, Isometry3::identity())
}

/// make mesh
pub fn make_mesh3_at(
    name: impl ToString,
    arr: &[([impl HasToVec3F32; 3], Color)],
    world_from_entity: Isometry3F64,
) -> SceneRenderable {
    let mut mesh = TriangleMesh3 {
        name: name.to_string(),
        triangles: vec![],
        world_from_entity,
        wireframe: false,
    };

    for (trig, color) in arr {
        mesh.triangles.push(Triangle3 {
            p0: trig[0].to_vec3(),
            p1: trig[1].to_vec3(),
            p2: trig[2].to_vec3(),
            color0: *color,
            color1: *color,
            color2: *color,
        });
    }

    SceneRenderable::Mesh3(mesh)
}

/// make mesh
pub fn make_mesh3(name: impl ToString, arr: &[([impl HasToVec3F32; 3], Color)]) -> SceneRenderable {
    make_mesh3_at(name, arr, Isometry3::identity())
}

/// make 3d textured mesh at a given pose
pub fn make_textured_mesh3_at(
    name: impl ToString,
    arr: &[[(impl HasToVec3F32, impl HasToVec2F32); 3]],
    texture: ArcImage4U8,
    world_from_entity: Isometry3F64,
) -> SceneRenderable {
    let mut mesh = TexturedTriangleMesh3 {
        name: name.to_string(),
        triangles: vec![],
        texture,
        world_from_entity,
        wireframe: false,
    };

    for trig in arr {
        mesh.triangles.push(TexturedTriangle3 {
            p0: trig[0].0.to_vec3(),
            p1: trig[1].0.to_vec3(),
            p2: trig[2].0.to_vec3(),
            tex0: trig[0].1.to_vec2(),
            tex1: trig[1].1.to_vec2(),
            tex2: trig[2].1.to_vec2(),
        });
    }

    SceneRenderable::TexturedMesh3(mesh)
}

/// make 3d textured mesh
pub fn make_textured_mesh3(
    name: impl ToString,
    arr: &[[(impl HasToVec3F32, impl HasToVec2F32); 3]],
    texture: ArcImage4U8,
) -> SceneRenderable {
    make_textured_mesh3_at(name, arr, texture, Isometry3::identity())
}

/// 3D line
#[derive(Clone, Debug)]
pub struct LineSegment3 {
    /// Start point
    pub p0: SVec<f32, 3>,
    /// End point
    pub p1: SVec<f32, 3>,
    /// Color
    pub color: Color,
    /// Line width
    pub line_width: f32,
}

/// 3D point
#[derive(Clone, Debug)]
pub struct Point3 {
    /// Point
    pub p: SVec<f32, 3>,
    /// Color
    pub color: Color,
    /// Point size in pixels
    pub point_size: f32,
}

/// An ellipsoid of the scene.
///
/// Ellipsoids are traced rather than rasterized - intersected with the ray through each pixel -
/// so they are as round as the shape says at any distance and under any distortion, rather than
/// as round as the triangles spent on them. A sphere is one of these, see [Ellipsoid3::sphere].
#[derive(Clone, Debug)]
pub struct Ellipsoid3 {
    /// centre, in the entity's frame
    pub center: SVec<f32, 3>,
    /// The map taking the unit sphere to this ellipsoid, so its columns are the semi-axes. For a
    /// covariance, see [Ellipsoid3::from_covariance].
    pub shape: MatF64<3, 3>,
    /// colour
    pub color: Color,
}

impl Ellipsoid3 {
    /// A sphere of the given radius.
    pub fn sphere(center: SVec<f32, 3>, radius: f64, color: Color) -> Self {
        Ellipsoid3 {
            center,
            shape: MatF64::<3, 3>::identity() * radius,
            color,
        }
    }

    /// The `k`-sigma ellipsoid of a covariance: the set of points within a Mahalanobis distance
    /// of `k`, which is `{ c + k L u : |u| <= 1 }` for `sigma = L L^T`.
    ///
    /// Returns [None] when the covariance is not positive definite, which a degenerate one - a
    /// point seen from one bearing only, say - will not be.
    pub fn from_covariance(
        center: SVec<f32, 3>,
        covariance: MatF64<3, 3>,
        k: f64,
        color: Color,
    ) -> Option<Self> {
        let cholesky = covariance.cholesky()?;
        Some(Ellipsoid3 {
            center,
            shape: cholesky.l() * k,
            color,
        })
    }
}

/// 3D triangle
#[derive(Clone, Debug)]
pub struct Triangle3 {
    /// Vertex 0
    pub p0: SVec<f32, 3>,
    /// Vertex 1
    pub p1: SVec<f32, 3>,
    /// Vertex 2
    pub p2: SVec<f32, 3>,
    /// Triangle color vertex 0
    pub color0: Color,
    /// Triangle color vertex 1
    pub color1: Color,
    /// Triangle color vertex 2
    pub color2: Color,
}

impl Triangle3 {
    /// Create a new triangle
    pub fn new(p0: SVec<f32, 3>, p1: SVec<f32, 3>, p2: SVec<f32, 3>, color: Color) -> Self {
        Triangle3 {
            p0,
            p1,
            p2,
            color0: color,
            color1: color,
            color2: color,
        }
    }
}

/// 3D textured triangle
#[derive(Clone, Debug)]
pub struct TexturedTriangle3 {
    /// Vertex 0
    pub p0: SVec<f32, 3>,
    /// Vertex 1
    pub p1: SVec<f32, 3>,
    /// Vertex 2
    pub p2: SVec<f32, 3>,
    /// Texture coordinates for vertex 0
    pub tex0: SVec<f32, 2>,
    /// Texture coordinates for vertex 1
    pub tex1: SVec<f32, 2>,
    /// Texture coordinates for vertex 2
    pub tex2: SVec<f32, 2>,
}

/// Can be converted to Vec3F32
pub trait HasToVec3F32 {
    /// returns Vec3F32
    fn to_vec3(&self) -> SVec<f32, 3>;
}

impl HasToVec3F32 for [f32; 3] {
    fn to_vec3(&self) -> SVec<f32, 3> {
        SVec::<f32, 3>::new(self[0], self[1], self[2])
    }
}

impl HasToVec3F32 for &[f32; 3] {
    fn to_vec3(&self) -> SVec<f32, 3> {
        SVec::<f32, 3>::new(self[0], self[1], self[2])
    }
}

impl HasToVec3F32 for SVec<f32, 3> {
    fn to_vec3(&self) -> SVec<f32, 3> {
        *self
    }
}

/// 3D points
#[derive(Clone, Debug)]
pub struct PointCloud3 {
    /// Name of the entity
    pub name: String,
    /// List of points
    pub points: Vec<Point3>,
    /// world-anchored pose of the entity
    pub world_from_entity: Isometry3F64,
}

/// A capsule: every point within `radius` of the segment from `from` to `to`.
///
/// The shape to reach for when something needs thickness in the scene rather than on the screen -
/// a link of a kinematic chain, the shaft of an arrow, a line which should stay round as the
/// camera comes close. Its hemispherical ends mean two of them meeting at a joint show no seam.
#[derive(Clone, Debug)]
pub struct Capsule3 {
    /// one end of the segment, in the entity's frame
    pub from: SVec<f32, 3>,
    /// the other end
    pub to: SVec<f32, 3>,
    /// radius, in scene units
    pub radius: f32,
    /// colour
    pub color: Color,
    /// Cut the ends off flat, at the two points, rather than rounding them over.
    ///
    /// The intersection is the same either way - it is the cylinder about the segment, clipped to
    /// the segment's own extent - and the hemispheres are only what is put on the ends afterwards.
    /// Cut flat, the ends are open: see [make_cylinder3], which closes them with disks.
    pub flat_ends: bool,
}

/// A cylinder: the capsule's own body, cut flat at both ends and closed with a disk.
///
/// Traced like everything else here, so it is as round as its radius says at any distance and
/// under any distortion - which a tessellated one is only as round as the triangles spent on it.
#[derive(Clone, Debug)]
pub struct Cylinder3 {
    /// the centre of one end, in the entity's frame
    pub from: SVec<f32, 3>,
    /// the centre of the other
    pub to: SVec<f32, 3>,
    /// radius, in scene units
    pub radius: f32,
    /// colour
    pub color: Color,
}

/// capsules
#[derive(Clone, Debug)]
pub struct CapsuleCloud3 {
    /// Name of the entity
    pub name: String,
    /// List of capsules
    pub capsules: Vec<Capsule3>,
    /// Draw this entity as edges rather than as surfaces, whatever the rest of the scene does.
    /// The view has a wireframe toggle of its own, and either of the two turns this on.
    pub wireframe: bool,
    /// world-anchored pose of the entity
    pub world_from_entity: Isometry3F64,
}

/// A cone, from its apex along its axis. Only the curved surface: give it a disk for a base.
#[derive(Clone, Debug)]
pub struct Cone3 {
    /// the tip, in the entity's frame
    pub apex: SVec<f32, 3>,
    /// from the apex towards the base, its length being the height of the cone
    pub axis: SVec<f32, 3>,
    /// radius at the base
    pub radius: f32,
    /// colour
    pub color: Color,
}

/// cones
#[derive(Clone, Debug)]
pub struct ConeCloud3 {
    /// Name of the entity
    pub name: String,
    /// List of cones
    pub cones: Vec<Cone3>,
    /// Draw this entity as edges rather than as surfaces, whatever the rest of the scene does.
    /// The view has a wireframe toggle of its own, and either of the two turns this on.
    pub wireframe: bool,
    /// world-anchored pose of the entity
    pub world_from_entity: Isometry3F64,
}

/// Capsules at a pose of their own.
pub fn make_capsule3_at(
    name: impl ToString,
    capsules: Vec<Capsule3>,
    world_from_entity: Isometry3F64,
) -> SceneRenderable {
    SceneRenderable::Capsule(CapsuleCloud3 {
        name: name.to_string(),
        capsules,
        world_from_entity,
        wireframe: false,
    })
}

/// Capsules, at the origin of the scene.
pub fn make_capsule3(name: impl ToString, capsules: Vec<Capsule3>) -> SceneRenderable {
    make_capsule3_at(name, capsules, Isometry3F64::identity())
}

/// Cones at a pose of their own.
pub fn make_cone3_at(
    name: impl ToString,
    cones: Vec<Cone3>,
    world_from_entity: Isometry3F64,
) -> SceneRenderable {
    SceneRenderable::Cone(ConeCloud3 {
        name: name.to_string(),
        cones,
        world_from_entity,
        wireframe: false,
    })
}

/// Cones, at the origin of the scene.
pub fn make_cone3(name: impl ToString, cones: Vec<Cone3>) -> SceneRenderable {
    make_cone3_at(name, cones, Isometry3F64::identity())
}

/// An arrow from `from` to `to`: a capsule for the shaft, a cone for the head, and a disk to
/// close the head off.
///
/// Three primitives rather than one, since each of them already exists - and the vector this
/// draws is the thing a scene of poses and residuals is mostly made of.
pub fn make_arrow3_at(
    name: impl ToString,
    arrows: &[(SVec<f32, 3>, SVec<f32, 3>)],
    shaft_radius: f32,
    color: &Color,
    world_from_entity: Isometry3F64,
) -> Vec<SceneRenderable> {
    let name = name.to_string();
    let mut shafts = vec![];
    let mut heads = vec![];
    let mut bases = vec![];
    for (from, to) in arrows {
        let along = to - from;
        let length = along.norm();
        if length < 1e-9 {
            continue;
        }
        let direction = along / length;
        let (head_length, head_radius) = arrow_head_shape(shaft_radius, length);
        let base = to - direction * head_length;

        shafts.push(Capsule3 {
            from: *from,
            to: base,
            radius: shaft_radius,
            color: *color,
            flat_ends: false,
        });
        heads.push(Cone3 {
            apex: *to,
            axis: -direction * head_length,
            radius: head_radius,
            color: *color,
        });
        bases.push(Planar3::disk(base, direction, head_radius, *color));
    }
    vec![
        make_capsule3_at(format!("{name}-shafts"), shafts, world_from_entity),
        make_cone3_at(format!("{name}-heads"), heads, world_from_entity),
        make_planar3_at(format!("{name}-bases"), bases, world_from_entity),
    ]
}

/// Cylinders, as the sides and the two disks which close them.
///
/// Three entities rather than one, since a side is a capsule and a lid is a flat piece of a
/// plane, and the two are traced by different intersections.
pub fn make_cylinder3_at(
    name: impl ToString,
    cylinders: &[Cylinder3],
    world_from_entity: Isometry3F64,
) -> Vec<SceneRenderable> {
    let name = name.to_string();
    let mut sides = vec![];
    let mut lids = vec![];
    for cylinder in cylinders {
        let along = cylinder.to - cylinder.from;
        if along.norm() < 1e-9 {
            continue;
        }
        sides.push(Capsule3 {
            from: cylinder.from,
            to: cylinder.to,
            radius: cylinder.radius,
            color: cylinder.color,
            flat_ends: true,
        });
        let direction = along / along.norm();
        lids.push(Planar3::disk(
            cylinder.to,
            direction,
            cylinder.radius,
            cylinder.color,
        ));
        lids.push(Planar3::disk(
            cylinder.from,
            -direction,
            cylinder.radius,
            cylinder.color,
        ));
    }
    vec![
        make_capsule3_at(format!("{name}-sides"), sides, world_from_entity),
        make_planar3_at(format!("{name}-lids"), lids, world_from_entity),
    ]
}

/// Cylinders, at the origin of the scene.
pub fn make_cylinder3(name: impl ToString, cylinders: &[Cylinder3]) -> Vec<SceneRenderable> {
    make_cylinder3_at(name, cylinders, Isometry3F64::identity())
}

/// The head of an arrow, from the radius of its shaft and the length of the whole: long enough to
/// read as a head at a glance, and never so long that it swallows the shaft.
///
/// One place, so that an arrow and a set of axes cannot drift apart.
fn arrow_head_shape(shaft_radius: f32, length: f32) -> (f32, f32) {
    ((8.0 * shaft_radius).min(0.4 * length), 2.5 * shaft_radius)
}

/// Coordinate axes, drawn as arrows: x, y and z, with their tips red, green and blue.
///
/// The shafts are one colour - black by default, see [make_axes_arrows3] - so that the tips carry
/// the meaning and three of these next to each other do not become a thicket of colour.
pub fn make_axes_arrows3_at(
    name: impl ToString,
    length: f32,
    shaft_radius: f32,
    shaft_color: &Color,
    world_from_entity: Isometry3F64,
) -> Vec<SceneRenderable> {
    axes3(&[world_from_entity])
        .scale(length as f64)
        .shaft_radius(shaft_radius as f64)
        .shaft_color(*shaft_color)
        .build(name)
}

/// Coordinate axes with black shafts and red, green and blue tips, at a pose of their own.
pub fn make_axes_arrows3(
    name: impl ToString,
    length: f32,
    world_from_entity: Isometry3F64,
) -> Vec<SceneRenderable> {
    make_axes_arrows3_at(
        name,
        length,
        0.02 * length,
        &Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        },
        world_from_entity,
    )
}

/// Arrows at the origin of the scene.
pub fn make_arrow3(
    name: impl ToString,
    arrows: &[(SVec<f32, 3>, SVec<f32, 3>)],
    shaft_radius: f32,
    color: &Color,
) -> Vec<SceneRenderable> {
    make_arrow3_at(name, arrows, shaft_radius, color, Isometry3F64::identity())
}

/// What is ruled on a flat piece of a plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanarPattern {
    /// nothing: the plane is drawn in its own colour
    Plain,
    /// Lines, of `line_width` view-port pixels, with nothing between them drawn. Lines of a fixed
    /// width on screen crowd together as the plane recedes, so this fades out where it can no
    /// longer be resolved - leaving the plane invisible before its horizon.
    Grid,
    /// Squares, alternating between the colour and a darker shade of it. Being half covered
    /// whatever the scale, this needs no fade: it settles onto an even tone as it recedes, so the
    /// plane keeps its surface all the way to the horizon.
    Checker,
}

/// What bounds a flat piece of a plane.
///
/// The surface is the same in every case - a plane - and only the test applied to the hit point
/// differs, so one primitive serves all of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanarBound {
    /// the whole plane, unbounded
    Infinite,
    /// inside the ellipse whose semi-axes are the two in-plane vectors
    Ellipse,
    /// inside the rectangle whose half-extents are the two in-plane vectors
    Rectangle,
}

/// A flat piece of a plane, traced rather than rasterized.
///
/// A disk cannot be tessellated exactly at any triangle count, and an unbounded plane cannot be
/// rasterized at all without choosing a size - so both are traced, and the rectangle comes along
/// with them for the cost of a comparison.
#[derive(Clone, Debug)]
pub struct Planar3 {
    /// centre, in the entity's frame
    pub center: SVec<f32, 3>,
    /// the two in-plane vectors: semi-axes of the ellipse, or half-extents of the rectangle
    pub axes: [SVec<f32, 3>; 2],
    /// what bounds it
    pub bound: PlanarBound,
    /// Width of the outline in view-port pixels, or zero to fill it. An outlined ellipse is a
    /// ring, which is how a covariance ellipsoid is drawn without hiding the scene behind it.
    /// With a grid, this is the width of the grid lines instead.
    pub line_width: f32,
    /// what is ruled on it
    pub pattern: PlanarPattern,
    /// size of one period of that pattern, in the units of its axes
    pub pattern_scale: f32,
    /// The other colour of a checkerboard - the squares which are not [Self::color]. Unused by
    /// the other patterns.
    pub pattern_color: Color,
    /// colour
    pub color: Color,
}

impl Planar3 {
    /// A disk of the given radius, facing along `normal`.
    pub fn disk(center: SVec<f32, 3>, normal: SVec<f32, 3>, radius: f32, color: Color) -> Self {
        let (u, v) = Self::basis_for(normal);
        Planar3 {
            center,
            axes: [u * radius, v * radius],
            bound: PlanarBound::Ellipse,
            line_width: 0.0,
            pattern: PlanarPattern::Plain,
            pattern_scale: 0.0,
            pattern_color: color,
            color,
        }
    }

    /// An ellipse with the given semi-axes, which need not be orthogonal.
    pub fn ellipse(center: SVec<f32, 3>, axes: [SVec<f32, 3>; 2], color: Color) -> Self {
        Planar3 {
            center,
            axes,
            bound: PlanarBound::Ellipse,
            line_width: 0.0,
            pattern: PlanarPattern::Plain,
            pattern_scale: 0.0,
            pattern_color: color,
            color,
        }
    }

    /// A rectangle with the given half-extents.
    pub fn rectangle(center: SVec<f32, 3>, axes: [SVec<f32, 3>; 2], color: Color) -> Self {
        Planar3 {
            center,
            axes,
            bound: PlanarBound::Rectangle,
            line_width: 0.0,
            pattern: PlanarPattern::Plain,
            pattern_scale: 0.0,
            pattern_color: color,
            color,
        }
    }

    /// The whole plane through `center`, facing along `normal`. The axes carry its scale, which
    /// nothing uses until it is given a grid.
    pub fn plane(center: SVec<f32, 3>, normal: SVec<f32, 3>, color: Color) -> Self {
        let (u, v) = Self::basis_for(normal);
        Planar3 {
            center,
            axes: [u, v],
            bound: PlanarBound::Infinite,
            line_width: 0.0,
            pattern: PlanarPattern::Plain,
            pattern_scale: 0.0,
            pattern_color: color,
            color,
        }
    }

    /// Outlined rather than filled, with the width in view-port pixels.
    pub fn outlined(mut self, line_width: f32) -> Self {
        self.line_width = line_width;
        self
    }

    /// Ruled with a grid of the given spacing, in the units of its axes, drawn in lines of
    /// `line_width` view-port pixels. Nothing between the lines is drawn.
    ///
    /// The pattern follows the plane's own axes, so it is square when they are.
    pub fn with_grid(mut self, spacing: f32, line_width: f32) -> Self {
        self.pattern = PlanarPattern::Grid;
        self.pattern_scale = spacing;
        self.line_width = line_width;
        self
    }

    /// Ruled with a checkerboard of the given square size, in the units of its axes, alternating
    /// between this plane's colour and `other`.
    ///
    /// The one to reach for on a ground plane: half of it is covered whatever the scale, so it
    /// settles onto an even tone as it recedes rather than crowding into a sheet, and the plane
    /// stays a surface all the way to its horizon.
    pub fn with_checker(mut self, size: f32, other: Color) -> Self {
        self.pattern = PlanarPattern::Checker;
        self.pattern_scale = size;
        self.pattern_color = other;
        self
    }

    /// Some pair of unit vectors spanning the plane with this normal.
    fn basis_for(normal: SVec<f32, 3>) -> (SVec<f32, 3>, SVec<f32, 3>) {
        let n = normal.normalize();
        // the axis the normal leans on least, so the cross product is well conditioned
        let away = match n[0].abs() < n[1].abs() && n[0].abs() < n[2].abs() {
            true => SVec::<f32, 3>::new(1.0, 0.0, 0.0),
            false => match n[1].abs() < n[2].abs() {
                true => SVec::<f32, 3>::new(0.0, 1.0, 0.0),
                false => SVec::<f32, 3>::new(0.0, 0.0, 1.0),
            },
        };
        let u = n.cross(&away).normalize();
        (u, n.cross(&u))
    }
}

/// flat pieces of a plane
#[derive(Clone, Debug)]
pub struct PlanarCloud3 {
    /// Name of the entity
    pub name: String,
    /// List of them
    pub planars: Vec<Planar3>,
    /// Draw this entity as edges rather than as surfaces, whatever the rest of the scene does.
    /// The view has a wireframe toggle of its own, and either of the two turns this on.
    pub wireframe: bool,
    /// world-anchored pose of the entity
    pub world_from_entity: Isometry3F64,
}

/// Flat pieces of a plane, at a pose of their own.
pub fn make_planar3_at(
    name: impl ToString,
    planars: Vec<Planar3>,
    world_from_entity: Isometry3F64,
) -> SceneRenderable {
    SceneRenderable::Planar(PlanarCloud3 {
        name: name.to_string(),
        planars,
        world_from_entity,
        wireframe: false,
    })
}

/// Flat pieces of a plane, at the origin of the scene.
pub fn make_planar3(name: impl ToString, planars: Vec<Planar3>) -> SceneRenderable {
    make_planar3_at(name, planars, Isometry3F64::identity())
}

/// 3D ellipsoids
#[derive(Clone, Debug)]
pub struct EllipsoidCloud3 {
    /// Name of the entity
    pub name: String,
    /// List of ellipsoids
    pub ellipsoids: Vec<Ellipsoid3>,
    /// Draw this entity as edges rather than as surfaces, whatever the rest of the scene does.
    /// The view has a wireframe toggle of its own, and either of the two turns this on.
    pub wireframe: bool,
    /// world-anchored pose of the entity
    pub world_from_entity: Isometry3F64,
}

/// 3D lines
#[derive(Clone, Debug)]
pub struct LineSegments3 {
    /// Name of the entity
    pub name: String,
    /// List of segments
    pub segments: Vec<LineSegment3>,
    /// world-anchored pose of the entity
    pub world_from_entity: Isometry3F64,
}

/// 3D mesh
#[derive(Clone, Debug)]
pub struct TriangleMesh3 {
    /// Name of the entity
    pub name: String,
    /// List of triangles
    pub triangles: Vec<Triangle3>,
    /// Draw this entity as edges rather than as surfaces, whatever the rest of the scene does.
    /// The view has a wireframe toggle of its own, and either of the two turns this on.
    pub wireframe: bool,
    /// world-anchored pose of the entity
    pub world_from_entity: Isometry3F64,
}

/// 3D textured mesh
#[derive(Clone, Debug)]
pub struct TexturedTriangleMesh3 {
    /// Name of the entity
    pub name: String,
    /// List of textured triangles
    pub triangles: Vec<TexturedTriangle3>,
    /// Texture the triangles are mapped to
    pub texture: ArcImage4U8,
    /// Draw this entity as edges rather than as surfaces, whatever the rest of the scene does.
    /// The view has a wireframe toggle of its own, and either of the two turns this on.
    pub wireframe: bool,
    /// world-anchored pose of the entity
    pub world_from_entity: Isometry3F64,
}
