use nalgebra::SVector;
use sophus_core::linalg::VecF64;
use sophus_lie::Isometry3;
use sophus_lie::Isometry3F64;

use crate::renderables::color::Color;
use crate::renderables::renderable2d::HasToVec2F32;
use crate::simple_viewer::ViewerCamera;

/// View3d renderable
#[derive(Clone, Debug)]
pub enum Renderable3d {
    /// 3D line segments
    Line(LineSegments3),
    /// 3D points
    Point(PointCloud3),
    /// 3D mesh
    Mesh3(TriangleMesh3),
}

impl Renderable3d {
    /// Get scene from entity
    pub fn scene_from_entity(&self) -> Isometry3F64 {
        match self {
            Renderable3d::Line(lines) => lines.scene_from_entity,
            Renderable3d::Point(points) => points.scene_from_entity,
            Renderable3d::Mesh3(mesh) => mesh.scene_from_entity,
        }
    }
}

/// make 3d points at a given pose
pub fn make_point3_at(
    name: impl ToString,
    arr: &[impl HasToVec3F32],
    color: &Color,
    point_size: f32,
    scene_from_entity: Isometry3F64,
) -> Renderable3d {
    let mut points = PointCloud3 {
        name: name.to_string(),
        points: vec![],
        scene_from_entity,
    };

    for p in arr {
        points.points.push(Point3 {
            p: p.to_vec3(),
            color: *color,
            point_size,
        });
    }
    Renderable3d::Point(points)
}

/// make 3d points at a given pose
pub fn make_point3(
    name: impl ToString,
    arr: &[impl HasToVec3F32],
    color: &Color,
    point_size: f32,
) -> Renderable3d {
    make_point3_at(name, arr, color, point_size, Isometry3::identity())
}

/// makes 3d line segments at a given pose
pub fn make_line3_at(
    name: impl ToString,
    arr: &[[impl HasToVec3F32; 2]],
    color: &Color,
    line_width: f32,
    scene_from_entity: Isometry3F64,
) -> Renderable3d {
    let mut lines = LineSegments3 {
        name: name.to_string(),
        segments: vec![],
        scene_from_entity,
    };

    for tuple in arr {
        lines.segments.push(Line3 {
            p0: tuple[0].to_vec3(),
            p1: tuple[1].to_vec3(),
            color: *color,
            line_width,
        });
    }

    Renderable3d::Line(lines)
}

/// makes 3d line segments
pub fn make_line3(
    name: impl ToString,
    arr: &[[impl HasToVec3F32; 2]],
    color: &Color,
    line_width: f32,
) -> Renderable3d {
    make_line3_at(name, arr, color, line_width, Isometry3::identity())
}

/// make mesh
pub fn make_mesh3_at(
    name: impl ToString,
    arr: &[([impl HasToVec3F32; 3], Color)],
    scene_from_entity: Isometry3F64,
) -> Renderable3d {
    let mut mesh = TriangleMesh3 {
        name: name.to_string(),
        triangles: vec![],
        scene_from_entity,
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

    Renderable3d::Mesh3(mesh)
}

/// make mesh
pub fn make_mesh3(name: impl ToString, arr: &[([impl HasToVec3F32; 3], Color)]) -> Renderable3d {
    make_mesh3_at(name, arr, Isometry3::identity())
}

/// make 3d textured mesh at a given pose
pub fn make_textured_mesh3_at(
    name: impl ToString,
    arr: &[[(impl HasToVec3F32, impl HasToVec2F32); 3]],
    scene_from_entity: Isometry3F64,
) -> TexturedTriangleMesh3 {
    let mut mesh = TexturedTriangleMesh3 {
        name: name.to_string(),
        triangles: vec![],
        scene_from_entity,
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

    mesh
}

/// make 3d textured mesh
pub fn make_textured_mesh3(
    name: impl ToString,
    arr: &[[(impl HasToVec3F32, impl HasToVec2F32); 3]],
) -> TexturedTriangleMesh3 {
    make_textured_mesh3_at(name, arr, Isometry3::identity())
}

/// Packet of view3d renderables
#[derive(Clone, Debug)]
pub struct View3dPacket {
    /// List of 3d renderables
    pub renderables3d: Vec<Renderable3d>,
    /// Name of the view
    pub view_label: String,
    /// Initial camera, ignored if not the first packet of "view_name"
    pub initial_camera: ViewerCamera,
}

/// 3D line
#[derive(Clone, Debug)]
pub struct Line3 {
    /// Start point
    pub p0: SVector<f32, 3>,
    /// End point
    pub p1: SVector<f32, 3>,
    /// Color
    pub color: Color,
    /// Line width
    pub line_width: f32,
}

/// 3D point
#[derive(Clone, Debug)]
pub struct Point3 {
    /// Point
    pub p: SVector<f32, 3>,
    /// Color
    pub color: Color,
    /// Point size in pixels
    pub point_size: f32,
}

/// 3D triangle
#[derive(Clone, Debug)]
pub struct Triangle3 {
    /// Vertex 0
    pub p0: SVector<f32, 3>,
    /// Vertex 1
    pub p1: SVector<f32, 3>,
    /// Vertex 2
    pub p2: SVector<f32, 3>,
    /// Triangle color vertex 0
    pub color0: Color,
    /// Triangle color vertex 1
    pub color1: Color,
    /// Triangle color vertex 2
    pub color2: Color,
}

/// 3D textured triangle
#[derive(Clone, Debug)]
pub struct TexturedTriangle3 {
    /// Vertex 0
    pub p0: SVector<f32, 3>,
    /// Vertex 1
    pub p1: SVector<f32, 3>,
    /// Vertex 2
    pub p2: SVector<f32, 3>,
    /// Texture coordinates for vertex 0
    pub tex0: SVector<f32, 2>,
    /// Texture coordinates for vertex 1
    pub tex1: SVector<f32, 2>,
    /// Texture coordinates for vertex 2
    pub tex2: SVector<f32, 2>,
}

/// Can be converted to Vec3F32
pub trait HasToVec3F32 {
    /// returns Vec3F32
    fn to_vec3(&self) -> SVector<f32, 3>;
}

impl HasToVec3F32 for [f32; 3] {
    fn to_vec3(&self) -> SVector<f32, 3> {
        SVector::<f32, 3>::new(self[0], self[1], self[2])
    }
}

impl HasToVec3F32 for &[f32; 3] {
    fn to_vec3(&self) -> SVector<f32, 3> {
        SVector::<f32, 3>::new(self[0], self[1], self[2])
    }
}

impl HasToVec3F32 for SVector<f32, 3> {
    fn to_vec3(&self) -> SVector<f32, 3> {
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
    /// scene-anchored pose of the entity
    pub scene_from_entity: Isometry3F64,
}

/// 3D lines
#[derive(Clone, Debug)]
pub struct LineSegments3 {
    /// Name of the entity
    pub name: String,
    /// List of segments
    pub segments: Vec<Line3>,
    /// scene-anchored pose of the entity
    pub scene_from_entity: Isometry3F64,
}

/// 3D mesh
#[derive(Clone, Debug)]
pub struct TriangleMesh3 {
    /// Name of the entity
    pub name: String,
    /// List of triangles
    pub triangles: Vec<Triangle3>,
    /// scene-anchored pose of the entity
    pub scene_from_entity: Isometry3F64,
}

/// 3D textured mesh
#[derive(Clone, Debug)]
pub struct TexturedTriangleMesh3 {
    /// Name of the entity
    pub name: String,
    /// List of textured triangles
    pub triangles: Vec<TexturedTriangle3>,
    /// scene-anchored pose of the entity
    pub scene_from_entity: Isometry3F64,
}

/// Make axis
pub fn make_axis(world_from_local: Isometry3F64, scale: f64) -> Vec<Line3> {
    let zero_in_local = VecF64::<3>::zeros();
    let x_axis_local = VecF64::<3>::new(scale, 0.0, 0.0);
    let y_axis_local = VecF64::<3>::new(0.0, scale, 0.0);
    let z_axis_local = VecF64::<3>::new(0.0, 0.0, scale);

    let mut lines = vec![];

    let zero_in_world = world_from_local.transform(&zero_in_local);
    let axis_x_in_world = world_from_local.transform(&x_axis_local);
    let axis_y_in_world = world_from_local.transform(&y_axis_local);
    let axis_z_in_world = world_from_local.transform(&z_axis_local);

    lines.push(Line3 {
        p0: zero_in_world.cast(),
        p1: axis_x_in_world.cast(),
        color: Color {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        },
        line_width: 2.0,
    });
    lines.push(Line3 {
        p0: zero_in_world.cast(),
        p1: axis_y_in_world.cast(),
        color: Color {
            r: 0.0,
            g: 1.0,
            b: 0.0,
            a: 1.0,
        },
        line_width: 2.0,
    });
    lines.push(Line3 {
        p0: zero_in_world.cast(),
        p1: axis_z_in_world.cast(),
        color: Color {
            r: 0.0,
            g: 0.0,
            b: 1.0,
            a: 1.0,
        },
        line_width: 2.0,
    });

    lines
}
