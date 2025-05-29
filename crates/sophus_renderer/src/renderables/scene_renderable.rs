mod axes;

pub use axes::*;
use sophus_autodiff::linalg::SVec;
use sophus_lie::{Isometry3, Isometry3F64};

use crate::{
    prelude::*,
    renderables::{color::Color, pixel_renderable::HasToVec2F32},
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
}

impl SceneRenderable {
    /// Get scene from entity
    pub fn world_from_entity(&self) -> Isometry3F64 {
        match self {
            SceneRenderable::Line(lines) => lines.world_from_entity,
            SceneRenderable::Point(points) => points.world_from_entity,
            SceneRenderable::Mesh3(mesh) => mesh.world_from_entity,
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
    };

    SceneRenderable::Mesh3(mesh)
}

/// creates a named mesh
pub fn named_mesh3(name: impl ToString, mesh: TriangleMesh3) -> SceneRenderable {
    named_mesh3_at(name, mesh, Isometry3::identity())
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
    world_from_entity: Isometry3F64,
) -> TexturedTriangleMesh3 {
    let mut mesh = TexturedTriangleMesh3 {
        name: name.to_string(),
        triangles: vec![],
        world_from_entity,
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
    /// world-anchored pose of the entity
    pub world_from_entity: Isometry3F64,
}
