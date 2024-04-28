use sophus_image::arc_image::ArcImage4F32;
use sophus_image::arc_image::ArcImage4U8;

use nalgebra::SVector;

/// Color with alpha channel
#[derive(Copy, Clone, Debug, Default)]
pub struct Color {
    /// Red channel
    pub r: f32,
    /// Green channel
    pub g: f32,
    /// Blue channel
    pub b: f32,
    /// Alpha channel
    pub a: f32,
}

/// Renderable entity
#[derive(Clone, Debug)]
pub enum Renderable {
    /// 2D lines
    Lines2(Lines2),
    /// 2D points
    Points2(Points2),
    /// 3D lines
    Lines3(Lines3),
    /// 3D points
    Points3(Points3),
    /// 3D mesh
    Mesh3(Mesh3),
    /// 3D textured mesh
    TexturedMesh3(TexturedMesh3),
    /// Background image
    BackgroundImage(ArcImage4U8),
}

/// 2D lines
#[derive(Clone, Debug)]
pub struct Lines2 {
    /// Name of the entity
    pub name: String,
    /// List of lines
    pub lines: Vec<Line2>,
}

/// 2D points
#[derive(Clone, Debug)]
pub struct Points2 {
    /// Name of the entity
    pub name: String,
    /// List of points
    pub points: Vec<Point2>,
}

/// 3D lines
#[derive(Clone, Debug)]
pub struct Lines3 {
    /// Name of the entity
    pub name: String,
    /// List of lines
    pub lines: Vec<Line3>,
}

/// 3D points
#[derive(Clone, Debug)]
pub struct Points3 {
    /// Name of the entity
    pub name: String,
    /// List of points
    pub points: Vec<Point3>,
}

/// 3D mesh
#[derive(Clone, Debug)]
pub struct Mesh3 {
    /// Name of the entity
    pub name: String,
    /// List of triangles
    pub mesh: Vec<Triangle3>,
}

/// 3D textured mesh
#[derive(Clone, Debug)]
pub struct TexturedMesh3 {
    /// Name of the entity
    pub name: String,
    /// List of textured triangles
    pub mesh: Vec<TexturedTriangle3>,
    /// Texture image
    pub texture: ArcImage4F32,
}

/// 2D line
#[derive(Clone, Debug)]
pub struct Line2 {
    /// Start point
    pub p0: SVector<f32, 2>,
    /// End point
    pub p1: SVector<f32, 2>,
    /// Color
    pub color: Color,
    /// Line width
    pub line_width: f32,
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

/// 2D point
#[derive(Clone, Debug)]
pub struct Point2 {
    /// Point
    pub p: SVector<f32, 2>,
    /// Color
    pub color: Color,
    /// Point size in pixels
    pub point_size: f32,
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
    /// Triangle color
    pub color: Color,
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
