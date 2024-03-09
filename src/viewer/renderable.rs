use nalgebra::SVector;

use crate::image::arc_image::ArcImage4F32;
use crate::image::arc_image::ArcImage4U8;

#[derive(Copy, Clone, Debug, Default)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[derive(Clone, Debug)]
pub enum Renderable {
    Lines2(Lines2),
    Points2(Points2),
    Lines3(Lines3),
    Points3(Points3),
    Mesh3(Mesh3),
    TexturedMesh3(TexturedMesh3),
    BackgroundImage(ArcImage4U8),
}

#[derive(Clone, Debug)]
pub struct Lines2 {
    pub name: String,
    pub lines: Vec<Line2>,
}

#[derive(Clone, Debug)]
pub struct Points2 {
    pub name: String,
    pub points: Vec<Point2>,
}

#[derive(Clone, Debug)]
pub struct Lines3 {
    pub name: String,
    pub lines: Vec<Line3>,
}

#[derive(Clone, Debug)]
pub struct Points3 {
    pub name: String,
    pub points: Vec<Point3>,
}

#[derive(Clone, Debug)]
pub struct Mesh3 {
    pub name: String,
    pub mesh: Vec<Triangle3>,
}

#[derive(Clone, Debug)]
pub struct TexturedMesh3 {
    pub name: String,
    pub mesh: Vec<TexturedTriangle3>,
    pub texture: ArcImage4F32,
}

#[derive(Clone, Debug)]
pub struct Line2 {
    pub p0: SVector<f32, 2>,
    pub p1: SVector<f32, 2>,
    pub color: Color,
    pub line_width: f32,
}

#[derive(Clone, Debug)]
pub struct Line3 {
    pub p0: SVector<f32, 3>,
    pub p1: SVector<f32, 3>,
    pub color: Color,
    pub line_width: f32,
}

#[derive(Clone, Debug)]
pub struct Point2 {
    pub p: SVector<f32, 2>,
    pub color: Color,
    pub point_size: f32,
}

#[derive(Clone, Debug)]
pub struct Point3 {
    pub p: SVector<f32, 3>,
    pub color: Color,
    pub point_size: f32,
}

#[derive(Clone, Debug)]
pub struct Triangle3 {
    pub p0: SVector<f32, 3>,
    pub p1: SVector<f32, 3>,
    pub p2: SVector<f32, 3>,
    pub color: Color,
}

#[derive(Clone, Debug)]
pub struct TexturedTriangle3 {
    pub p0: SVector<f32, 3>,
    pub p1: SVector<f32, 3>,
    pub p2: SVector<f32, 3>,
    pub tex0: SVector<f32, 2>,
    pub tex1: SVector<f32, 2>,
    pub tex2: SVector<f32, 2>,
}
