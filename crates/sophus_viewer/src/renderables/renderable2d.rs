use nalgebra::SVector;

use crate::renderables::color::Color;
use crate::renderables::frame::Frame;
use crate::renderables::renderable3d::Renderable3d;

/// View3d renderable
#[derive(Clone, Debug)]
pub enum Renderable2d {
    /// 2D line segments
    Line(LineSegments2),
    /// 2D point cloud
    Point(PointCloud2),
}

/// make lines 2d
pub fn make_line2(
    name: impl ToString,
    arr: &[[impl HasToVec2F32; 2]],
    color: &Color,
    line_width: f32,
) -> Renderable2d {
    let mut line_segments = LineSegments2 {
        name: name.to_string(),
        segments: vec![],
    };

    for tuple in arr {
        line_segments.segments.push(LineSegment2 {
            p0: tuple[0].to_vec2(),
            p1: tuple[1].to_vec2(),
            color: *color,
            line_width,
        });
    }

    Renderable2d::Line(line_segments)
}

/// make 2d point cloud
pub fn make_point2(
    name: impl ToString,
    arr: &[impl HasToVec2F32],
    color: &Color,
    point_size: f32,
) -> Renderable2d {
    let mut cloud = PointCloud2 {
        name: name.to_string(),
        points: vec![],
    };

    for p in arr {
        cloud.points.push(Point2 {
            p: p.to_vec2(),
            color: *color,
            point_size,
        });
    }
    Renderable2d::Point(cloud)
}

/// Packet of image renderables
#[derive(Clone, Debug)]
pub struct View2dPacket {
    /// Frame to hold content
    ///
    ///  1. For each `view_label`, content (i.e. renderables2d, renderables3d) will be added to
    ///     the existing frame. If no frame exists yet, e.g. frame was always None for `view_label`,
    ///     the content is ignored.
    ///  2. If we have a new frame, that is `frame == Some(...)`, all previous content is deleted, but
    ///     content from this packet will be added.
    pub frame: Option<Frame>,
    /// List of 2d renderables
    pub renderables2d: Vec<Renderable2d>,
    /// List of 3d renderables
    pub renderables3d: Vec<Renderable3d>,
    /// Name of the view
    pub view_label: String,
}

/// Can be converted to Vec2F32
pub trait HasToVec2F32 {
    /// returns Vec2F32
    fn to_vec2(&self) -> SVector<f32, 2>;
}

impl HasToVec2F32 for [f32; 2] {
    fn to_vec2(&self) -> SVector<f32, 2> {
        SVector::<f32, 2>::new(self[0], self[1])
    }
}

impl HasToVec2F32 for &[f32; 2] {
    fn to_vec2(&self) -> SVector<f32, 2> {
        SVector::<f32, 2>::new(self[0], self[1])
    }
}

impl HasToVec2F32 for SVector<f32, 2> {
    fn to_vec2(&self) -> SVector<f32, 2> {
        *self
    }
}

/// 2D line
#[derive(Clone, Debug)]
pub struct LineSegment2 {
    /// Start point
    pub p0: SVector<f32, 2>,
    /// End point
    pub p1: SVector<f32, 2>,
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

/// 2D line segments
#[derive(Clone, Debug)]
pub struct LineSegments2 {
    /// Name of the entity
    pub name: String,
    /// List of line segments
    pub segments: Vec<LineSegment2>,
}

/// 2D point cloud
#[derive(Clone, Debug)]
pub struct PointCloud2 {
    /// Name of the entity
    pub name: String,
    /// List of points
    pub points: Vec<Point2>,
}
