use sophus_autodiff::linalg::SVec;

use crate::{
    prelude::*,
    renderables::color::Color,
};

/// pixel renderable
#[derive(Clone, Debug)]
pub enum PixelRenderable {
    /// 2D line segments
    Line(LineSegments2),
    /// 2D point cloud
    Point(PointCloud2),
}

/// named line segments
pub fn named_line2(name: impl ToString, segments: Vec<LineSegment2>) -> PixelRenderable {
    PixelRenderable::Line(LineSegments2 {
        name: name.to_string(),
        segments,
    })
}

/// named point cloud
pub fn named_point2(name: impl ToString, points: Vec<Point2>) -> PixelRenderable {
    PixelRenderable::Point(PointCloud2 {
        name: name.to_string(),
        points,
    })
}

/// make lines 2d
pub fn make_line2(
    name: impl ToString,
    arr: &[[impl HasToVec2F32; 2]],
    color: &Color,
    line_width: f32,
) -> PixelRenderable {
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

    PixelRenderable::Line(line_segments)
}

/// make 2d point cloud
pub fn make_point2(
    name: impl ToString,
    arr: &[impl HasToVec2F32],
    color: &Color,
    point_size: f32,
) -> PixelRenderable {
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
    PixelRenderable::Point(cloud)
}

/// Can be converted to Vec2F32
pub trait HasToVec2F32 {
    /// returns Vec2F32
    fn to_vec2(&self) -> SVec<f32, 2>;
}

impl HasToVec2F32 for [f32; 2] {
    fn to_vec2(&self) -> SVec<f32, 2> {
        SVec::<f32, 2>::new(self[0], self[1])
    }
}

impl HasToVec2F32 for &[f32; 2] {
    fn to_vec2(&self) -> SVec<f32, 2> {
        SVec::<f32, 2>::new(self[0], self[1])
    }
}

impl HasToVec2F32 for SVec<f32, 2> {
    fn to_vec2(&self) -> SVec<f32, 2> {
        *self
    }
}

/// 2D line
#[derive(Clone, Debug)]
pub struct LineSegment2 {
    /// Start point
    pub p0: SVec<f32, 2>,
    /// End point
    pub p1: SVec<f32, 2>,
    /// Color
    pub color: Color,
    /// Line width
    pub line_width: f32,
}

/// 2D point
#[derive(Clone, Debug)]
pub struct Point2 {
    /// Point
    pub p: SVec<f32, 2>,
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
