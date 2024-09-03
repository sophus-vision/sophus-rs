use nalgebra::SVector;

use crate::renderables::color::Color;
use crate::renderables::frame::Frame;
use crate::renderables::renderable3d::Renderable3d;

/// View3d renderable
#[derive(Clone, Debug)]
pub enum Renderable2d {
    /// 2D lines
    Lines2(Lines2),
    /// 2D points
    Points2(Points2),
}

/// Create 2d lines
pub fn make_lines2d(name: &str, viz_lines_2d: Vec<Line2>) -> Renderable2d {
    Renderable2d::Lines2(Lines2 {
        name: name.to_owned(),
        lines: viz_lines_2d,
    })
}

/// Create 2d points  
pub fn make_points2d(name: &str, viz_points_2d: Vec<Point2>) -> Renderable2d {
    Renderable2d::Points2(Points2 {
        name: name.to_owned(),
        points: viz_points_2d,
    })
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

/// 2D lines
#[derive(Clone, Debug)]
pub struct Lines2 {
    /// Name of the entity
    pub name: String,
    /// List of lines
    pub lines: Vec<Line2>,
}

impl Lines2 {
    /// make lines 2d
    pub fn make(
        name: impl ToString,
        arr: &[[impl HasToVec2F32; 2]],
        color: &Color,
        line_width: f32,
    ) -> Self {
        let mut lines = Lines2 {
            name: name.to_string(),
            lines: vec![],
        };

        for tuple in arr {
            lines.lines.push(Line2 {
                p0: tuple[0].to_vec2(),
                p1: tuple[1].to_vec2(),
                color: *color,
                line_width,
            });
        }

        lines
    }
}

/// 2D points
#[derive(Clone, Debug)]
pub struct Points2 {
    /// Name of the entity
    pub name: String,
    /// List of points
    pub points: Vec<Point2>,
}

impl Points2 {
    /// make points 2d
    pub fn make(
        name: impl ToString,
        arr: &[impl HasToVec2F32],
        color: &Color,
        point_size: f32,
    ) -> Self {
        let mut points = Points2 {
            name: name.to_string(),
            points: vec![],
        };

        for p in arr {
            points.points.push(Point2 {
                p: p.to_vec2(),
                color: *color,
                point_size,
            });
        }
        points
    }
}
