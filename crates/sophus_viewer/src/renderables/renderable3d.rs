use nalgebra::SVector;
use sophus_core::linalg::VecF64;
use sophus_lie::Isometry3;

use crate::renderables::color::Color;
use crate::renderables::renderable2d::HasToVec2F32;
use crate::simple_viewer::ViewerCamera;

/// View3d renderable
#[derive(Clone, Debug)]
pub enum Renderable3d {
    /// 3D lines
    Lines3(Lines3),
    /// 3D points
    Points3(Points3),
    /// 3D mesh
    Mesh3(Mesh3),
}

/// Create 3d points
pub fn make_lines3d(name: &str, viz_lines_3d: Vec<Line3>) -> Renderable3d {
    Renderable3d::Lines3(Lines3 {
        name: name.to_owned(),
        lines: viz_lines_3d,
    })
}

/// Create 3d points
pub fn make_points3d(name: &str, viz_points_3d: Vec<Point3>) -> Renderable3d {
    Renderable3d::Points3(Points3 {
        name: name.to_owned(),
        points: viz_points_3d,
    })
}

/// Create 3d mesh
pub fn make_mesh3d(name: &str, viz_mesh_3d: Vec<Triangle3>) -> Renderable3d {
    Renderable3d::Mesh3(Mesh3 {
        name: name.to_owned(),
        mesh: viz_mesh_3d,
    })
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
pub struct Points3 {
    /// Name of the entity
    pub name: String,
    /// List of points
    pub points: Vec<Point3>,
}

impl Points3 {
    /// make points 3d
    pub fn make(
        name: impl ToString,
        arr: &[impl HasToVec3F32],
        color: &Color,
        point_size: f32,
    ) -> Self {
        let mut points = Points3 {
            name: name.to_string(),
            points: vec![],
        };

        for p in arr {
            points.points.push(Point3 {
                p: p.to_vec3(),
                color: *color,
                point_size,
            });
        }
        points
    }
}

/// 3D lines
#[derive(Clone, Debug)]
pub struct Lines3 {
    /// Name of the entity
    pub name: String,
    /// List of lines
    pub lines: Vec<Line3>,
}

impl Lines3 {
    /// make lines 3d
    pub fn make(
        name: impl ToString,
        arr: &[[impl HasToVec3F32; 2]],
        color: &Color,
        line_width: f32,
    ) -> Self {
        let mut lines = Lines3 {
            name: name.to_string(),
            lines: vec![],
        };

        for tuple in arr {
            lines.lines.push(Line3 {
                p0: tuple[0].to_vec3(),
                p1: tuple[1].to_vec3(),
                color: *color,
                line_width,
            });
        }

        lines
    }
}

/// 3D mesh
#[derive(Clone, Debug)]
pub struct Mesh3 {
    /// Name of the entity
    pub name: String,
    /// List of triangles
    pub mesh: Vec<Triangle3>,
}

impl Mesh3 {
    /// make mesh
    pub fn make(name: impl ToString, arr: &[([impl HasToVec3F32; 3], Color)]) -> Self {
        let mut mesh = Mesh3 {
            name: name.to_string(),
            mesh: vec![],
        };

        for (trig, color) in arr {
            mesh.mesh.push(Triangle3 {
                p0: trig[0].to_vec3(),
                p1: trig[1].to_vec3(),
                p2: trig[2].to_vec3(),
                color: *color,
            });
        }

        mesh
    }
}

/// 3D textured mesh
#[derive(Clone, Debug)]
pub struct TexturedMesh3 {
    /// List of textured triangles
    pub mesh: Vec<TexturedTriangle3>,
}

impl TexturedMesh3 {
    /// make textured mesh
    pub fn make(arr: &[[(impl HasToVec3F32, impl HasToVec2F32); 3]]) -> Self {
        let mut mesh = TexturedMesh3 { mesh: vec![] };

        for trig in arr {
            mesh.mesh.push(TexturedTriangle3 {
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
}

/// Make axis
pub fn make_axis(world_from_local: Isometry3<f64, 1>, scale: f64) -> Vec<Line3> {
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
