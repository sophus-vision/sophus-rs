use crate::preludes::*;
use crate::renderables::color::Color;
use crate::renderables::scene_renderable::LineSegment3;
use sophus_core::linalg::VecF64;
use sophus_lie::Isometry3F64;

/// opaque axes builder type.
///
/// To be used with the following functions:
///
/// - `axes3`
/// - `axis3`
/// - `make_axes3`
/// - `make_axis3`
pub struct Axes3Builder {
    world_from_local_axes: Vec<Isometry3F64>,
    scale: f64,
    line_width: f32,
}

impl Axes3Builder {
    fn new(world_from_local: &[Isometry3F64]) -> Self {
        Self {
            world_from_local_axes: world_from_local.to_vec(),
            scale: 1.0,
            line_width: 5.0,
        }
    }

    /// Set the scale of the axes
    pub fn scale(&mut self, scale: f64) -> &mut Self {
        self.scale = scale;
        self
    }

    /// Set the line width of the axes
    pub fn line_width(&mut self, line_width: f32) -> &mut Self {
        self.line_width = line_width;
        self
    }

    /// Build the axes
    pub fn build(&self) -> Vec<LineSegment3> {
        let zero_in_local = VecF64::<3>::zeros();
        let x_axis_local = VecF64::<3>::new(self.scale, 0.0, 0.0);
        let y_axis_local = VecF64::<3>::new(0.0, self.scale, 0.0);
        let z_axis_local = VecF64::<3>::new(0.0, 0.0, self.scale);

        let mut lines = vec![];

        for world_from_local in self.world_from_local_axes.iter() {
            let zero_in_world = world_from_local.transform(&zero_in_local);
            let axis_x_in_world = world_from_local.transform(&x_axis_local);
            let axis_y_in_world = world_from_local.transform(&y_axis_local);
            let axis_z_in_world = world_from_local.transform(&z_axis_local);

            lines.push(LineSegment3 {
                p0: zero_in_world.cast(),
                p1: axis_x_in_world.cast(),
                color: Color {
                    r: 1.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
                line_width: self.line_width,
            });
            lines.push(LineSegment3 {
                p0: zero_in_world.cast(),
                p1: axis_y_in_world.cast(),
                color: Color {
                    r: 0.0,
                    g: 1.0,
                    b: 0.0,
                    a: 1.0,
                },
                line_width: self.line_width,
            });
            lines.push(LineSegment3 {
                p0: zero_in_world.cast(),
                p1: axis_z_in_world.cast(),
                color: Color {
                    r: 0.0,
                    g: 0.0,
                    b: 1.0,
                    a: 1.0,
                },
                line_width: self.line_width,
            });
        }

        lines
    }
}

/// Make 3d axes, with default scale of 1.0 and line width of 5.0
pub fn make_axes3(world_from_local_axes: &[Isometry3F64]) -> Vec<LineSegment3> {
    Axes3Builder::new(world_from_local_axes).build()
}

/// Create an 3d axes builder
///
/// Example:
///
/// ```
/// use sophus_renderer::renderables::scene_renderable::axes::axes3;
/// use sophus_lie::Isometry3F64;
///
/// let axes = axes3(&[Isometry3F64::identity()]).scale(0.5).line_width(3.0).build();
/// ```
pub fn axes3(world_from_local_axes: &[Isometry3F64]) -> Axes3Builder {
    Axes3Builder::new(world_from_local_axes)
}

/// Make a single 3d axis, with default scale of 1.0 and line width of 5.0
pub fn make_axis3(world_from_local: Isometry3F64) -> Vec<LineSegment3> {
    make_axes3(&[world_from_local])
}

/// Create a single 3d axis builder
pub fn axis3(world_from_local: Isometry3F64) -> Axes3Builder {
    Axes3Builder::new(&[world_from_local])
}
