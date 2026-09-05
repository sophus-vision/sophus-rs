use sophus_autodiff::linalg::{
    SVec,
    VecF64,
};
use sophus_lie::Isometry3F64;

use super::{
    Capsule3,
    Cone3,
    Planar3,
    SceneRenderable,
    arrow_head_shape,
    make_capsule3,
    make_cone3,
    make_planar3,
};
use crate::{
    prelude::*,
    renderables::color::Color,
};

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
    shaft_radius: Option<f64>,
    shaft_color: Color,
    tip_colors: [Color; 3],
    wireframe: bool,
}

impl Axes3Builder {
    // use axes3(..) or axis3(..) for public API
    fn new(world_from_local: &[Isometry3F64]) -> Self {
        Self {
            world_from_local_axes: world_from_local.to_vec(),
            scale: 1.0,
            shaft_radius: None,
            shaft_color: Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
            tip_colors: [Color::red(), Color::green(), Color::blue()],
            wireframe: false,
        }
    }

    /// Set the length of each axis
    pub fn scale(&mut self, scale: f64) -> &mut Self {
        self.scale = scale;
        self
    }

    /// Set the radius of the shafts, in scene units. Defaults to a fiftieth of the length, which
    /// is what keeps a set of axes readable across the range of scales the demos use.
    pub fn shaft_radius(&mut self, shaft_radius: f64) -> &mut Self {
        self.shaft_radius = Some(shaft_radius);
        self
    }

    /// Set the colour of the shafts, black by default. This and [Self::wireframe] are what tell
    /// two sets in one scene apart - an estimate from the ground truth it is converging on -
    /// without spending the tips, which are what says which axis is which.
    pub fn shaft_color(&mut self, shaft_color: Color) -> &mut Self {
        self.shaft_color = shaft_color;
        self
    }

    /// Paint the whole set - shafts and tips alike - in one colour, rather than marking x, y and
    /// z red, green and blue.
    ///
    /// For a set which stands for one thing rather than for a frame whose axes have to be told
    /// apart - the pose of a body among others of its kind. Where the axes themselves still
    /// matter, separate the sets with [Self::shaft_color] and [Self::wireframe] instead.
    pub fn color(&mut self, color: Color) -> &mut Self {
        self.shaft_color = color;
        self.tip_colors = [color; 3];
        self
    }

    /// Draw the axes as edges rather than as surfaces, see [SceneRenderable::wireframe].
    pub fn wireframe(&mut self, wireframe: bool) -> &mut Self {
        self.wireframe = wireframe;
        self
    }

    /// Build the axes, as three entities named after `name`.
    ///
    /// All the poses go into those same three - one cloud of shafts, one of tips and one of the
    /// disks closing the tips off - rather than three entities each: a set is often hundreds of
    /// poses, and the renderer keeps a table entry, and the tracing loop a pass, per entity.
    pub fn build(&self, name: impl ToString) -> Vec<SceneRenderable> {
        let name = name.to_string();
        let length = self.scale as f32;
        let shaft_radius = self.shaft_radius.unwrap_or(0.02 * self.scale) as f32;
        let (head_length, head_radius) = arrow_head_shape(shaft_radius, length);

        let mut shafts = vec![];
        let mut tips = vec![];
        let mut tip_bases = vec![];

        for world_from_local in self.world_from_local_axes.iter() {
            let origin: SVec<f32, 3> = world_from_local.translation().cast();

            for axis in 0..3 {
                let mut unit_in_local = VecF64::<3>::zeros();
                unit_in_local[axis] = 1.0;
                // the primitives are emitted in world coordinates, so the entity holding them
                // needs no pose of its own and every pose of the set can share it
                let direction: SVec<f32, 3> =
                    world_from_local.rotation().transform(unit_in_local).cast();
                let base = origin + direction * (length - head_length);

                shafts.push(Capsule3 {
                    from: origin,
                    to: base,
                    radius: shaft_radius,
                    color: self.shaft_color,
                    flat_ends: false,
                });
                tips.push(Cone3 {
                    apex: origin + direction * length,
                    axis: -direction * head_length,
                    radius: head_radius,
                    color: self.tip_colors[axis],
                });
                tip_bases.push(Planar3::disk(
                    base,
                    direction,
                    head_radius,
                    self.tip_colors[axis],
                ));
            }
        }

        vec![
            make_capsule3(format!("{name}-shafts"), shafts).wireframe(self.wireframe),
            make_cone3(format!("{name}-tips"), tips).wireframe(self.wireframe),
            make_planar3(format!("{name}-tip-bases"), tip_bases).wireframe(self.wireframe),
        ]
    }
}

/// Make 3d axes, with a default scale of 1.0
pub fn make_axes3(
    name: impl ToString,
    world_from_local_axes: &[Isometry3F64],
) -> Vec<SceneRenderable> {
    Axes3Builder::new(world_from_local_axes).build(name)
}

/// Create an 3d axes builder
///
/// The axes are drawn as arrows of traced capsules and cones, so they have a thickness in the
/// scene rather than on the screen: they grow as the camera comes close, and stay round under any
/// distortion.
///
/// Example:
///
/// ```
/// use sophus_lie::Isometry3F64;
/// use sophus_renderer::renderables::axes3;
///
/// let axes = axes3(&[Isometry3F64::identity()]).scale(0.5).build("poses");
/// ```
pub fn axes3(world_from_local_axes: &[Isometry3F64]) -> Axes3Builder {
    Axes3Builder::new(world_from_local_axes)
}

/// Make a single 3d axis, with a default scale of 1.0
pub fn make_axis3(name: impl ToString, world_from_local: Isometry3F64) -> Vec<SceneRenderable> {
    make_axes3(name, &[world_from_local])
}

/// Create a single 3d axis builder
pub fn axis3(world_from_local: Isometry3F64) -> Axes3Builder {
    Axes3Builder::new(&[world_from_local])
}
